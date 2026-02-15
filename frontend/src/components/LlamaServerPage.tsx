/**
 * llama.cpp Server Manager Page
 *
 * Provides UI for managing a local llama-server process:
 * - Start/stop server with configurable parameters
 * - Browse and select local GGUF models
 * - Download recommended models from HuggingFace
 * - Configuration presets (save/load/apply)
 * - Real-time server logs viewer
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  Cpu,
  RefreshCw,
  Play,
  Square,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Loader2,
  AlertCircle,
  Download,
  Check,
  Copy,
  Trash2,
  Eye,
  EyeOff,
  Save,
  AlertTriangle,
  HardDrive,
  Settings,
  Terminal,
  BookOpen,
  FolderOpen
} from 'lucide-react'
import {
  startServer,
  stopServer,
  resetServer,
  getServerStatus,
  getServerLogs,
  listLocalModels,
  downloadModel,
  getDownloadProgress,
  getPresets,
  savePreset,
  deletePreset,
  DEFAULT_CONFIG,
  type LlamaServerConfig,
  type ServerStatusResponse,
  type GGUFModelInfo,
  type DownloadProgress,
  type ServerPreset,
  type RecommendedModel
} from '../api/llamacpp'

export function LlamaServerPage() {
  const navigate = useNavigate()

  // --- Server status state ---
  const [serverStatus, setServerStatus] = useState<ServerStatusResponse | null>(null)
  const [isLoadingStatus, setIsLoadingStatus] = useState(true)

  // --- Configuration state ---
  const [config, setConfig] = useState<LlamaServerConfig>({ ...DEFAULT_CONFIG })

  // --- Collapsible sections ---
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    basic: true,
    performance: false,
    features: false,
    kvcache: false,
    advanced: false,
  })

  // --- Local models ---
  const [localModels, setLocalModels] = useState<GGUFModelInfo[]>([])
  const [modelsDir, setModelsDir] = useState<string>('')
  const [isLoadingModels, setIsLoadingModels] = useState(false)

  // --- Presets ---
  const [presetsList, setPresetsList] = useState<ServerPreset[]>([])
  const [recommendedModels, setRecommendedModels] = useState<RecommendedModel[]>([])
  const [selectedPreset, setSelectedPreset] = useState<string>('')
  const [savePresetName, setSavePresetName] = useState('')
  const [savePresetDesc, setSavePresetDesc] = useState('')
  const [isSavingPreset, setIsSavingPreset] = useState(false)
  const [showSavePresetForm, setShowSavePresetForm] = useState(false)

  // --- Model library tab ---
  const [modelLibraryTab, setModelLibraryTab] = useState<'downloaded' | 'recommended'>('downloaded')

  // --- Download state ---
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)

  // --- Server logs ---
  const [logs, setLogs] = useState<string[]>([])
  const logsContainerRef = useRef<HTMLDivElement>(null)
  const [logsCopied, setLogsCopied] = useState(false)

  // --- API key visibility ---
  const [showApiKey, setShowApiKey] = useState(false)

  // --- GPU layers custom mode ---
  const [gpuLayersMode, setGpuLayersMode] = useState<'auto' | 'all' | 'custom'>(
    DEFAULT_CONFIG.n_gpu_layers === 'auto' ? 'auto' : DEFAULT_CONFIG.n_gpu_layers === '-1' ? 'all' : 'custom'
  )
  const [gpuLayersCustom, setGpuLayersCustom] = useState<number>(0)

  // --- Flash attention mode ---
  const [flashAttnMode, setFlashAttnMode] = useState<'auto' | 'on' | 'off'>(
    DEFAULT_CONFIG.flash_attn === 'auto' ? 'auto' : DEFAULT_CONFIG.flash_attn === 'on' ? 'on' : 'off'
  )

  // --- Action states ---
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [isResetting, setIsResetting] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)

  // ==================== Data Fetching ====================

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getServerStatus()
      setServerStatus(status)
      return status
    } catch (error) {
      console.error('Failed to fetch server status:', error)
      return null
    }
  }, [])

  const fetchModels = useCallback(async () => {
    setIsLoadingModels(true)
    try {
      const result = await listLocalModels()
      setLocalModels(result.models)
      setModelsDir(result.models_dir)
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setIsLoadingModels(false)
    }
  }, [])

  const fetchPresets = useCallback(async () => {
    try {
      const result = await getPresets()
      setPresetsList(result.presets)
      setRecommendedModels(result.recommended_models)
    } catch (error) {
      console.error('Failed to fetch presets:', error)
    }
  }, [])

  const fetchLogs = useCallback(async () => {
    try {
      const result = await getServerLogs(500)
      setLogs(result.logs)
    } catch (error) {
      console.error('Failed to fetch logs:', error)
    }
  }, [])

  const refreshAll = useCallback(async () => {
    setIsLoadingStatus(true)
    await Promise.all([fetchStatus(), fetchModels(), fetchPresets()])
    setIsLoadingStatus(false)
  }, [fetchStatus, fetchModels, fetchPresets])

  // ==================== On Mount ====================

  useEffect(() => {
    refreshAll()
  }, [refreshAll])

  // ==================== Status Polling (every 3 seconds) ====================

  useEffect(() => {
    const interval = setInterval(() => {
      fetchStatus()
    }, 3000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  // ==================== Log Polling (every 2 seconds when running/starting/error) ====================

  useEffect(() => {
    // Poll logs when server is running, starting, or in error state (to show crash logs)
    if (
      serverStatus?.status !== 'running' &&
      serverStatus?.status !== 'starting' &&
      serverStatus?.status !== 'error'
    ) return

    const interval = setInterval(() => {
      fetchLogs()
    }, 2000)

    // Initial fetch
    fetchLogs()

    return () => clearInterval(interval)
  }, [serverStatus?.status, fetchLogs])

  // ==================== Auto-scroll logs ====================

  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight
    }
  }, [logs])

  // ==================== Download Progress Polling (every 1.5 seconds when downloading) ====================

  useEffect(() => {
    if (!isDownloading) return

    const interval = setInterval(async () => {
      try {
        const progress = await getDownloadProgress()
        setDownloadProgress(progress)

        if (progress.status === 'completed') {
          setIsDownloading(false)
          setDownloadProgress(null)
          setDownloadError(null)
          // Refresh models list
          fetchModels()
        } else if (progress.status === 'error' || progress.status === 'failed') {
          setIsDownloading(false)
          setDownloadError(progress.error_message || 'Download failed')
        }
      } catch (error) {
        console.error('Failed to get download progress:', error)
      }
    }, 1500)

    return () => clearInterval(interval)
  }, [isDownloading, fetchModels])

  // ==================== Sync GPU layers & flash attention with config ====================

  useEffect(() => {
    if (config.n_gpu_layers === 'auto') {
      setGpuLayersMode('auto')
    } else if (config.n_gpu_layers === '-1') {
      setGpuLayersMode('all')
    } else {
      setGpuLayersMode('custom')
      setGpuLayersCustom(parseInt(config.n_gpu_layers) || 0)
    }
  }, [config.n_gpu_layers])

  useEffect(() => {
    if (config.flash_attn === 'auto') {
      setFlashAttnMode('auto')
    } else if (config.flash_attn === 'on' || config.flash_attn === 'true') {
      setFlashAttnMode('on')
    } else {
      setFlashAttnMode('off')
    }
  }, [config.flash_attn])

  // ==================== Handlers ====================

  const handleStartServer = async () => {
    if (!config.model_path) {
      setActionError('Please select a model before starting the server.')
      return
    }
    if (!window.confirm('Start the llama-server with the current configuration?')) return

    setIsStarting(true)
    setActionError(null)
    try {
      await startServer(config)
      await fetchStatus()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Failed to start server')
    } finally {
      setIsStarting(false)
    }
  }

  const handleStopServer = async () => {
    if (!window.confirm('Stop the running llama-server?')) return

    setIsStopping(true)
    setActionError(null)
    try {
      await stopServer()
      await fetchStatus()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Failed to stop server')
    } finally {
      setIsStopping(false)
    }
  }

  const handleResetServer = async () => {
    setIsResetting(true)
    setActionError(null)
    try {
      await resetServer()
      await fetchStatus()
      // Also fetch logs so user can see crash info
      await fetchLogs()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Failed to reset server')
    } finally {
      setIsResetting(false)
    }
  }

  const handleApplyPreset = () => {
    const preset = presetsList.find((p) => p.name === selectedPreset)
    if (preset) {
      setConfig({ ...preset.config })
    }
  }

  const handleSavePreset = async () => {
    if (!savePresetName.trim()) return
    setIsSavingPreset(true)
    try {
      await savePreset(savePresetName.trim(), savePresetDesc.trim(), config)
      await fetchPresets()
      setSavePresetName('')
      setSavePresetDesc('')
      setShowSavePresetForm(false)
    } catch (error) {
      console.error('Failed to save preset:', error)
    } finally {
      setIsSavingPreset(false)
    }
  }

  const handleDeletePreset = async (name: string) => {
    if (!window.confirm(`Delete preset "${name}"?`)) return
    try {
      await deletePreset(name)
      await fetchPresets()
      if (selectedPreset === name) setSelectedPreset('')
    } catch (error) {
      console.error('Failed to delete preset:', error)
    }
  }

  const handleDownloadModel = async (repoId: string, filename: string) => {
    setIsDownloading(true)
    setDownloadError(null)
    setDownloadProgress(null)
    try {
      const result = await downloadModel({ repo_id: repoId, filename })
      if (result.already_exists) {
        setIsDownloading(false)
        fetchModels()
      }
    } catch (error) {
      setIsDownloading(false)
      setDownloadError(error instanceof Error ? error.message : 'Failed to start download')
    }
  }

  const handleSelectModel = (filepath: string) => {
    setConfig((prev) => ({ ...prev, model_path: filepath }))
  }

  const handleCopyLogs = () => {
    navigator.clipboard.writeText(logs.join('\n'))
    setLogsCopied(true)
    setTimeout(() => setLogsCopied(false), 2000)
  }

  const handleClearLogs = () => {
    setLogs([])
  }

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }))
  }

  const updateConfig = <K extends keyof LlamaServerConfig>(key: K, value: LlamaServerConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }

  const handleGpuLayersModeChange = (mode: 'auto' | 'all' | 'custom') => {
    setGpuLayersMode(mode)
    if (mode === 'auto') {
      updateConfig('n_gpu_layers', 'auto')
    } else if (mode === 'all') {
      updateConfig('n_gpu_layers', '-1')
    } else {
      updateConfig('n_gpu_layers', String(gpuLayersCustom))
    }
  }

  const handleGpuLayersCustomChange = (value: number) => {
    setGpuLayersCustom(value)
    updateConfig('n_gpu_layers', String(value))
  }

  const handleFlashAttnChange = (mode: 'auto' | 'on' | 'off') => {
    setFlashAttnMode(mode)
    updateConfig('flash_attn', mode)
  }

  // ==================== Helpers ====================

  const getStatusDotColor = (status: string | undefined) => {
    switch (status) {
      case 'running':
        return 'bg-green-500'
      case 'stopped':
        return 'bg-red-500'
      case 'starting':
      case 'stopping':
        return 'bg-yellow-500'
      case 'error':
        return 'bg-orange-500'
      default:
        return 'bg-gray-400'
    }
  }

  const getStatusText = (status: string | undefined) => {
    switch (status) {
      case 'running':
        return 'Running'
      case 'stopped':
        return 'Stopped'
      case 'starting':
        return 'Starting...'
      case 'stopping':
        return 'Stopping...'
      case 'error':
        return 'Error'
      default:
        return 'Unknown'
    }
  }

  const getCategoryBadgeColor = (category: string) => {
    switch (category.toLowerCase()) {
      case 'general':
        return 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'
      case 'coding':
        return 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300'
      case 'reasoning':
        return 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300'
      case 'small':
        return 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300'
      case 'embedding':
        return 'bg-pink-100 dark:bg-pink-900/40 text-pink-700 dark:text-pink-300'
      default:
        return 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
    }
  }

  const formatEta = (seconds: number | undefined) => {
    if (!seconds || seconds <= 0) return '--'
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
  }

  const inputClasses =
    'w-full px-3 py-2 border border-light-border dark:border-dark-border rounded-lg bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text focus:outline-none focus:ring-2 focus:ring-primary'

  const selectClasses = inputClasses

  const labelClasses = 'block text-sm font-medium text-light-text dark:text-dark-text mb-1'

  const cacheTypeOptions = ['f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1']

  // ==================== Memory Estimation ====================

  /**
   * Estimate total GPU memory usage for the selected model + KV cache.
   * Returns an object with model size, KV cache estimate, total, and warning level.
   */
  const getMemoryEstimate = () => {
    const selectedModel = localModels.find((m) => m.filepath === config.model_path)
    if (!selectedModel) return null

    const modelSizeGB = selectedModel.size_bytes / (1024 * 1024 * 1024)

    // Estimate KV cache size based on context size and cache type
    // Formula: 2 * n_layers * n_kv_heads * head_dim * ctx_size * bytes_per_element * n_parallel
    // Simplified: use ratio relative to known models
    // For 70B models: ~80 layers, 8 KV heads, 128 head_dim
    // KV cache per token (f16) = 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes ≈ 0.3125 MB
    // For q8_0: roughly half, for q4_0: roughly quarter

    const kvBytesPerToken: Record<string, number> = {
      'f16': 0.3125,    // MB per token (for 70B-class models)
      'bf16': 0.3125,
      'q8_0': 0.15625,
      'q4_0': 0.078125,
      'q4_1': 0.078125,
      'iq4_nl': 0.078125,
      'q5_0': 0.09765,
      'q5_1': 0.09765,
    }

    // Scale KV estimate based on model size (smaller models = less KV cache)
    const modelScale = modelSizeGB < 5 ? 0.05 : modelSizeGB < 20 ? 0.3 : 1.0

    const avgKvBytes = ((kvBytesPerToken[config.cache_type_k] || 0.3125) +
                        (kvBytesPerToken[config.cache_type_v] || 0.3125)) / 2

    const ctxSize = config.ctx_size > 0 ? config.ctx_size : (modelSizeGB > 30 ? 32768 : 8192)
    const parallel = config.parallel || 1

    const kvCacheSizeGB = (avgKvBytes * ctxSize * parallel * modelScale) / 1024

    const totalGB = modelSizeGB + kvCacheSizeGB
    const maxMemoryGB = 52 // ~64GB - macOS overhead

    let warning: 'safe' | 'tight' | 'danger' = 'safe'
    if (totalGB > maxMemoryGB) {
      warning = 'danger'
    } else if (totalGB > maxMemoryGB - 4) {
      warning = 'tight'
    }

    return {
      modelSizeGB: Math.round(modelSizeGB * 10) / 10,
      kvCacheSizeGB: Math.round(kvCacheSizeGB * 10) / 10,
      totalGB: Math.round(totalGB * 10) / 10,
      ctxSizeUsed: ctxSize,
      warning,
    }
  }

  const memoryEstimate = getMemoryEstimate()

  // ==================== Render ====================

  return (
    <div className="min-h-screen bg-light-bg dark:bg-dark-bg">
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-light-sidebar dark:bg-dark-sidebar border-b border-light-border dark:border-dark-border">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 hover:bg-light-border dark:hover:bg-dark-border rounded-lg transition-colors"
              aria-label="Back to home"
            >
              <ArrowLeft className="w-5 h-5 text-light-text dark:text-dark-text" />
            </button>
            <div className="flex items-center gap-2">
              <Cpu className="w-6 h-6 text-primary" />
              <h1 className="text-xl font-semibold text-light-text dark:text-dark-text">
                llama.cpp Server
              </h1>
            </div>
          </div>
          <button
            onClick={refreshAll}
            disabled={isLoadingStatus}
            className="p-2 hover:bg-light-border dark:hover:bg-dark-border rounded-lg transition-colors disabled:opacity-50"
            aria-label="Refresh"
          >
            <RefreshCw
              className={`w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary ${
                isLoadingStatus ? 'animate-spin' : ''
              }`}
            />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        {/* ==================== Card 1: Server Status ==================== */}
        <div className="bg-light-sidebar dark:bg-dark-sidebar rounded-xl border border-light-border dark:border-dark-border overflow-hidden">
          <div className="px-4 py-3 border-b border-light-border dark:border-dark-border">
            <h2 className="text-lg font-semibold text-light-text dark:text-dark-text flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              Server Status
            </h2>
          </div>
          <div className="p-4">
            {isLoadingStatus && !serverStatus ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
              </div>
            ) : (
              <div className="space-y-4">
                {/* Status row */}
                <div className="flex flex-wrap items-center gap-6">
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-3 h-3 rounded-full ${getStatusDotColor(serverStatus?.status)} ${
                        serverStatus?.status === 'starting' || serverStatus?.status === 'stopping'
                          ? 'animate-pulse'
                          : ''
                      }`}
                    />
                    <span className="text-lg font-medium text-light-text dark:text-dark-text">
                      {getStatusText(serverStatus?.status)}
                    </span>
                  </div>

                  {serverStatus?.pid && (
                    <div className="text-sm text-light-text-secondary dark:text-dark-text-secondary">
                      PID: <span className="font-mono">{serverStatus.pid}</span>
                    </div>
                  )}

                  {serverStatus?.uptime_pretty && (
                    <div className="text-sm text-light-text-secondary dark:text-dark-text-secondary">
                      Uptime: {serverStatus.uptime_pretty}
                    </div>
                  )}

                  {serverStatus?.loaded_model && (
                    <div className="text-sm text-light-text-secondary dark:text-dark-text-secondary">
                      Model: <span className="font-mono text-primary">{serverStatus.loaded_model}</span>
                    </div>
                  )}

                  {serverStatus?.port && (
                    <div className="text-sm text-light-text-secondary dark:text-dark-text-secondary">
                      Port: <span className="font-mono">{serverStatus.port}</span>
                    </div>
                  )}
                </div>

                {/* Binary status */}
                {serverStatus && !serverStatus.binary_found && (
                  <div className="flex items-start gap-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                    <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-yellow-700 dark:text-yellow-300">
                        llama-server binary not found
                      </p>
                      <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-1">
                        Expected at: {serverStatus.binary_path || 'unknown path'}
                      </p>
                    </div>
                  </div>
                )}

                {/* Model mismatch warning */}
                {serverStatus?.status === 'running' && (() => {
                  try {
                    const saved = localStorage.getItem('rag-settings')
                    const llmModel = saved ? JSON.parse(saved).llm_model || 'gpt-4o' : 'gpt-4o'
                    if (!llmModel.startsWith('llamacpp:')) {
                      return (
                        <div className="flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                          <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                          <div>
                            <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
                              Settings use <code className="bg-amber-100 dark:bg-amber-900/40 px-1 py-0.5 rounded text-xs">{llmModel}</code> — not this llama.cpp server
                            </p>
                            <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                              The model <strong>{serverStatus?.loaded_model}</strong> is loaded in memory but RAG queries go to a different provider. Go to Settings to switch, or stop the server to free RAM.
                            </p>
                            <button
                              onClick={() => navigate('/settings')}
                              className="mt-2 text-sm text-amber-700 dark:text-amber-300 hover:underline font-medium flex items-center gap-1"
                            >
                              <Settings className="w-3.5 h-3.5" />
                              Go to Settings →
                            </button>
                          </div>
                        </div>
                      )
                    }
                  } catch { /* ignore */ }
                  return null
                })()}

                {/* Error message */}
                {serverStatus?.error_message && (
                  <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-red-700 dark:text-red-300">
                        {serverStatus.error_message}
                      </p>
                      {serverStatus.status === 'error' && (
                        <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                          Click <strong>Reset</strong> to clear the error, then adjust parameters (try reducing context size or using quantized KV cache) and retry.
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Action error */}
                {actionError && (
                  <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-red-700 dark:text-red-300">{actionError}</p>
                  </div>
                )}

                {/* Memory Estimation (shown when server is not running) */}
                {memoryEstimate && serverStatus?.status !== 'running' && (
                  <div className={`flex items-start gap-2 p-3 rounded-lg border ${
                    memoryEstimate.warning === 'danger'
                      ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                      : memoryEstimate.warning === 'tight'
                        ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
                        : 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800'
                  }`}>
                    {memoryEstimate.warning === 'danger' ? (
                      <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    ) : memoryEstimate.warning === 'tight' ? (
                      <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                    ) : (
                      <HardDrive className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
                    )}
                    <div className="text-sm">
                      <p className={`font-medium ${
                        memoryEstimate.warning === 'danger'
                          ? 'text-red-700 dark:text-red-300'
                          : memoryEstimate.warning === 'tight'
                            ? 'text-yellow-700 dark:text-yellow-300'
                            : 'text-blue-700 dark:text-blue-300'
                      }`}>
                        Stima memoria: ~{memoryEstimate.totalGB} GB
                        {memoryEstimate.warning === 'danger' && ' ⚠️ Rischio OOM!'}
                        {memoryEstimate.warning === 'tight' && ' — al limite'}
                      </p>
                      <p className={`text-xs mt-0.5 ${
                        memoryEstimate.warning === 'danger'
                          ? 'text-red-600 dark:text-red-400'
                          : memoryEstimate.warning === 'tight'
                            ? 'text-yellow-600 dark:text-yellow-400'
                            : 'text-blue-600 dark:text-blue-400'
                      }`}>
                        Modello: {memoryEstimate.modelSizeGB} GB + KV cache ({config.cache_type_k}): ~{memoryEstimate.kvCacheSizeGB} GB
                        {config.ctx_size === 0 && ` (ctx: ${memoryEstimate.ctxSizeUsed} default)`}
                        {config.ctx_size > 0 && ` (ctx: ${config.ctx_size})`}
                      </p>
                      {memoryEstimate.warning === 'danger' && (
                        <p className="text-xs text-red-600 dark:text-red-400 mt-1">
                          💡 Riduci il context size o usa KV cache quantizzato (q8_0/q4_0). Applica il preset "M1 Ultra Optimized".
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Buttons */}
                <div className="flex items-center gap-3">
                  {serverStatus?.status === 'error' ? (
                    <>
                      {/* When in ERROR state, show Reset + Retry buttons */}
                      <button
                        onClick={handleResetServer}
                        disabled={isResetting}
                        className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        {isResetting ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <RefreshCw className="w-4 h-4" />
                        )}
                        Reset
                      </button>
                      <button
                        onClick={handleStartServer}
                        disabled={isStarting || !serverStatus?.binary_found}
                        className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        {isStarting ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        Retry Start
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={handleStartServer}
                        disabled={
                          isStarting ||
                          serverStatus?.status === 'running' ||
                          serverStatus?.status === 'starting' ||
                          !serverStatus?.binary_found
                        }
                        className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        {isStarting ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        Start
                      </button>

                      <button
                        onClick={handleStopServer}
                        disabled={
                          isStopping ||
                          serverStatus?.status === 'stopped' ||
                          serverStatus?.status === 'stopping'
                        }
                        className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        {isStopping ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Square className="w-4 h-4" />
                        )}
                        Stop
                      </button>
                    </>
                  )}

                  {serverStatus?.status === 'running' && serverStatus.host && serverStatus.port && (
                    <a
                      href={`http://${serverStatus.host}:${serverStatus.port}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover transition-colors flex items-center gap-2"
                    >
                      <ExternalLink className="w-4 h-4" />
                      Open API
                    </a>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ==================== Card 2: Configuration ==================== */}
        <div className="bg-light-sidebar dark:bg-dark-sidebar rounded-xl border border-light-border dark:border-dark-border overflow-hidden">
          <div className="px-4 py-3 border-b border-light-border dark:border-dark-border">
            <h2 className="text-lg font-semibold text-light-text dark:text-dark-text flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Configuration
            </h2>
          </div>
          <div className="p-4 space-y-4">
            {/* Preset selector */}
            <div className="flex flex-col sm:flex-row items-start sm:items-end gap-3">
              <div className="flex-1 w-full">
                <label className={labelClasses}>Preset</label>
                <select
                  value={selectedPreset}
                  onChange={(e) => setSelectedPreset(e.target.value)}
                  className={selectClasses}
                >
                  <option value="">-- Select a preset --</option>
                  {presetsList.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.name} {p.builtin ? '(built-in)' : ''}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex gap-2 flex-shrink-0">
                <button
                  onClick={handleApplyPreset}
                  disabled={!selectedPreset}
                  className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Check className="w-4 h-4" />
                  Apply
                </button>
                {selectedPreset &&
                  presetsList.find((p) => p.name === selectedPreset && !p.builtin) && (
                    <button
                      onClick={() => handleDeletePreset(selectedPreset)}
                      className="px-3 py-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors flex items-center gap-1"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                <button
                  onClick={() => setShowSavePresetForm(!showSavePresetForm)}
                  className="px-4 py-2 border border-light-border dark:border-dark-border rounded-lg hover:bg-light-hover dark:hover:bg-dark-hover transition-colors text-light-text dark:text-dark-text flex items-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  Save Preset
                </button>
              </div>
            </div>

            {/* Save preset form */}
            {showSavePresetForm && (
              <div className="p-4 bg-light-bg dark:bg-dark-bg rounded-lg border border-light-border dark:border-dark-border space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className={labelClasses}>Preset Name</label>
                    <input
                      type="text"
                      value={savePresetName}
                      onChange={(e) => setSavePresetName(e.target.value)}
                      placeholder="My Config"
                      className={inputClasses}
                    />
                  </div>
                  <div>
                    <label className={labelClasses}>Description</label>
                    <input
                      type="text"
                      value={savePresetDesc}
                      onChange={(e) => setSavePresetDesc(e.target.value)}
                      placeholder="Optional description"
                      className={inputClasses}
                    />
                  </div>
                </div>
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => setShowSavePresetForm(false)}
                    className="px-3 py-1.5 text-sm text-light-text dark:text-dark-text hover:bg-light-border dark:hover:bg-dark-border rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSavePreset}
                    disabled={!savePresetName.trim() || isSavingPreset}
                    className="px-3 py-1.5 text-sm bg-primary text-white rounded-lg hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                  >
                    {isSavingPreset ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <Save className="w-3 h-3" />
                    )}
                    Save
                  </button>
                </div>
              </div>
            )}

            {/* ---- Basic section (always open) ---- */}
            <div className="border border-light-border dark:border-dark-border rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection('basic')}
                className="w-full px-4 py-3 flex items-center justify-between bg-light-bg dark:bg-dark-bg hover:bg-light-hover dark:hover:bg-dark-hover transition-colors"
              >
                <span className="font-medium text-light-text dark:text-dark-text">Basic</span>
                {expandedSections.basic ? (
                  <ChevronUp className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                )}
              </button>
              {expandedSections.basic && (
                <div className="p-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Model */}
                    <div className="sm:col-span-2">
                      <label className={labelClasses}>Model</label>
                      <select
                        value={config.model_path}
                        onChange={(e) => updateConfig('model_path', e.target.value)}
                        className={selectClasses}
                      >
                        <option value="">-- Select a GGUF model --</option>
                        {localModels.map((m) => (
                          <option key={m.filepath} value={m.filepath}>
                            {m.filename} ({m.size_pretty})
                          </option>
                        ))}
                      </select>
                      {isLoadingModels && (
                        <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1 flex items-center gap-1">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          Loading models...
                        </p>
                      )}
                      {!isLoadingModels && localModels.length === 0 && (
                        <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1">
                          No GGUF models found{modelsDir ? ` in ${modelsDir}` : ''}. Download one from the Model Library below.
                        </p>
                      )}
                    </div>

                    {/* Port */}
                    <div>
                      <label className={labelClasses}>Port</label>
                      <input
                        type="number"
                        value={config.port}
                        onChange={(e) => updateConfig('port', parseInt(e.target.value) || 8080)}
                        min={1}
                        max={65535}
                        className={inputClasses}
                      />
                    </div>

                    {/* Host */}
                    <div>
                      <label className={labelClasses}>Host</label>
                      <input
                        type="text"
                        value={config.host}
                        onChange={(e) => updateConfig('host', e.target.value)}
                        className={inputClasses}
                      />
                      {config.host !== '127.0.0.1' && config.host !== 'localhost' && (
                        <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-1 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          Non-localhost binding exposes the server to the network.
                        </p>
                      )}
                    </div>

                    {/* Alias */}
                    <div>
                      <label className={labelClasses}>Alias</label>
                      <input
                        type="text"
                        value={config.alias || ''}
                        onChange={(e) => updateConfig('alias', e.target.value || undefined)}
                        placeholder="Optional model alias"
                        className={inputClasses}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ---- Performance section ---- */}
            <div className="border border-light-border dark:border-dark-border rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection('performance')}
                className="w-full px-4 py-3 flex items-center justify-between bg-light-bg dark:bg-dark-bg hover:bg-light-hover dark:hover:bg-dark-hover transition-colors"
              >
                <span className="font-medium text-light-text dark:text-dark-text">Performance</span>
                {expandedSections.performance ? (
                  <ChevronUp className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                )}
              </button>
              {expandedSections.performance && (
                <div className="p-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Context Size */}
                    <div>
                      <label className={labelClasses}>Context Size</label>
                      <input
                        type="number"
                        value={config.ctx_size}
                        onChange={(e) => updateConfig('ctx_size', parseInt(e.target.value) || 0)}
                        min={0}
                        className={inputClasses}
                      />
                      <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1">
                        0 = model default
                      </p>
                    </div>

                    {/* GPU Layers */}
                    <div>
                      <label className={labelClasses}>GPU Layers</label>
                      <select
                        value={gpuLayersMode}
                        onChange={(e) =>
                          handleGpuLayersModeChange(e.target.value as 'auto' | 'all' | 'custom')
                        }
                        className={selectClasses}
                      >
                        <option value="auto">Auto</option>
                        <option value="all">All</option>
                        <option value="custom">Custom</option>
                      </select>
                      {gpuLayersMode === 'custom' && (
                        <input
                          type="number"
                          value={gpuLayersCustom}
                          onChange={(e) =>
                            handleGpuLayersCustomChange(parseInt(e.target.value) || 0)
                          }
                          min={0}
                          className={`${inputClasses} mt-2`}
                          placeholder="Number of layers"
                        />
                      )}
                    </div>

                    {/* Threads */}
                    <div>
                      <label className={labelClasses}>Threads</label>
                      <input
                        type="number"
                        value={config.threads}
                        onChange={(e) => updateConfig('threads', parseInt(e.target.value) || 1)}
                        min={1}
                        className={inputClasses}
                      />
                    </div>

                    {/* Batch Size */}
                    <div>
                      <label className={labelClasses}>Batch Size</label>
                      <input
                        type="number"
                        value={config.batch_size}
                        onChange={(e) =>
                          updateConfig('batch_size', parseInt(e.target.value) || 512)
                        }
                        min={1}
                        className={inputClasses}
                      />
                    </div>

                    {/* Ubatch Size */}
                    <div>
                      <label className={labelClasses}>Ubatch Size</label>
                      <input
                        type="number"
                        value={config.ubatch_size}
                        onChange={(e) =>
                          updateConfig('ubatch_size', parseInt(e.target.value) || 512)
                        }
                        min={1}
                        className={inputClasses}
                      />
                    </div>

                    {/* Flash Attention */}
                    <div>
                      <label className={labelClasses}>Flash Attention</label>
                      <select
                        value={flashAttnMode}
                        onChange={(e) =>
                          handleFlashAttnChange(e.target.value as 'auto' | 'on' | 'off')
                        }
                        className={selectClasses}
                      >
                        <option value="auto">Auto</option>
                        <option value="on">On</option>
                        <option value="off">Off</option>
                      </select>
                    </div>

                    {/* Parallel Slots */}
                    <div>
                      <label className={labelClasses}>Parallel Slots</label>
                      <input
                        type="number"
                        value={config.parallel}
                        onChange={(e) => updateConfig('parallel', parseInt(e.target.value) || 1)}
                        min={1}
                        className={inputClasses}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ---- Features section ---- */}
            <div className="border border-light-border dark:border-dark-border rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection('features')}
                className="w-full px-4 py-3 flex items-center justify-between bg-light-bg dark:bg-dark-bg hover:bg-light-hover dark:hover:bg-dark-hover transition-colors"
              >
                <span className="font-medium text-light-text dark:text-dark-text">Features</span>
                {expandedSections.features ? (
                  <ChevronUp className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                )}
              </button>
              {expandedSections.features && (
                <div className="p-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Embedding */}
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-light-text dark:text-dark-text">
                        Embedding
                      </label>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={config.embedding}
                        onClick={() => updateConfig('embedding', !config.embedding)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary ${
                          config.embedding ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            config.embedding ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>

                    {/* Continuous Batching */}
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-light-text dark:text-dark-text">
                        Continuous Batching
                      </label>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={config.cont_batching}
                        onClick={() => updateConfig('cont_batching', !config.cont_batching)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary ${
                          config.cont_batching ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            config.cont_batching ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>

                    {/* Cache Prompt */}
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-light-text dark:text-dark-text">
                        Cache Prompt
                      </label>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={config.cache_prompt}
                        onClick={() => updateConfig('cache_prompt', !config.cache_prompt)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary ${
                          config.cache_prompt ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            config.cache_prompt ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>

                    {/* MLock */}
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-light-text dark:text-dark-text">
                        MLock
                      </label>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={config.mlock}
                        onClick={() => updateConfig('mlock', !config.mlock)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary ${
                          config.mlock ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            config.mlock ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>

                    {/* Metrics */}
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-light-text dark:text-dark-text">
                        Metrics
                      </label>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={config.metrics}
                        onClick={() => updateConfig('metrics', !config.metrics)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary ${
                          config.metrics ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            config.metrics ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>

                    {/* Jinja */}
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-light-text dark:text-dark-text">
                        Jinja
                      </label>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={config.jinja}
                        onClick={() => updateConfig('jinja', !config.jinja)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary ${
                          config.jinja ? 'bg-primary' : 'bg-gray-300 dark:bg-gray-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            config.jinja ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ---- KV Cache section ---- */}
            <div className="border border-light-border dark:border-dark-border rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection('kvcache')}
                className="w-full px-4 py-3 flex items-center justify-between bg-light-bg dark:bg-dark-bg hover:bg-light-hover dark:hover:bg-dark-hover transition-colors"
              >
                <span className="font-medium text-light-text dark:text-dark-text">KV Cache</span>
                {expandedSections.kvcache ? (
                  <ChevronUp className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                )}
              </button>
              {expandedSections.kvcache && (
                <div className="p-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Cache Type K */}
                    <div>
                      <label className={labelClasses}>Cache Type K</label>
                      <select
                        value={config.cache_type_k}
                        onChange={(e) => updateConfig('cache_type_k', e.target.value)}
                        className={selectClasses}
                      >
                        {cacheTypeOptions.map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Cache Type V */}
                    <div>
                      <label className={labelClasses}>Cache Type V</label>
                      <select
                        value={config.cache_type_v}
                        onChange={(e) => updateConfig('cache_type_v', e.target.value)}
                        className={selectClasses}
                      >
                        {cacheTypeOptions.map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ---- Advanced section ---- */}
            <div className="border border-light-border dark:border-dark-border rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection('advanced')}
                className="w-full px-4 py-3 flex items-center justify-between bg-light-bg dark:bg-dark-bg hover:bg-light-hover dark:hover:bg-dark-hover transition-colors"
              >
                <span className="font-medium text-light-text dark:text-dark-text">Advanced</span>
                {expandedSections.advanced ? (
                  <ChevronUp className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-light-text-secondary dark:text-dark-text-secondary" />
                )}
              </button>
              {expandedSections.advanced && (
                <div className="p-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Chat Template */}
                    <div className="sm:col-span-2">
                      <label className={labelClasses}>Chat Template</label>
                      <input
                        type="text"
                        value={config.chat_template || ''}
                        onChange={(e) =>
                          updateConfig('chat_template', e.target.value || undefined)
                        }
                        placeholder="Leave empty for model default"
                        className={inputClasses}
                      />
                    </div>

                    {/* Reasoning Format */}
                    <div>
                      <label className={labelClasses}>Reasoning Format</label>
                      <select
                        value={config.reasoning_format}
                        onChange={(e) => updateConfig('reasoning_format', e.target.value)}
                        className={selectClasses}
                      >
                        <option value="none">None</option>
                        <option value="deepseek">DeepSeek</option>
                        <option value="auto">Auto</option>
                      </select>
                    </div>

                    {/* API Key */}
                    <div>
                      <label className={labelClasses}>API Key</label>
                      <div className="relative">
                        <input
                          type={showApiKey ? 'text' : 'password'}
                          value={config.api_key || ''}
                          onChange={(e) =>
                            updateConfig('api_key', e.target.value || undefined)
                          }
                          placeholder="Optional API key"
                          className={`${inputClasses} pr-10`}
                        />
                        <button
                          type="button"
                          onClick={() => setShowApiKey(!showApiKey)}
                          className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-light-text-secondary dark:text-dark-text-secondary hover:text-light-text dark:hover:text-dark-text"
                        >
                          {showApiKey ? (
                            <EyeOff className="w-4 h-4" />
                          ) : (
                            <Eye className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Log File */}
                    <div>
                      <label className={labelClasses}>Log File</label>
                      <input
                        type="text"
                        value={config.log_file || ''}
                        onChange={(e) =>
                          updateConfig('log_file', e.target.value || undefined)
                        }
                        placeholder="Optional log file path"
                        className={inputClasses}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ==================== Card 3: Model Library ==================== */}
        <div className="bg-light-sidebar dark:bg-dark-sidebar rounded-xl border border-light-border dark:border-dark-border overflow-hidden">
          <div className="px-4 py-3 border-b border-light-border dark:border-dark-border">
            <h2 className="text-lg font-semibold text-light-text dark:text-dark-text flex items-center gap-2">
              <BookOpen className="w-5 h-5" />
              Model Library
            </h2>
          </div>
          <div className="p-4">
            {/* Tab buttons */}
            <div className="flex border-b border-light-border dark:border-dark-border mb-4">
              <button
                onClick={() => setModelLibraryTab('downloaded')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  modelLibraryTab === 'downloaded'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-light-text-secondary dark:text-dark-text-secondary hover:text-light-text dark:hover:text-dark-text'
                }`}
              >
                <FolderOpen className="w-4 h-4 inline mr-1.5" />
                Downloaded
              </button>
              <button
                onClick={() => setModelLibraryTab('recommended')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  modelLibraryTab === 'recommended'
                    ? 'border-primary text-primary'
                    : 'border-transparent text-light-text-secondary dark:text-dark-text-secondary hover:text-light-text dark:hover:text-dark-text'
                }`}
              >
                <Download className="w-4 h-4 inline mr-1.5" />
                Recommended
              </button>
            </div>

            {/* Download progress bar */}
            {isDownloading && downloadProgress && (
              <div className="mb-4 p-4 bg-light-bg dark:bg-dark-bg rounded-lg border border-light-border dark:border-dark-border">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-light-text dark:text-dark-text flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Downloading {downloadProgress.filename || 'model'}...
                  </span>
                  <span className="text-sm text-light-text-secondary dark:text-dark-text-secondary">
                    {downloadProgress.percentage != null
                      ? `${downloadProgress.percentage.toFixed(1)}%`
                      : '--'}
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                  <div
                    className="bg-primary h-2.5 rounded-full transition-all"
                    style={{
                      width: `${downloadProgress.percentage ?? 0}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between mt-1.5 text-xs text-light-text-secondary dark:text-dark-text-secondary">
                  <span>
                    {downloadProgress.speed_mbps != null
                      ? `${downloadProgress.speed_mbps.toFixed(1)} MB/s`
                      : ''}
                  </span>
                  <span>ETA: {formatEta(downloadProgress.eta_seconds)}</span>
                </div>
              </div>
            )}

            {/* Download error */}
            {downloadError && (
              <div className="mb-4 flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-red-700 dark:text-red-300">{downloadError}</p>
                  <button
                    onClick={() => setDownloadError(null)}
                    className="text-xs text-red-500 hover:underline mt-1"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            )}

            {/* Downloaded tab */}
            {modelLibraryTab === 'downloaded' && (
              <div>
                {isLoadingModels ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 text-primary animate-spin" />
                  </div>
                ) : localModels.length === 0 ? (
                  <div className="text-center py-8">
                    <FolderOpen className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                    <p className="text-light-text-secondary dark:text-dark-text-secondary">
                      No GGUF models found.
                    </p>
                    {modelsDir && (
                      <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1">
                        Models directory: {modelsDir}
                      </p>
                    )}
                    <button
                      onClick={() => setModelLibraryTab('recommended')}
                      className="mt-3 text-sm text-primary hover:text-primary-hover"
                    >
                      Browse recommended models
                    </button>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-light-bg dark:bg-dark-bg">
                        <tr>
                          <th className="px-4 py-2 text-left text-sm font-medium text-light-text-secondary dark:text-dark-text-secondary">
                            Model Name
                          </th>
                          <th className="px-4 py-2 text-right text-sm font-medium text-light-text-secondary dark:text-dark-text-secondary">
                            Size
                          </th>
                          <th className="px-4 py-2 text-right text-sm font-medium text-light-text-secondary dark:text-dark-text-secondary">
                            Modified
                          </th>
                          <th className="px-4 py-2 text-right text-sm font-medium text-light-text-secondary dark:text-dark-text-secondary">
                            Actions
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {localModels.map((model, idx) => (
                          <tr
                            key={model.filepath}
                            className={`${
                              idx % 2 === 0 ? '' : 'bg-light-bg/50 dark:bg-dark-bg/50'
                            } ${
                              config.model_path === model.filepath
                                ? 'ring-1 ring-inset ring-primary/30'
                                : ''
                            }`}
                          >
                            <td className="px-4 py-2 text-sm text-light-text dark:text-dark-text">
                              <div className="font-medium">{model.filename}</div>
                              <div className="text-xs text-light-text-secondary dark:text-dark-text-secondary truncate max-w-xs">
                                {model.filepath}
                              </div>
                            </td>
                            <td className="px-4 py-2 text-sm text-right text-light-text dark:text-dark-text whitespace-nowrap">
                              {model.size_pretty}
                            </td>
                            <td className="px-4 py-2 text-sm text-right text-light-text-secondary dark:text-dark-text-secondary whitespace-nowrap">
                              {new Date(model.modified_at).toLocaleDateString()}
                            </td>
                            <td className="px-4 py-2 text-right">
                              <button
                                onClick={() => handleSelectModel(model.filepath)}
                                disabled={config.model_path === model.filepath}
                                className={`px-3 py-1.5 text-sm rounded-lg transition-colors flex items-center gap-1 ml-auto ${
                                  config.model_path === model.filepath
                                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 cursor-default'
                                    : 'bg-primary text-white hover:bg-primary-hover'
                                }`}
                              >
                                {config.model_path === model.filepath ? (
                                  <>
                                    <Check className="w-3 h-3" />
                                    Selected
                                  </>
                                ) : (
                                  'Select'
                                )}
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* Recommended tab */}
            {modelLibraryTab === 'recommended' && (
              <div className="space-y-3">
                {recommendedModels.length === 0 ? (
                  <div className="text-center py-8">
                    <Download className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                    <p className="text-light-text-secondary dark:text-dark-text-secondary">
                      No recommended models available.
                    </p>
                  </div>
                ) : (
                  recommendedModels.map((model) => (
                    <div
                      key={`${model.repo_id}/${model.filename}`}
                      className="p-4 bg-light-bg dark:bg-dark-bg rounded-lg border border-light-border dark:border-dark-border"
                    >
                      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <h3 className="font-medium text-light-text dark:text-dark-text">
                              {model.name}
                            </h3>
                            <span
                              className={`px-2 py-0.5 text-xs font-medium rounded-full ${getCategoryBadgeColor(
                                model.category
                              )}`}
                            >
                              {model.category}
                            </span>
                            <span className="text-xs text-light-text-secondary dark:text-dark-text-secondary">
                              {model.size_gb} GB
                            </span>
                          </div>
                          <p className="text-sm text-light-text-secondary dark:text-dark-text-secondary mt-1">
                            {model.description}
                          </p>
                          <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1 font-mono">
                            {model.repo_id}/{model.filename}
                          </p>
                        </div>
                        <button
                          onClick={() => handleDownloadModel(model.repo_id, model.filename)}
                          disabled={isDownloading}
                          className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 flex-shrink-0"
                        >
                          {isDownloading ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Download className="w-4 h-4" />
                          )}
                          Download
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>

        {/* ==================== Card 4: Server Logs ==================== */}
        <div className="bg-light-sidebar dark:bg-dark-sidebar rounded-xl border border-light-border dark:border-dark-border overflow-hidden">
          <div className="px-4 py-3 border-b border-light-border dark:border-dark-border flex items-center justify-between">
            <h2 className="text-lg font-semibold text-light-text dark:text-dark-text flex items-center gap-2">
              <Terminal className="w-5 h-5" />
              Server Logs
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={handleClearLogs}
                className="px-3 py-1.5 text-sm text-light-text-secondary dark:text-dark-text-secondary hover:bg-light-border dark:hover:bg-dark-border rounded-lg transition-colors flex items-center gap-1"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Clear
              </button>
              <button
                onClick={handleCopyLogs}
                className="px-3 py-1.5 text-sm text-light-text-secondary dark:text-dark-text-secondary hover:bg-light-border dark:hover:bg-dark-border rounded-lg transition-colors flex items-center gap-1"
              >
                {logsCopied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-green-500" />
                    <span className="text-green-500">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    Copy
                  </>
                )}
              </button>
            </div>
          </div>
          <div
            ref={logsContainerRef}
            className="bg-gray-900 text-green-400 font-mono text-xs p-4 h-80 overflow-y-auto"
          >
            {logs.length === 0 ? (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No logs available.</p>
                  <p className="text-xs mt-1">
                    {serverStatus?.status === 'running'
                      ? 'Waiting for log output...'
                      : 'Start the server to see logs.'}
                  </p>
                </div>
              </div>
            ) : (
              logs.map((line, idx) => (
                <div key={idx} className="whitespace-pre-wrap break-all leading-5">
                  {line}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
