/**
 * MLX Server Manager Page
 *
 * Provides UI for managing a local mlx_lm.server process:
 * - Start/stop server with configurable parameters
 * - Browse cached MLX models from HuggingFace
 * - Download recommended models
 * - Configuration presets (save/load/apply)
 * - Real-time server logs viewer
 *
 * Simplified compared to LlamaServerPage since MLX has fewer config options.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  Cpu,
  RefreshCw,
  Play,
  Square,
  ChevronDown,
  ChevronUp,
  Loader2,
  AlertCircle,
  Download,
  Check,
  Copy,
  Save,
  AlertTriangle,
  HardDrive,
  Settings,
  Terminal,
  BookOpen,
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
  DEFAULT_CONFIG,
  type MLXServerConfig,
  type ServerStatusResponse,
  type MLXModelInfo,
  type DownloadProgress,
  type ServerPreset,
  type RecommendedModel
} from '../api/mlx'

export function MLXServerPage() {
  const navigate = useNavigate()

  // --- Server status state ---
  const [serverStatus, setServerStatus] = useState<ServerStatusResponse | null>(null)
  const [, setIsLoadingStatus] = useState(true)

  // --- Configuration state ---
  const [config, setConfig] = useState<MLXServerConfig>({ ...DEFAULT_CONFIG })

  // --- Collapsible sections ---
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    basic: true,
    generation: false,
  })

  // --- Local models ---
  const [localModels, setLocalModels] = useState<MLXModelInfo[]>([])
  const [isLoadingModels, setIsLoadingModels] = useState(false)

  // --- Presets ---
  const [presets, setPresets] = useState<ServerPreset[]>([])
  const [recommendedModels, setRecommendedModels] = useState<RecommendedModel[]>([])
  const [selectedPreset, setSelectedPreset] = useState<string>('')
  const [showSavePreset, setShowSavePreset] = useState(false)
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetDesc, setNewPresetDesc] = useState('')

  // --- Download state ---
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null)
  const [isDownloading, setIsDownloading] = useState(false)

  // --- Logs ---
  const [logs, setLogs] = useState<string[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const logsEndRef = useRef<HTMLDivElement>(null)

  // --- Action state ---
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [actionSuccess, setActionSuccess] = useState<string | null>(null)

  // --- Model tabs ---
  const [modelTab, setModelTab] = useState<'downloaded' | 'recommended'>('downloaded')

  // --- Toggle section ---
  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  // --- Load server status ---
  const loadStatus = useCallback(async () => {
    try {
      const status = await getServerStatus()
      setServerStatus(status)
    } catch {
      // Server might not be available
    } finally {
      setIsLoadingStatus(false)
    }
  }, [])

  // --- Load local models ---
  const loadModels = useCallback(async () => {
    setIsLoadingModels(true)
    try {
      const data = await listLocalModels()
      setLocalModels(data.models)
    } catch {
      // ignore
    } finally {
      setIsLoadingModels(false)
    }
  }, [])

  // --- Load presets ---
  const loadPresets = useCallback(async () => {
    try {
      const data = await getPresets()
      setPresets(data.presets)
      setRecommendedModels(data.recommended_models)
    } catch {
      // ignore
    }
  }, [])

  // --- Load logs ---
  const loadLogs = useCallback(async () => {
    try {
      const data = await getServerLogs()
      setLogs(data.logs)
    } catch {
      // ignore
    }
  }, [])

  // --- Initial load ---
  useEffect(() => {
    loadStatus()
    loadModels()
    loadPresets()
  }, [loadStatus, loadModels, loadPresets])

  // --- Status polling ---
  useEffect(() => {
    const interval = setInterval(loadStatus, 3000)
    return () => clearInterval(interval)
  }, [loadStatus])

  // --- Log polling ---
  useEffect(() => {
    if (!showLogs) return
    loadLogs()
    const interval = setInterval(loadLogs, 2000)
    return () => clearInterval(interval)
  }, [showLogs, loadLogs])

  // --- Download progress polling ---
  useEffect(() => {
    if (!isDownloading) return
    const interval = setInterval(async () => {
      try {
        const progress = await getDownloadProgress()
        setDownloadProgress(progress)
        if (progress.status === 'completed' || progress.status === 'failed') {
          setIsDownloading(false)
          if (progress.status === 'completed') {
            loadModels()
            setActionSuccess('Model downloaded successfully!')
            setTimeout(() => setActionSuccess(null), 3000)
          } else if (progress.error_message) {
            setActionError(progress.error_message)
          }
        }
      } catch {
        setIsDownloading(false)
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [isDownloading, loadModels])

  // --- Auto-scroll logs ---
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  // --- Start server ---
  const handleStart = async () => {
    if (!config.model) {
      setActionError('Please select a model first')
      return
    }
    setIsStarting(true)
    setActionError(null)
    try {
      await startServer(config)
      setActionSuccess('Server started!')
      setTimeout(() => setActionSuccess(null), 3000)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to start server')
    } finally {
      setIsStarting(false)
    }
  }

  // --- Stop server ---
  const handleStop = async () => {
    setIsStopping(true)
    setActionError(null)
    try {
      await stopServer()
      setActionSuccess('Server stopped')
      setTimeout(() => setActionSuccess(null), 3000)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to stop server')
    } finally {
      setIsStopping(false)
    }
  }

  // --- Reset server ---
  const handleReset = async () => {
    setActionError(null)
    try {
      await resetServer()
      setActionSuccess('Server reset')
      setTimeout(() => setActionSuccess(null), 3000)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to reset server')
    }
  }

  // --- Download model ---
  const handleDownload = async (repoId: string) => {
    setActionError(null)
    setIsDownloading(true)
    try {
      const result = await downloadModel({ repo_id: repoId })
      if (result.already_exists) {
        setActionSuccess('Model already downloaded!')
        setTimeout(() => setActionSuccess(null), 3000)
        setIsDownloading(false)
      }
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to start download')
      setIsDownloading(false)
    }
  }

  // --- Apply preset ---
  const handleApplyPreset = (preset: ServerPreset) => {
    setConfig(prev => ({
      ...preset.config,
      model: prev.model, // Keep current model selection
    }))
    setSelectedPreset(preset.name)
  }

  // --- Save preset ---
  const handleSavePreset = async () => {
    if (!newPresetName.trim()) return
    try {
      await savePreset(newPresetName.trim(), newPresetDesc.trim(), config)
      setShowSavePreset(false)
      setNewPresetName('')
      setNewPresetDesc('')
      loadPresets()
      setActionSuccess(`Preset "${newPresetName}" saved!`)
      setTimeout(() => setActionSuccess(null), 3000)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to save preset')
    }
  }

  // --- Copy logs ---
  const handleCopyLogs = () => {
    navigator.clipboard.writeText(logs.join('\n'))
    setActionSuccess('Logs copied!')
    setTimeout(() => setActionSuccess(null), 2000)
  }

  // --- Status helpers ---
  const status = serverStatus?.status || 'stopped'
  const isRunning = status === 'running'
  const isStartingState = status === 'starting'
  const isError = status === 'error'

  const statusColor = {
    stopped: 'text-gray-500',
    starting: 'text-yellow-500',
    running: 'text-green-500',
    stopping: 'text-yellow-500',
    error: 'text-red-500',
  }[status] || 'text-gray-500'

  const statusBg = {
    stopped: 'bg-gray-500/10',
    starting: 'bg-yellow-500/10',
    running: 'bg-green-500/10',
    stopping: 'bg-yellow-500/10',
    error: 'bg-red-500/10',
  }[status] || 'bg-gray-500/10'

  return (
    <div className="flex-1 flex flex-col h-screen overflow-y-auto bg-light-bg dark:bg-dark-bg">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-light-bg/95 dark:bg-dark-bg/95 backdrop-blur border-b border-light-border dark:border-dark-border">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="p-2 rounded-lg hover:bg-light-border dark:hover:bg-dark-border transition-colors"
          >
            <ArrowLeft size={20} className="text-light-text dark:text-dark-text" />
          </button>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-purple-500/10">
              <Cpu size={24} className="text-purple-500" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-light-text dark:text-dark-text">MLX Server</h1>
              <p className="text-sm text-light-text-secondary dark:text-dark-text-secondary">
                Apple MLX — optimized for Apple Silicon
              </p>
            </div>
          </div>

          {/* Status badge */}
          <div className={`ml-auto px-3 py-1.5 rounded-full text-sm font-medium ${statusBg} ${statusColor}`}>
            {isStartingState && <Loader2 size={14} className="inline animate-spin mr-1" />}
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-6 w-full space-y-6">
        {/* Error/Success messages */}
        {actionError && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4 flex items-start gap-3">
            <AlertCircle size={20} className="text-red-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm text-red-700 dark:text-red-300">{actionError}</div>
            <button onClick={() => setActionError(null)} className="ml-auto text-red-500 hover:text-red-700">&times;</button>
          </div>
        )}
        {actionSuccess && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-4 flex items-center gap-3">
            <Check size={20} className="text-green-500" />
            <div className="text-sm text-green-700 dark:text-green-300">{actionSuccess}</div>
          </div>
        )}

        {/* Server Info (when running) */}
        {isRunning && serverStatus && (
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-light-text-secondary dark:text-dark-text-secondary">Model</span>
                <p className="font-medium text-light-text dark:text-dark-text truncate">{serverStatus.loaded_model}</p>
              </div>
              <div>
                <span className="text-light-text-secondary dark:text-dark-text-secondary">Port</span>
                <p className="font-medium text-light-text dark:text-dark-text">{serverStatus.port}</p>
              </div>
              <div>
                <span className="text-light-text-secondary dark:text-dark-text-secondary">PID</span>
                <p className="font-medium text-light-text dark:text-dark-text">{serverStatus.pid}</p>
              </div>
              <div>
                <span className="text-light-text-secondary dark:text-dark-text-secondary">Uptime</span>
                <p className="font-medium text-light-text dark:text-dark-text">{serverStatus.uptime_pretty || '-'}</p>
              </div>
            </div>
          </div>
        )}

        {/* Model mismatch warning */}
        {isRunning && (() => {
          try {
            const saved = localStorage.getItem('rag-settings')
            const llmModel = saved ? JSON.parse(saved).llm_model || 'gpt-4o' : 'gpt-4o'
            if (!llmModel.startsWith('mlx:')) {
              return (
                <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl p-4 flex items-start gap-3">
                  <AlertTriangle size={20} className="text-amber-500 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
                      Settings use <code className="bg-amber-100 dark:bg-amber-900/40 px-1 py-0.5 rounded text-xs">{llmModel}</code> — not this MLX server
                    </p>
                    <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                      The model <strong>{serverStatus?.loaded_model}</strong> is loaded in memory but RAG queries go to a different provider. Go to Settings to switch, or stop the server to free RAM.
                    </p>
                    <button
                      onClick={() => navigate('/settings')}
                      className="mt-2 text-sm text-amber-700 dark:text-amber-300 hover:underline font-medium flex items-center gap-1"
                    >
                      <Settings size={14} />
                      Go to Settings →
                    </button>
                  </div>
                </div>
              )
            }
          } catch { /* ignore */ }
          return null
        })()}

        {/* Error info */}
        {isError && serverStatus?.error_message && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle size={16} className="text-red-500" />
              <span className="font-medium text-red-700 dark:text-red-300">Server Error</span>
            </div>
            <p className="text-sm text-red-600 dark:text-red-400">{serverStatus.error_message}</p>
            <button onClick={handleReset} className="mt-2 text-sm text-red-600 hover:text-red-800 underline">
              Reset server state
            </button>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-3">
          {!isRunning && !isStartingState ? (
            <button
              onClick={handleStart}
              disabled={isStarting || !config.model}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-xl transition-colors"
            >
              {isStarting ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
              Start Server
            </button>
          ) : (
            <button
              onClick={handleStop}
              disabled={isStopping}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-xl transition-colors"
            >
              {isStopping ? <Loader2 size={16} className="animate-spin" /> : <Square size={16} />}
              Stop Server
            </button>
          )}

          <button
            onClick={() => setShowLogs(!showLogs)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-colors ${
              showLogs
                ? 'bg-primary/10 border-primary text-primary'
                : 'border-light-border dark:border-dark-border text-light-text dark:text-dark-text hover:bg-light-border dark:hover:bg-dark-border'
            }`}
          >
            <Terminal size={16} />
            Logs
          </button>

          {/* Preset dropdown */}
          <div className="relative ml-auto">
            <select
              value={selectedPreset}
              onChange={(e) => {
                const preset = presets.find(p => p.name === e.target.value)
                if (preset) handleApplyPreset(preset)
              }}
              className="px-4 py-2 rounded-xl border border-light-border dark:border-dark-border bg-white dark:bg-dark-sidebar text-light-text dark:text-dark-text text-sm"
            >
              <option value="">Select Preset</option>
              {presets.map(p => (
                <option key={p.name} value={p.name}>{p.name} {p.builtin ? '' : '(custom)'}</option>
              ))}
            </select>
          </div>

          <button
            onClick={() => setShowSavePreset(!showSavePreset)}
            className="p-2 rounded-xl border border-light-border dark:border-dark-border hover:bg-light-border dark:hover:bg-dark-border transition-colors"
            title="Save current config as preset"
          >
            <Save size={16} className="text-light-text dark:text-dark-text" />
          </button>
        </div>

        {/* Save preset form */}
        {showSavePreset && (
          <div className="bg-white dark:bg-dark-sidebar border border-light-border dark:border-dark-border rounded-xl p-4 space-y-3">
            <h3 className="text-sm font-medium text-light-text dark:text-dark-text">Save Current Config as Preset</h3>
            <input
              type="text"
              placeholder="Preset name"
              value={newPresetName}
              onChange={e => setNewPresetName(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-light-border dark:border-dark-border bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text text-sm"
            />
            <input
              type="text"
              placeholder="Description (optional)"
              value={newPresetDesc}
              onChange={e => setNewPresetDesc(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-light-border dark:border-dark-border bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text text-sm"
            />
            <div className="flex gap-2">
              <button onClick={handleSavePreset} disabled={!newPresetName.trim()} className="px-3 py-1.5 bg-primary text-white rounded-lg text-sm disabled:bg-gray-400">
                Save
              </button>
              <button onClick={() => setShowSavePreset(false)} className="px-3 py-1.5 border border-light-border dark:border-dark-border rounded-lg text-sm text-light-text dark:text-dark-text">
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Configuration */}
        <div className="bg-white dark:bg-dark-sidebar border border-light-border dark:border-dark-border rounded-xl overflow-hidden">
          {/* Basic section */}
          <button
            onClick={() => toggleSection('basic')}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-light-bg/50 dark:hover:bg-dark-bg/50"
          >
            <div className="flex items-center gap-2">
              <Settings size={16} className="text-purple-500" />
              <span className="font-medium text-light-text dark:text-dark-text">Basic Configuration</span>
            </div>
            {expandedSections.basic ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>

          {expandedSections.basic && (
            <div className="px-4 pb-4 space-y-4 border-t border-light-border dark:border-dark-border pt-4">
              {/* Model selection */}
              <div>
                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                  Model (HuggingFace repo)
                </label>
                <input
                  type="text"
                  value={config.model}
                  onChange={e => setConfig(prev => ({ ...prev, model: e.target.value }))}
                  placeholder="mlx-community/Qwen2.5-32B-Instruct-4bit"
                  className="w-full px-3 py-2 rounded-lg border border-light-border dark:border-dark-border bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text text-sm"
                  disabled={isRunning}
                />
                <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1">
                  Enter a HuggingFace model repo or select from downloaded models below
                </p>
              </div>

              {/* Host & Port */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">Host</label>
                  <input
                    type="text"
                    value={config.host}
                    onChange={e => setConfig(prev => ({ ...prev, host: e.target.value }))}
                    className="w-full px-3 py-2 rounded-lg border border-light-border dark:border-dark-border bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text text-sm"
                    disabled={isRunning}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">Port</label>
                  <input
                    type="number"
                    value={config.port}
                    onChange={e => setConfig(prev => ({ ...prev, port: parseInt(e.target.value) || 8081 }))}
                    className="w-full px-3 py-2 rounded-lg border border-light-border dark:border-dark-border bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text text-sm"
                    disabled={isRunning}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Generation section */}
          <button
            onClick={() => toggleSection('generation')}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-light-bg/50 dark:hover:bg-dark-bg/50 border-t border-light-border dark:border-dark-border"
          >
            <div className="flex items-center gap-2">
              <BookOpen size={16} className="text-blue-500" />
              <span className="font-medium text-light-text dark:text-dark-text">Generation Settings</span>
            </div>
            {expandedSections.generation ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>

          {expandedSections.generation && (
            <div className="px-4 pb-4 space-y-4 border-t border-light-border dark:border-dark-border pt-4">
              <div>
                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                  Max Tokens <span className="text-xs text-light-text-secondary">(default: 8192)</span>
                </label>
                <input
                  type="number"
                  value={config.max_tokens}
                  onChange={e => setConfig(prev => ({ ...prev, max_tokens: parseInt(e.target.value) || 8192 }))}
                  min={128}
                  max={131072}
                  className="w-full px-3 py-2 rounded-lg border border-light-border dark:border-dark-border bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text text-sm"
                  disabled={isRunning}
                />
                <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1">
                  Maximum tokens per generation. MLX default is 512 which is too low — we override to 8192.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                  Max KV Cache Size <span className="text-xs text-light-text-secondary">(optional)</span>
                </label>
                <input
                  type="number"
                  value={config.max_kv_size || ''}
                  onChange={e => {
                    const val = e.target.value ? parseInt(e.target.value) : null
                    setConfig(prev => ({ ...prev, max_kv_size: val }))
                  }}
                  min={512}
                  placeholder="Unlimited (leave empty)"
                  className="w-full px-3 py-2 rounded-lg border border-light-border dark:border-dark-border bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text text-sm"
                  disabled={isRunning}
                />
                <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1">
                  Limit KV cache memory usage. Leave empty for unlimited. Lower values save RAM but may reduce quality.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Models section */}
        <div className="bg-white dark:bg-dark-sidebar border border-light-border dark:border-dark-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-light-border dark:border-dark-border">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <HardDrive size={16} className="text-purple-500" />
                <span className="font-medium text-light-text dark:text-dark-text">Models</span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setModelTab('downloaded')}
                  className={`px-3 py-1 rounded-lg text-sm transition-colors ${modelTab === 'downloaded' ? 'bg-primary/10 text-primary' : 'text-light-text-secondary dark:text-dark-text-secondary hover:bg-light-border dark:hover:bg-dark-border'}`}
                >
                  Downloaded ({localModels.length})
                </button>
                <button
                  onClick={() => setModelTab('recommended')}
                  className={`px-3 py-1 rounded-lg text-sm transition-colors ${modelTab === 'recommended' ? 'bg-primary/10 text-primary' : 'text-light-text-secondary dark:text-dark-text-secondary hover:bg-light-border dark:hover:bg-dark-border'}`}
                >
                  Recommended
                </button>
                <button
                  onClick={loadModels}
                  disabled={isLoadingModels}
                  className="p-1 rounded-lg hover:bg-light-border dark:hover:bg-dark-border transition-colors"
                  title="Refresh"
                >
                  <RefreshCw size={14} className={`text-light-text-secondary ${isLoadingModels ? 'animate-spin' : ''}`} />
                </button>
              </div>
            </div>
          </div>

          <div className="p-4">
            {modelTab === 'downloaded' ? (
              localModels.length === 0 ? (
                <p className="text-sm text-light-text-secondary dark:text-dark-text-secondary text-center py-4">
                  No MLX models found in cache. Download one from the Recommended tab.
                </p>
              ) : (
                <div className="space-y-2">
                  {localModels.map(model => (
                    <div
                      key={model.name}
                      className={`flex items-center justify-between p-3 rounded-lg border transition-colors cursor-pointer ${
                        config.model === model.name
                          ? 'border-primary bg-primary/5'
                          : 'border-light-border dark:border-dark-border hover:border-primary/50'
                      }`}
                      onClick={() => !isRunning && setConfig(prev => ({ ...prev, model: model.name }))}
                    >
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-light-text dark:text-dark-text truncate">{model.name}</p>
                        <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary">
                          {model.size_pretty} &middot; {model.modified_at}
                        </p>
                      </div>
                      {config.model === model.name && (
                        <Check size={16} className="text-primary flex-shrink-0 ml-2" />
                      )}
                    </div>
                  ))}
                </div>
              )
            ) : (
              <div className="space-y-2">
                {recommendedModels.map(model => {
                  const isDownloaded = localModels.some(m => m.name === model.repo_id)
                  return (
                    <div key={model.repo_id} className="flex items-center justify-between p-3 rounded-lg border border-light-border dark:border-dark-border">
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-light-text dark:text-dark-text">{model.name}</p>
                        <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary">
                          {model.description}
                        </p>
                        <p className="text-xs text-light-text-secondary dark:text-dark-text-secondary mt-1">
                          ~{model.size_gb} GB &middot; {model.repo_id}
                        </p>
                      </div>
                      <div className="flex-shrink-0 ml-3">
                        {isDownloaded ? (
                          <button
                            onClick={() => !isRunning && setConfig(prev => ({ ...prev, model: model.repo_id }))}
                            className="flex items-center gap-1 px-3 py-1.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-lg text-xs"
                          >
                            <Check size={12} /> Select
                          </button>
                        ) : (
                          <button
                            onClick={() => handleDownload(model.repo_id)}
                            disabled={isDownloading}
                            className="flex items-center gap-1 px-3 py-1.5 bg-primary/10 text-primary rounded-lg text-xs hover:bg-primary/20 disabled:opacity-50"
                          >
                            <Download size={12} /> Download
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {/* Download progress */}
            {isDownloading && downloadProgress && (
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2 text-blue-700 dark:text-blue-300">
                    <Loader2 size={14} className="animate-spin" />
                    <span className="font-medium">Downloading {downloadProgress.repo_id}</span>
                  </div>
                  {downloadProgress.percentage != null && (
                    <span className="text-blue-600 dark:text-blue-400 font-mono font-medium">
                      {downloadProgress.percentage}%
                    </span>
                  )}
                </div>

                {/* Progress bar */}
                <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2.5">
                  <div
                    className="bg-blue-600 dark:bg-blue-400 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${downloadProgress.percentage ?? 0}%` }}
                  />
                </div>

                {/* Details */}
                <div className="flex items-center justify-between text-xs text-blue-600/70 dark:text-blue-400/70">
                  <span>
                    {downloadProgress.downloaded_pretty && downloadProgress.total_pretty
                      ? `${downloadProgress.downloaded_pretty} / ${downloadProgress.total_pretty}`
                      : 'Calculating size...'}
                  </span>
                  <span>
                    {downloadProgress.files_total > 0
                      ? `File ${downloadProgress.files_done}/${downloadProgress.files_total}`
                      : 'Resolving files...'}
                  </span>
                </div>

                {/* Current file */}
                {downloadProgress.current_file && (
                  <div className="text-xs text-blue-600/50 dark:text-blue-400/50 truncate">
                    {downloadProgress.current_file}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Logs section */}
        {showLogs && (
          <div className="bg-white dark:bg-dark-sidebar border border-light-border dark:border-dark-border rounded-xl overflow-hidden">
            <div className="p-4 border-b border-light-border dark:border-dark-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Terminal size={16} className="text-purple-500" />
                <span className="font-medium text-light-text dark:text-dark-text">Server Logs</span>
              </div>
              <button
                onClick={handleCopyLogs}
                className="flex items-center gap-1 px-2 py-1 text-xs rounded-lg hover:bg-light-border dark:hover:bg-dark-border text-light-text-secondary"
              >
                <Copy size={12} /> Copy
              </button>
            </div>
            <div className="p-4 max-h-96 overflow-y-auto font-mono text-xs">
              {logs.length === 0 ? (
                <p className="text-light-text-secondary dark:text-dark-text-secondary">No logs yet. Start the server to see logs.</p>
              ) : (
                logs.map((line, i) => (
                  <div key={i} className={`py-0.5 ${line.includes('ERROR') ? 'text-red-500' : line.includes('SYSTEM') ? 'text-blue-500' : 'text-light-text dark:text-dark-text'}`}>
                    {line}
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </div>
        )}

        {/* Info box */}
        <div className="bg-purple-50 dark:bg-purple-900/10 border border-purple-200 dark:border-purple-800 rounded-xl p-4 text-sm text-purple-700 dark:text-purple-300">
          <p className="font-medium mb-1">About MLX</p>
          <p>
            MLX is Apple's machine learning framework optimized for Apple Silicon.
            It provides ~20-40% faster inference compared to llama.cpp on Mac thanks to native unified memory support.
            Models are downloaded from the <strong>mlx-community</strong> on HuggingFace.
          </p>
          <p className="mt-2 text-xs">
            Requires: <code className="bg-purple-100 dark:bg-purple-900/30 px-1 rounded">pip install mlx-lm</code>
          </p>
        </div>
      </div>
    </div>
  )
}

export default MLXServerPage
