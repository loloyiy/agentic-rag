/**
 * useServerMismatch hook
 *
 * Detects when an MLX or llama.cpp server is running but the Settings
 * model selection doesn't match. This prevents users from accidentally
 * wasting RAM on an unused local model (which can crash the system).
 */

import { useState, useEffect, useCallback } from 'react'
import { getServerStatus as getMLXStatus } from '../api/mlx'
import { getServerStatus as getLlamaCppStatus } from '../api/llamacpp'

const POLL_INTERVAL_MS = 15000 // 15 seconds

export interface ServerMismatchWarning {
  type: 'mlx' | 'llamacpp'
  serverName: string
  loadedModel: string
  currentSetting: string
  currentSettingDisplay: string
}

export interface UseServerMismatchResult {
  warnings: ServerMismatchWarning[]
  dismiss: (type: 'mlx' | 'llamacpp') => void
}

function describeModel(llmModel: string): string {
  if (llmModel.startsWith('mlx:')) return `MLX (${llmModel.slice(4)})`
  if (llmModel.startsWith('llamacpp:')) return `llama.cpp (${llmModel.slice(9)})`
  if (llmModel.startsWith('ollama:')) return `Ollama (${llmModel.slice(7)})`
  if (llmModel.startsWith('openrouter:')) return `OpenRouter (${llmModel.slice(11)})`
  return `OpenAI (${llmModel})`
}

function getCurrentLlmModel(): string {
  try {
    const saved = localStorage.getItem('rag-settings')
    if (saved) {
      const parsed = JSON.parse(saved)
      return parsed.llm_model || 'gpt-4o'
    }
  } catch {
    // ignore parse errors
  }
  return 'gpt-4o'
}

export function useServerMismatch(): UseServerMismatchResult {
  const [mlxRunning, setMlxRunning] = useState(false)
  const [mlxModel, setMlxModel] = useState<string>('')
  const [llamacppRunning, setLlamacppRunning] = useState(false)
  const [llamacppModel, setLlamacppModel] = useState<string>('')
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())

  const checkServers = useCallback(async () => {
    // Check MLX
    try {
      const mlxStatus = await getMLXStatus()
      setMlxRunning(mlxStatus.status === 'running')
      setMlxModel(mlxStatus.loaded_model || '')
    } catch {
      setMlxRunning(false)
    }

    // Check llama.cpp
    try {
      const llamaStatus = await getLlamaCppStatus()
      setLlamacppRunning(llamaStatus.status === 'running')
      setLlamacppModel(llamaStatus.loaded_model || '')
    } catch {
      setLlamacppRunning(false)
    }
  }, [])

  useEffect(() => {
    checkServers()
    const interval = setInterval(checkServers, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [checkServers])

  // Build warnings
  const warnings: ServerMismatchWarning[] = []
  const llmModel = getCurrentLlmModel()

  if (mlxRunning && !llmModel.startsWith('mlx:') && !dismissed.has('mlx')) {
    warnings.push({
      type: 'mlx',
      serverName: 'MLX',
      loadedModel: mlxModel,
      currentSetting: llmModel,
      currentSettingDisplay: describeModel(llmModel),
    })
  }

  if (llamacppRunning && !llmModel.startsWith('llamacpp:') && !dismissed.has('llamacpp')) {
    warnings.push({
      type: 'llamacpp',
      serverName: 'llama.cpp',
      loadedModel: llamacppModel,
      currentSetting: llmModel,
      currentSettingDisplay: describeModel(llmModel),
    })
  }

  const dismiss = useCallback((type: 'mlx' | 'llamacpp') => {
    setDismissed(prev => new Set([...prev, type]))
  }, [])

  return { warnings, dismiss }
}
