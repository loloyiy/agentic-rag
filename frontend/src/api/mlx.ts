/**
 * MLX Server Manager API client
 * Handles server lifecycle, model management, configuration presets, and downloads.
 * MLX uses Apple's ML framework for fast LLM inference on Apple Silicon.
 */

const API_BASE = '/api/mlx'

// ==================== Types ====================

export type ServerStatus = 'stopped' | 'starting' | 'running' | 'stopping' | 'error'

export interface MLXServerConfig {
  model: string
  port: number
  host: string
  max_tokens: number
  max_kv_size?: number | null
}

export interface ServerStatusResponse {
  status: ServerStatus
  pid?: number
  uptime_seconds?: number
  uptime_pretty?: string
  loaded_model?: string
  port?: number
  host?: string
  config?: MLXServerConfig
  error_message?: string
  mlx_available: boolean
}

export interface MLXModelInfo {
  name: string
  path: string
  size_bytes: number
  size_pretty: string
  modified_at: string
}

export interface ModelDownloadRequest {
  repo_id: string
}

export interface DownloadProgress {
  status: string
  repo_id?: string
  percentage?: number
  downloaded_bytes?: number
  total_bytes?: number
  downloaded_pretty?: string
  total_pretty?: string
  current_file?: string
  files_done: number
  files_total: number
  error_message?: string
}

export interface ServerPreset {
  name: string
  description: string
  config: MLXServerConfig
  builtin: boolean
}

export interface RecommendedModel {
  name: string
  description: string
  repo_id: string
  size_gb: number
  category: string
}

// ==================== Default Config ====================

export const DEFAULT_CONFIG: MLXServerConfig = {
  model: '',
  port: 8081,
  host: '127.0.0.1',
  max_tokens: 8192,
  max_kv_size: null,
}

// ==================== API Functions ====================

export async function startServer(config: MLXServerConfig): Promise<ServerStatusResponse> {
  const response = await fetch(`${API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to start server' }))
    throw new Error(error.detail || 'Failed to start server')
  }
  return response.json()
}

export async function stopServer(): Promise<ServerStatusResponse> {
  const response = await fetch(`${API_BASE}/stop`, {
    method: 'POST',
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to stop server' }))
    throw new Error(error.detail || 'Failed to stop server')
  }
  return response.json()
}

export async function resetServer(): Promise<ServerStatusResponse> {
  const response = await fetch(`${API_BASE}/reset`, {
    method: 'POST',
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to reset server' }))
    throw new Error(error.detail || 'Failed to reset server')
  }
  return response.json()
}

export async function getServerStatus(): Promise<ServerStatusResponse> {
  const response = await fetch(`${API_BASE}/status`)
  if (!response.ok) {
    throw new Error('Failed to get server status')
  }
  return response.json()
}

export async function getServerLogs(lines: number = 200): Promise<{ logs: string[] }> {
  const response = await fetch(`${API_BASE}/logs?lines=${lines}`)
  if (!response.ok) {
    throw new Error('Failed to get server logs')
  }
  return response.json()
}

export async function listLocalModels(): Promise<{ models: MLXModelInfo[]; cache_dir: string }> {
  const response = await fetch(`${API_BASE}/models`)
  if (!response.ok) {
    throw new Error('Failed to list models')
  }
  return response.json()
}

export async function downloadModel(request: ModelDownloadRequest): Promise<{ message: string; already_exists: boolean }> {
  const response = await fetch(`${API_BASE}/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to start download' }))
    throw new Error(error.detail || 'Failed to start download')
  }
  return response.json()
}

export async function getDownloadProgress(): Promise<DownloadProgress> {
  const response = await fetch(`${API_BASE}/download/progress`)
  if (!response.ok) {
    throw new Error('Failed to get download progress')
  }
  return response.json()
}

export async function getPresets(): Promise<{ presets: ServerPreset[]; recommended_models: RecommendedModel[] }> {
  const response = await fetch(`${API_BASE}/presets`)
  if (!response.ok) {
    throw new Error('Failed to get presets')
  }
  return response.json()
}

export async function savePreset(name: string, description: string, config: MLXServerConfig): Promise<{ success: boolean }> {
  const response = await fetch(`${API_BASE}/config/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, config }),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to save preset' }))
    throw new Error(error.detail || 'Failed to save preset')
  }
  return response.json()
}

export async function deletePreset(name: string): Promise<{ success: boolean }> {
  const response = await fetch(`${API_BASE}/config/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to delete preset' }))
    throw new Error(error.detail || 'Failed to delete preset')
  }
  return response.json()
}
