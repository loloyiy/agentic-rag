/**
 * llama.cpp Server Manager API client
 * Handles server lifecycle, model management, configuration presets, and downloads.
 */

const API_BASE = '/api/llamacpp'

// ==================== Types ====================

export type ServerStatus = 'stopped' | 'starting' | 'running' | 'stopping' | 'error'

export interface LlamaServerConfig {
  model_path: string
  port: number
  host: string
  alias?: string
  ctx_size: number
  n_gpu_layers: string
  threads: number
  batch_size: number
  ubatch_size: number
  flash_attn: string
  parallel: number
  embedding: boolean
  cont_batching: boolean
  cache_prompt: boolean
  mlock: boolean
  metrics: boolean
  jinja: boolean
  cache_type_k: string
  cache_type_v: string
  chat_template?: string
  reasoning_format: string
  api_key?: string
  log_file?: string
}

export interface ServerStatusResponse {
  status: ServerStatus
  pid?: number
  uptime_seconds?: number
  uptime_pretty?: string
  loaded_model?: string
  port?: number
  host?: string
  config?: LlamaServerConfig
  error_message?: string
  binary_path?: string
  binary_found: boolean
}

export interface GGUFModelInfo {
  filename: string
  filepath: string
  size_bytes: number
  size_pretty: string
  modified_at: string
}

export interface ModelDownloadRequest {
  repo_id: string
  filename: string
}

export interface DownloadProgress {
  status: string
  filename?: string
  total_bytes?: number
  downloaded_bytes: number
  percentage?: number
  speed_mbps?: number
  eta_seconds?: number
  error_message?: string
}

export interface ServerPreset {
  name: string
  description: string
  config: LlamaServerConfig
  builtin: boolean
}

export interface RecommendedModel {
  name: string
  description: string
  repo_id: string
  filename: string
  size_gb: number
  category: string
}

// ==================== Default Config ====================

export const DEFAULT_CONFIG: LlamaServerConfig = {
  model_path: '',
  port: 8080,
  host: '127.0.0.1',
  ctx_size: 0,
  n_gpu_layers: 'auto',
  threads: 1,
  batch_size: 2048,
  ubatch_size: 512,
  flash_attn: 'auto',
  parallel: 1,
  embedding: false,
  cont_batching: true,
  cache_prompt: true,
  mlock: false,
  metrics: false,
  jinja: true,
  cache_type_k: 'f16',
  cache_type_v: 'f16',
  reasoning_format: 'none',
}

// ==================== API Functions ====================

export async function startServer(config: LlamaServerConfig): Promise<ServerStatusResponse> {
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

export async function listLocalModels(): Promise<{ models: GGUFModelInfo[]; models_dir: string }> {
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

export async function savePreset(name: string, description: string, config: LlamaServerConfig): Promise<{ success: boolean }> {
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
