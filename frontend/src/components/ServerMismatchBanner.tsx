/**
 * ServerMismatchBanner
 *
 * Displays an amber warning banner when a local LLM server (MLX or llama.cpp)
 * is running but the Settings model doesn't match, meaning the model is
 * consuming RAM without being used.
 */

import { AlertTriangle, X, Settings, Square } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import type { ServerMismatchWarning } from '../hooks/useServerMismatch'

interface ServerMismatchBannerProps {
  warnings: ServerMismatchWarning[]
  onDismiss: (type: 'mlx' | 'llamacpp') => void
}

export function ServerMismatchBanner({ warnings, onDismiss }: ServerMismatchBannerProps) {
  const navigate = useNavigate()

  if (warnings.length === 0) return null

  return (
    <div className="flex flex-col gap-0">
      {warnings.map((warning) => (
        <div
          key={warning.type}
          className="bg-amber-50 dark:bg-amber-900/20 border-b border-amber-200 dark:border-amber-800 px-4 py-2.5 flex items-center gap-3"
        >
          <AlertTriangle size={18} className="text-amber-500 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-amber-800 dark:text-amber-200">
              <span className="font-semibold">{warning.serverName} server is running</span>
              {warning.loadedModel && (
                <span className="hidden sm:inline"> with <code className="text-xs bg-amber-100 dark:bg-amber-900/40 px-1 py-0.5 rounded">{warning.loadedModel}</code></span>
              )}
              {' '}but Settings use <span className="font-semibold">{warning.currentSettingDisplay}</span>.
              <span className="hidden md:inline text-amber-600 dark:text-amber-400"> The model is consuming RAM without being used.</span>
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={() => navigate('/settings')}
              className="text-xs px-2.5 py-1.5 bg-amber-100 dark:bg-amber-800/40 text-amber-700 dark:text-amber-300 rounded-md hover:bg-amber-200 dark:hover:bg-amber-800/60 transition-colors flex items-center gap-1.5 font-medium"
            >
              <Settings size={13} />
              Fix in Settings
            </button>
            <button
              onClick={() => navigate(warning.type === 'mlx' ? '/admin/mlx' : '/admin/llamacpp')}
              className="text-xs px-2.5 py-1.5 bg-amber-100 dark:bg-amber-800/40 text-amber-700 dark:text-amber-300 rounded-md hover:bg-amber-200 dark:hover:bg-amber-800/60 transition-colors flex items-center gap-1.5 font-medium"
            >
              <Square size={13} />
              Stop Server
            </button>
            <button
              onClick={() => onDismiss(warning.type)}
              className="p-1 text-amber-400 dark:text-amber-500 hover:text-amber-600 dark:hover:text-amber-300 transition-colors rounded"
              title="Dismiss"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

export default ServerMismatchBanner
