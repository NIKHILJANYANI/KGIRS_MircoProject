import React from 'react'

const STATUS_CONFIG = {
  ok: {
    dotClass: 'bg-green-400',
    pingClass: 'bg-green-400',
    textClass: 'text-green-400',
    defaultLabel: 'Connected',
  },
  unavailable: {
    dotClass: 'bg-red-500',
    pingClass: 'bg-red-500',
    textClass: 'text-red-400',
    defaultLabel: 'Unavailable',
  },
  not_initialized: {
    dotClass: 'bg-gray-500',
    pingClass: 'bg-gray-500',
    textClass: 'text-gray-400',
    defaultLabel: 'Not Initialized',
  },
  degraded: {
    dotClass: 'bg-yellow-400',
    pingClass: 'bg-yellow-400',
    textClass: 'text-yellow-400',
    defaultLabel: 'Degraded',
  },
}

export default function StatusBadge({ status = 'not_initialized', label }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.not_initialized
  const displayLabel = label || cfg.defaultLabel
  const animate = status === 'ok' || status === 'degraded'

  return (
    <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-gray-800 border border-gray-700 text-xs font-medium">
      {/* Dot with optional pulse for live statuses */}
      <span className="relative flex h-2 w-2">
        {animate && (
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-60 ${cfg.pingClass}`}
          />
        )}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${cfg.dotClass}`} />
      </span>
      <span className={cfg.textClass}>{displayLabel}</span>
    </span>
  )
}
