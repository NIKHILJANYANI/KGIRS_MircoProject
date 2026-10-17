import React from 'react'
import { AlertTriangle } from 'lucide-react'

const SIGNAL_STYLES = {
  HIGH_MODEL_SCORE:       { badge: 'bg-red-900/70 text-red-300 border-red-700',    label: 'High Model Score' },
  EXTREME_RATING:         { badge: 'bg-orange-900/70 text-orange-300 border-orange-700', label: 'Extreme Rating' },
  SHORT_REVIEW:           { badge: 'bg-yellow-900/70 text-yellow-300 border-yellow-700', label: 'Short Review' },
  HIGH_REVIEWER_VOLUME:   { badge: 'bg-blue-900/70 text-blue-300 border-blue-700',  label: 'High Reviewer Volume' },
  TEXT_SIMILARITY:        { badge: 'bg-purple-900/70 text-purple-300 border-purple-700', label: 'Text Similarity' },
}

const DEFAULT_STYLE = { badge: 'bg-gray-800 text-gray-300 border-gray-600', label: null }

function isSuspicious(signal) {
  const v = signal?.signal_value
  if (v === undefined || v === null) return false
  if (typeof v === 'number') return v >= 0.5
  if (typeof v === 'string') {
    const n = parseFloat(v)
    if (!isNaN(n)) return n >= 0.5
  }
  return false
}

function formatValue(v) {
  if (v === undefined || v === null) return '—'
  if (typeof v === 'number') {
    if (v <= 1 && v >= 0) return `${Math.round(v * 100)}%`
    return v.toLocaleString(undefined, { maximumFractionDigits: 3 })
  }
  return String(v)
}

export default function SignalCard({ signal }) {
  if (!signal) return null

  const { signal_type, signal_value, description, computation_method } = signal
  const cfg = SIGNAL_STYLES[signal_type] || DEFAULT_STYLE
  const displayLabel = cfg.label || signal_type?.replace(/_/g, ' ') || 'Unknown Signal'
  const suspicious = isSuspicious(signal)

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-700 hover:border-gray-600 transition-colors p-4 space-y-3 flex flex-col">
      {/* Header row */}
      <div className="flex items-center justify-between gap-2">
        <span className={`px-2.5 py-1 rounded-full text-[11px] font-semibold border ${cfg.badge}`}>
          {displayLabel}
        </span>
        {suspicious && (
          <AlertTriangle size={15} className="text-yellow-400 flex-shrink-0" />
        )}
      </div>

      {/* Value */}
      <div className="text-3xl font-bold text-white leading-none">
        {formatValue(signal_value)}
      </div>

      {/* Description */}
      {description && (
        <p className="text-sm text-gray-400 leading-relaxed">{description}</p>
      )}

      {/* Computation method */}
      {computation_method && (
        <p className="text-[11px] text-gray-600 italic border-t border-gray-800 pt-2">
          {computation_method}
        </p>
      )}
    </div>
  )
}
