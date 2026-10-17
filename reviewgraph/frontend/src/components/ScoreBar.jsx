import React, { useEffect, useState } from 'react'

const SCHEME_CLASSES = {
  red:   'bg-red-500',
  green: 'bg-green-500',
  blue:  'bg-blue-500',
}

function getAutoColor(value) {
  if (value > 0.7) return 'bg-red-500'
  if (value > 0.4) return 'bg-yellow-500'
  return 'bg-green-500'
}

export default function ScoreBar({
  value = 0,
  label,
  colorScheme = 'auto',
}) {
  const [animated, setAnimated] = useState(0)

  useEffect(() => {
    const t = setTimeout(() => setAnimated(value), 80)
    return () => clearTimeout(t)
  }, [value])

  const pct = Math.min(100, Math.max(0, Math.round(animated * 100)))
  const barColor =
    colorScheme === 'auto'
      ? getAutoColor(value)
      : SCHEME_CLASSES[colorScheme] || SCHEME_CLASSES.blue

  return (
    <div className="w-full space-y-1">
      {(label !== undefined) && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">{label}</span>
          <span className="text-xs font-semibold text-gray-300">{pct}%</span>
        </div>
      )}
      <div className="h-2 w-full bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
