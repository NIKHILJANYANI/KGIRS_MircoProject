import React, { useEffect, useState } from 'react'

const SIZE_MAP = {
  sm: { svgSize: 80, radius: 30, strokeWidth: 6, fontSize: '0.6rem', levelSize: '0.5rem' },
  md: { svgSize: 120, radius: 45, strokeWidth: 8, fontSize: '0.9rem', levelSize: '0.65rem' },
  lg: { svgSize: 160, radius: 62, strokeWidth: 10, fontSize: '1.2rem', levelSize: '0.85rem' },
}

function getColor(score) {
  if (score >= 0.75) return '#ef4444' // red-500
  if (score >= 0.5)  return '#f97316' // orange-500
  if (score >= 0.25) return '#eab308' // yellow-500
  return '#22c55e'                    // green-500
}

function getLevel(score) {
  if (score >= 0.75) return 'Critical'
  if (score >= 0.5)  return 'High'
  if (score >= 0.25) return 'Medium'
  return 'Low'
}

export default function SuspiciousnessGauge({ score = 0, size = 'md', showLabel = true }) {
  const [animatedScore, setAnimatedScore] = useState(0)

  useEffect(() => {
    const timeout = setTimeout(() => setAnimatedScore(score), 100)
    return () => clearTimeout(timeout)
  }, [score])

  const cfg = SIZE_MAP[size] || SIZE_MAP.md
  const { svgSize, radius, strokeWidth } = cfg
  const center = svgSize / 2

  // Full arc is 270 degrees (from 135° to 405°, i.e. bottom-left to bottom-right going clockwise)
  const circumference = 2 * Math.PI * radius
  const arcFraction = 0.75 // 270/360
  const dashArray = circumference * arcFraction
  const dashOffset = dashArray - dashArray * animatedScore

  // Rotate so arc starts at 135° (bottom-left)
  const rotation = 135

  const color = getColor(animatedScore)
  const level = getLevel(animatedScore)
  const pct = Math.round(animatedScore * 100)

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: svgSize, height: svgSize }}>
        <svg
          width={svgSize}
          height={svgSize}
          viewBox={`0 0 ${svgSize} ${svgSize}`}
          className="block"
        >
          {/* Background track arc */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="#374151"
            strokeWidth={strokeWidth}
            strokeDasharray={`${dashArray} ${circumference}`}
            strokeDashoffset={0}
            strokeLinecap="round"
            transform={`rotate(${rotation} ${center} ${center})`}
          />
          {/* Foreground value arc */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={`${dashArray} ${circumference}`}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            transform={`rotate(${rotation} ${center} ${center})`}
            style={{ transition: 'stroke-dashoffset 0.8s ease, stroke 0.4s ease' }}
          />
        </svg>

        {/* Center text */}
        <div
          className="absolute inset-0 flex flex-col items-center justify-center"
          style={{ pointerEvents: 'none' }}
        >
          <span
            className="font-bold leading-none"
            style={{ fontSize: cfg.fontSize, color }}
          >
            {pct}%
          </span>
          <span
            className="text-gray-400 leading-none mt-0.5"
            style={{ fontSize: cfg.levelSize }}
          >
            {level}
          </span>
        </div>
      </div>

      {showLabel && (
        <span className="text-xs text-gray-400 tracking-wide">Suspiciousness Score</span>
      )}
    </div>
  )
}
