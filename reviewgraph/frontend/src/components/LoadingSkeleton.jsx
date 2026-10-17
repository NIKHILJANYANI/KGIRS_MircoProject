import React from 'react'

/* Reusable pulse block */
function Pulse({ className = '' }) {
  return <div className={`animate-pulse rounded bg-gray-800 ${className}`} />
}

/* ── Stat Card Skeleton ────────────────────────────────────────────────── */
export function StatCardSkeleton() {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-700 p-5 space-y-3">
      <div className="flex items-center justify-between">
        <Pulse className="h-4 w-24" />
        <Pulse className="h-8 w-8 rounded-lg" />
      </div>
      <Pulse className="h-8 w-20" />
      <Pulse className="h-3 w-32" />
    </div>
  )
}

/* ── Review Card Skeleton ──────────────────────────────────────────────── */
export function ReviewCardSkeleton() {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-700 p-5 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <Pulse className="h-3 w-28" />
        <div className="flex gap-2">
          <Pulse className="h-5 w-14 rounded-full" />
          <Pulse className="h-5 w-14 rounded-full" />
        </div>
      </div>
      {/* Stars */}
      <Pulse className="h-3 w-20" />
      {/* Score bar */}
      <div className="space-y-1">
        <Pulse className="h-2 w-16" />
        <Pulse className="h-1.5 w-full rounded-full" />
      </div>
      {/* Text lines */}
      <div className="space-y-2">
        <Pulse className="h-3 w-full" />
        <Pulse className="h-3 w-5/6" />
        <Pulse className="h-3 w-4/6" />
      </div>
      {/* Footer */}
      <div className="flex items-center justify-between pt-1">
        <Pulse className="h-3 w-20" />
        <Pulse className="h-3 w-20" />
      </div>
    </div>
  )
}

/* ── Chart Skeleton ────────────────────────────────────────────────────── */
export function ChartSkeleton({ height = 'h-64' }) {
  return (
    <div className={`rounded-xl bg-gray-900 border border-gray-700 p-5 flex flex-col space-y-3 ${height}`}>
      {/* Title */}
      <Pulse className="h-4 w-40" />
      {/* Chart area */}
      <div className="flex-1 flex items-end gap-2">
        {[60, 40, 80, 55, 70, 45, 90, 65, 50, 75].map((h, i) => (
          <div key={i} className="flex-1 flex flex-col justify-end">
            <Pulse
              className="w-full rounded-t"
              style={{ height: `${h}%`, animationDelay: `${i * 80}ms` }}
            />
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Table Skeleton ────────────────────────────────────────────────────── */
export function TableSkeleton({ rows = 5 }) {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="flex gap-4 px-5 py-3 border-b border-gray-700">
        {[28, 16, 20, 14, 14].map((w, i) => (
          <Pulse key={i} className={`h-3 flex-none`} style={{ width: `${w}%` }} />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIdx) => (
        <div
          key={rowIdx}
          className="flex gap-4 px-5 py-4 border-b border-gray-800 last:border-0"
          style={{ animationDelay: `${rowIdx * 60}ms` }}
        >
          {[28, 16, 20, 14, 14].map((w, colIdx) => (
            <Pulse
              key={colIdx}
              className="h-3 flex-none"
              style={{ width: `${w}%`, animationDelay: `${(rowIdx * 5 + colIdx) * 40}ms` }}
            />
          ))}
        </div>
      ))}
    </div>
  )
}

/* ── Default export convenience wrapper ───────────────────────────────── */
export default function LoadingSkeleton({ type = 'card', rows, height }) {
  switch (type) {
    case 'stat':    return <StatCardSkeleton />
    case 'review':  return <ReviewCardSkeleton />
    case 'chart':   return <ChartSkeleton height={height} />
    case 'table':   return <TableSkeleton rows={rows} />
    default:        return <ReviewCardSkeleton />
  }
}
