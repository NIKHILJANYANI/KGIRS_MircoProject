import React, { useState } from 'react'
import { Star, Copy, Check, ExternalLink, AlertTriangle } from 'lucide-react'
import { normalizeReviewText } from '../utils/reviewText'

const CLASS_STYLES = {
  fake:    'bg-red-900/60 text-red-300 border border-red-700',
  genuine: 'bg-green-900/60 text-green-300 border border-green-700',
  OR:      'bg-red-900/60 text-red-300 border border-red-700',
  CG:      'bg-green-900/60 text-green-300 border border-green-700',
  '1':     'bg-red-900/60 text-red-300 border border-red-700',
  '0':     'bg-green-900/60 text-green-300 border border-green-700',
}

const DATASET_STYLES = {
  yelp:    'bg-blue-900/60 text-blue-300 border border-blue-700',
  amazon:  'bg-yellow-900/60 text-yellow-300 border border-yellow-700',
  default: 'bg-gray-800 text-gray-300 border border-gray-600',
}

function StarRating({ rating, max = 5 }) {
  const stars = Math.round(rating)
  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: max }).map((_, i) => (
        <Star
          key={i}
          size={13}
          className={i < stars ? 'text-yellow-400 fill-yellow-400' : 'text-gray-600'}
        />
      ))}
      <span className="ml-1 text-xs text-gray-400">{rating?.toFixed(1)}</span>
    </div>
  )
}

function ScoreBar({ value }) {
  const pct = Math.round((value ?? 0) * 100)
  const color =
    pct >= 70 ? 'bg-red-500' :
    pct >= 40 ? 'bg-yellow-500' :
    'bg-green-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-8 text-right">{pct}%</span>
    </div>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch (_) {}
  }
  return (
    <button
      onClick={handleCopy}
      className="ml-1 p-0.5 rounded text-gray-500 hover:text-gray-300 transition-colors"
      title="Copy ID"
    >
      {copied ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
    </button>
  )
}

function formatTimestamp(ts) {
  if (!ts) return null
  try {
    return new Date(ts).toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
    })
  } catch {
    return ts
  }
}

export default function ReviewCard({ review, onViewDetails, compact = false }) {
  const [expanded, setExpanded] = useState(false)

  if (!review) return null

  const {
    review_id,
    rating,
    review_text,
    predicted_class,
    model_score,
    dataset,
    timestamp,
    date,
    is_synthetic,
  } = review

  const text = normalizeReviewText(review_text || review.text || '')
  const excerpt = text.slice(0, 150)
  const hasMore = text.length > 150
  const classKey = String(predicted_class ?? '').toLowerCase()
  const classStyle = CLASS_STYLES[classKey] || CLASS_STYLES[predicted_class] || 'bg-gray-800 text-gray-300 border border-gray-600'
  const dsKey = (dataset || '').toLowerCase()
  const dsStyle = DATASET_STYLES[dsKey] || DATASET_STYLES.default
  const displayDate = formatTimestamp(timestamp || date)
  const shortId = review_id ? String(review_id).slice(0, 12) : '—'

  return (
    <div className={`rounded-xl bg-gray-900 border border-gray-700 hover:border-gray-600 transition-all ${compact ? 'p-3' : 'p-5'} space-y-3 flex flex-col`}>
      {/* Header row */}
      <div className="flex items-start justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-1 min-w-0">
          <span className="text-xs text-gray-500 font-mono truncate">{shortId}…</span>
          {review_id && <CopyButton text={String(review_id)} />}
          {is_synthetic && (
            <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] font-bold bg-yellow-900/70 text-yellow-300 border border-yellow-700 uppercase tracking-wider">
              Synthetic
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 flex-wrap">
          {dataset && (
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase ${dsStyle}`}>
              {dataset}
            </span>
          )}
          {predicted_class !== undefined && predicted_class !== null && (
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${classStyle}`}>
              {String(predicted_class)}
            </span>
          )}
        </div>
      </div>

      {/* Rating */}
      {rating !== undefined && rating !== null && (
        <StarRating rating={Number(rating)} />
      )}

      {/* Model score bar */}
      {model_score !== undefined && model_score !== null && !compact && (
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] text-gray-500 uppercase tracking-wide">Model Score</span>
          </div>
          <ScoreBar value={model_score} />
        </div>
      )}

      {/* Review text */}
      {text && (
        <div className="text-sm text-gray-300 leading-relaxed whitespace-pre-line">
          {expanded || !hasMore ? text : excerpt}
          {!expanded && hasMore && <span className="text-gray-500">…</span>}
          {hasMore && (
            <button
              onClick={() => setExpanded(e => !e)}
              className="ml-2 text-indigo-400 hover:text-indigo-300 text-xs font-medium transition-colors"
            >
              {expanded ? 'show less' : 'read more'}
            </button>
          )}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between mt-auto pt-1">
        {displayDate ? (
          <span className="text-[11px] text-gray-500">{displayDate}</span>
        ) : <span />}
        {onViewDetails && (
          <button
            onClick={() => onViewDetails(review)}
            className="flex items-center gap-1.5 text-xs font-medium text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            View Details <ExternalLink size={11} />
          </button>
        )}
      </div>
    </div>
  )
}
