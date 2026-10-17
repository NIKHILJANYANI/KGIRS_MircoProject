import React from 'react'
import { ChevronRight, ShieldAlert } from 'lucide-react'

const ENTITY_STYLES = {
  Product: 'bg-indigo-900 text-indigo-300 border border-indigo-700',
  Review: 'bg-amber-900 text-amber-300 border border-amber-700',
  Reviewer: 'bg-emerald-900 text-emerald-300 border border-emerald-700',
  EvidenceSignal: 'bg-red-900 text-red-300 border border-red-700',
}

function EntityPill({ entity }) {
  const style = ENTITY_STYLES[entity.type] || 'bg-gray-800 text-gray-300 border border-gray-600'
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold whitespace-nowrap ${style}`}
    >
      <span className="opacity-70 text-[10px] uppercase tracking-wider">{entity.type}</span>
      <span className="opacity-90">{entity.name || entity.id}</span>
    </span>
  )
}

function RelationshipArrow({ relationship }) {
  return (
    <div className="flex flex-col items-center mx-2 select-none">
      <span className="text-[10px] text-gray-400 mb-0.5 whitespace-nowrap">{relationship}</span>
      <ChevronRight className="text-gray-500" size={16} />
    </div>
  )
}

export default function EvidencePath({ path }) {
  if (!path) {
    return (
      <div className="text-gray-500 text-sm italic">No evidence path available.</div>
    )
  }

  const { entities = [], relationships = [], explanation, confidence } = path

  const confidenceColor =
    confidence >= 0.75 ? 'text-red-400' :
    confidence >= 0.5  ? 'text-orange-400' :
    confidence >= 0.25 ? 'text-yellow-400' :
    'text-green-400'

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-700 p-4 space-y-4">
      {/* Path visualization */}
      <div className="flex flex-wrap items-center gap-y-3">
        {entities.map((entity, idx) => (
          <React.Fragment key={idx}>
            <EntityPill entity={entity} />
            {idx < relationships.length && (
              <RelationshipArrow relationship={relationships[idx]} />
            )}
          </React.Fragment>
        ))}
      </div>

      {/* Explanation */}
      {explanation && (
        <p className="text-sm text-gray-400 border-t border-gray-700 pt-3 leading-relaxed">
          {explanation}
        </p>
      )}

      {/* Confidence badge */}
      {confidence !== undefined && (
        <div className="flex items-center gap-2">
          <ShieldAlert size={14} className={confidenceColor} />
          <span className={`text-xs font-semibold ${confidenceColor}`}>
            Confidence: {Math.round(confidence * 100)}%
          </span>
        </div>
      )}
    </div>
  )
}
