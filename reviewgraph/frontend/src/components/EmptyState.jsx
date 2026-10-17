import React from 'react'

export default function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-16 px-6 space-y-4">
      {/* Icon */}
      {Icon && (
        <div className="flex items-center justify-center h-16 w-16 rounded-2xl bg-gray-800 border border-gray-700">
          <Icon size={32} className="text-gray-500" strokeWidth={1.5} />
        </div>
      )}

      {/* Title */}
      {title && (
        <h3 className="text-base font-semibold text-gray-300">{title}</h3>
      )}

      {/* Description */}
      {description && (
        <p className="text-sm text-gray-500 max-w-sm leading-relaxed">{description}</p>
      )}

      {/* Optional action button */}
      {action && (
        <button
          onClick={action.onClick}
          className="mt-2 inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-gray-950"
        >
          {action.label}
        </button>
      )}
    </div>
  )
}
