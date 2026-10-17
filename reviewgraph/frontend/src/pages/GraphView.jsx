import React, { useEffect, useRef, useState, useCallback } from 'react'
import cytoscape from 'cytoscape'
import {
  Network,
  Search,
  Loader2,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  Maximize,
  RefreshCw,
  Info,
} from 'lucide-react'
import { getProductGraph } from '../services/api'
import clsx from 'clsx'

// ── Cytoscape stylesheet ─────────────────────────────────────────────────────
const CY_STYLE = [
  {
    selector: 'node',
    style: {
      label: 'data(label)',
      'font-size': '10px',
      color: '#e5e7eb',
      'text-valign': 'bottom',
      'text-margin-y': '4px',
      'text-outline-color': '#0f172a',
      'text-outline-width': '2px',
      width: 30,
      height: 30,
      'border-width': 2,
      'border-color': '#374151',
    },
  },
  {
    selector: 'node[type = "product"]',
    style: {
      'background-color': '#6366f1',
      'border-color': '#818cf8',
      width: 40,
      height: 40,
      'font-size': '12px',
      'font-weight': 'bold',
    },
  },
  {
    selector: 'node[type = "reviewer"]',
    style: { 'background-color': '#10b981', 'border-color': '#34d399' },
  },
  {
    selector: 'node[type = "review"]',
    style: { 'background-color': '#f59e0b', 'border-color': '#fbbf24' },
  },
  {
    selector: 'node[type = "signal"]',
    style: { 'background-color': '#ef4444', 'border-color': '#f87171', shape: 'diamond' },
  },
  {
    selector: 'node.suspicious',
    style: { 'border-color': '#ef4444', 'border-width': 3 },
  },
  {
    selector: 'edge',
    style: {
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'arrow-scale': 1,
      'line-color': '#374151',
      'target-arrow-color': '#374151',
      width: 1.5,
      label: 'data(label)',
      'font-size': '8px',
      color: '#6b7280',
      'text-outline-color': '#0f172a',
      'text-outline-width': '1px',
    },
  },
  {
    selector: 'edge[weight]',
    style: { width: 'mapData(weight, 0, 1, 1, 4)' },
  },
  {
    selector: ':selected',
    style: {
      'border-color': '#a5b4fc',
      'border-width': 3,
      'line-color': '#a5b4fc',
      'target-arrow-color': '#a5b4fc',
    },
  },
]

const LAYOUTS = ['cose', 'circle', 'grid', 'breadthfirst']

const NODE_TYPE_COLORS = {
  product:  '#6366f1',
  reviewer: '#10b981',
  review:   '#f59e0b',
  signal:   '#ef4444',
}

function Legend() {
  return (
    <div className="flex flex-wrap gap-3">
      {Object.entries(NODE_TYPE_COLORS).map(([type, color]) => (
        <div key={type} className="flex items-center gap-1.5 text-xs text-gray-400">
          <span className="w-3 h-3 rounded-full inline-block" style={{ background: color }} />
          {type.charAt(0).toUpperCase() + type.slice(1)}
        </div>
      ))}
    </div>
  )
}

function NodePanel({ node, onClose }) {
  if (!node) return null
  const data = node.data()
  return (
    <div className="absolute right-4 top-4 w-64 bg-gray-900 border border-gray-700 rounded-xl p-4 shadow-2xl z-10">
      <div className="flex items-start justify-between mb-3">
        <span
          className="text-xs font-semibold uppercase tracking-wide px-2 py-0.5 rounded"
          style={{
            color: NODE_TYPE_COLORS[data.type] ?? '#9ca3af',
            background: `${NODE_TYPE_COLORS[data.type] ?? '#374151'}20`,
          }}
        >
          {data.type ?? 'node'}
        </span>
        <button
          onClick={onClose}
          className="text-gray-600 hover:text-gray-400 text-xs"
        >
          ✕
        </button>
      </div>
      <p className="text-sm font-medium text-gray-200 mb-2 break-all">{data.label ?? data.id}</p>
      {data.suspiciousness_score != null && (
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-gray-500">Suspiciousness</span>
          <span
            className={clsx(
              'font-semibold',
              data.suspiciousness_score > 0.6
                ? 'text-red-400'
                : data.suspiciousness_score > 0.3
                ? 'text-amber-400'
                : 'text-emerald-400'
            )}
          >
            {(data.suspiciousness_score * 100).toFixed(1)}%
          </span>
        </div>
      )}
      {data.rating != null && (
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-gray-500">Rating</span>
          <span className="text-amber-400">{'★'.repeat(Math.round(data.rating))} {data.rating}</span>
        </div>
      )}
      {data.review_count != null && (
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Reviews</span>
          <span className="text-gray-300">{data.review_count}</span>
        </div>
      )}
    </div>
  )
}

export default function GraphView() {
  const cyRef       = useRef(null)
  const cyInstance  = useRef(null)

  const urlParams = new URLSearchParams(window.location.search)
  const initialId = urlParams.get('product_id') || 'P001'

  const [productId, setProductId] = useState(initialId)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [layout, setLayout]       = useState('cose')
  const [selectedNode, setSelectedNode] = useState(null)
  const [stats, setStats]         = useState(null)

  const formatElements = useCallback((raw) => {
    let rawNodes = []
    let rawEdges = []

    if (Array.isArray(raw)) {
      rawNodes = raw.filter(el => !el.source && !el.data?.source)
      rawEdges = raw.filter(el => el.source || el.data?.source)
    } else if (raw && typeof raw === 'object') {
      rawNodes = raw.nodes || raw.elements?.nodes || []
      rawEdges = raw.edges || raw.elements?.edges || []
    }

    const cyNodes = []
    const validNodeIds = new Set()

    for (const n of rawNodes) {
      const d = n.data ? { ...n.data } : { ...n }
      const id = String(d.id || d.node_id || d.product_id || d.review_id || d.reviewer_id || d.signal_id || '')
      if (!id || validNodeIds.has(id)) continue
      validNodeIds.add(id)

      const rawType = String(d.type || d.label || 'node').toLowerCase()
      let normType = 'node'
      if (rawType.includes('product')) normType = 'product'
      else if (rawType.includes('reviewer')) normType = 'reviewer'
      else if (rawType.includes('review')) normType = 'review'
      else if (rawType.includes('signal') || rawType.includes('evidence')) normType = 'signal'

      cyNodes.push({
        data: {
          id,
          label: d.label || d.name || id,
          type: normType,
          rating: d.properties?.rating ?? d.rating,
          review_count: d.properties?.review_count ?? d.review_count,
          suspiciousness_score: d.properties?.suspiciousness_score ?? d.properties?.model_score ?? d.suspiciousness_score ?? d.model_score,
          ...d.properties,
        },
        classes: normType,
      })
    }

    const cyEdges = []
    let edgeCounter = 1
    for (const e of rawEdges) {
      const d = e.data ? { ...e.data } : { ...e }
      const source = String(d.source || d.from || d.start_node || '')
      const target = String(d.target || d.to || d.end_node || '')

      if (!source || !target || !validNodeIds.has(source) || !validNodeIds.has(target)) {
        continue
      }

      const id = String(d.id || `edge-${edgeCounter++}`)
      const relType = String(d.type || d.label || 'RELATED')

      cyEdges.push({
        data: {
          id,
          source,
          target,
          label: relType,
          type: relType,
          weight: d.properties?.score ?? d.properties?.similarity_score ?? d.weight ?? 1,
          ...d.properties,
        },
      })
    }

    return [...cyNodes, ...cyEdges]
  }, [])

  const initCy = useCallback((rawElements) => {
    if (cyInstance.current) cyInstance.current.destroy()
    if (!cyRef.current) return

    const elements = formatElements(rawElements)

    const cy = cytoscape({
      container: cyRef.current,
      elements,
      style: CY_STYLE,
      layout: { name: layout, animate: true, animationDuration: 600 },
      minZoom: 0.2,
      maxZoom: 5,
    })

    cy.on('tap', 'node', e => setSelectedNode(e.target))
    cy.on('tap', e => { if (e.target === cy) setSelectedNode(null) })

    cyInstance.current = cy
    setStats({ nodes: cy.nodes().length, edges: cy.edges().length })
  }, [layout, formatElements])

  const loadGraph = useCallback(async (idToLoad) => {
    const id = (idToLoad || productId || '').trim()
    if (!id) return
    setLoading(true)
    setError(null)
    setSelectedNode(null)
    try {
      const data = await getProductGraph(id)
      initCy(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [productId, initCy])

  async function handleLoad(e) {
    if (e) e.preventDefault()
    loadGraph(productId)
  }

  function reLayout() {
    if (!cyInstance.current) return
    cyInstance.current.layout({ name: layout, animate: true, animationDuration: 600 }).run()
  }

  function zoomIn()  { cyInstance.current?.zoom(cyInstance.current.zoom() * 1.3) }
  function zoomOut() { cyInstance.current?.zoom(cyInstance.current.zoom() / 1.3) }
  function fitAll()  { cyInstance.current?.fit(undefined, 30) }

  useEffect(() => {
    loadGraph(initialId)
    return () => { cyInstance.current?.destroy() }
  }, []) // eslint-disable-line

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-gray-100 mb-1">Knowledge Graph</h1>
        <p className="text-sm text-gray-500">
          Interactive Cytoscape.js visualization of product review networks
        </p>
      </div>

      {/* Toolbar */}
      <form onSubmit={handleLoad} className="card flex flex-wrap gap-3 items-end">
        <div className="flex-1 min-w-0">
          <label className="text-xs text-gray-500 mb-1.5 block font-medium">Product ID</label>
          <div className="relative">
            <Network
              size={15}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none"
            />
            <input
              type="text"
              value={productId}
              onChange={e => setProductId(e.target.value)}
              placeholder="e.g. B001E4KFG0"
              className="input-field w-full pl-8 text-sm"
            />
          </div>
        </div>

        <div>
          <label className="text-xs text-gray-500 mb-1.5 block font-medium">Layout</label>
          <select
            value={layout}
            onChange={e => setLayout(e.target.value)}
            className="input-field text-sm py-2"
          >
            {LAYOUTS.map(l => (
              <option key={l} value={l}>{l.charAt(0).toUpperCase() + l.slice(1)}</option>
            ))}
          </select>
        </div>

        <button
          type="submit"
          className="btn-primary flex items-center gap-2"
          disabled={loading}
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
          Load Graph
        </button>

        {/* Graph controls */}
        {cyInstance.current && (
          <>
            <button type="button" onClick={reLayout}  className="btn-secondary p-2" title="Re-layout"><RefreshCw size={14} /></button>
            <button type="button" onClick={zoomIn}    className="btn-secondary p-2" title="Zoom in"><ZoomIn size={14} /></button>
            <button type="button" onClick={zoomOut}   className="btn-secondary p-2" title="Zoom out"><ZoomOut size={14} /></button>
            <button type="button" onClick={fitAll}    className="btn-secondary p-2" title="Fit"><Maximize size={14} /></button>
          </>
        )}

        <div className="w-full flex items-center gap-2 pt-2 border-t border-gray-800">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Samples:</span>
          {['P001', 'P002', 'P003', 'P004', 'P005'].map(id => (
            <button
              key={id}
              type="button"
              onClick={() => { setProductId(id); loadGraph(id); }}
              className="px-2.5 py-1 bg-indigo-950/60 hover:bg-indigo-900 border border-indigo-700/60 rounded-full text-xs text-indigo-300 transition font-mono"
            >
              {id}
            </button>
          ))}
        </div>
      </form>

      {error && (
        <div className="card border-red-900 text-red-400 flex items-center gap-2">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {/* Legend + stats */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <Legend />
        {stats && (
          <span className="text-xs text-gray-600">
            {stats.nodes} nodes · {stats.edges} edges
          </span>
        )}
      </div>

      {/* Graph canvas */}
      <div className="card !p-0 relative overflow-hidden" style={{ height: 560 }}>
        <div ref={cyRef} className="cy-container" style={{ height: '100%' }} />

        {/* Node info panel */}
        <NodePanel node={selectedNode} onClose={() => setSelectedNode(null)} />

        {/* Idle overlay */}
        {!cyInstance.current && !loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-700 pointer-events-none">
            <Network size={48} className="mb-3" />
            <p className="text-sm">Load a product graph to visualize</p>
          </div>
        )}

        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-950/60">
            <Loader2 size={32} className="animate-spin text-indigo-400" />
          </div>
        )}
      </div>

      {/* Tip */}
      <p className="text-xs text-gray-600 flex items-center gap-1.5">
        <Info size={11} />
        Click any node to see details · Scroll to zoom · Drag to pan
      </p>
    </div>
  )
}
