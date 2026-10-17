import React, { useState, useEffect, useCallback, Component } from 'react';
import {
  AlertTriangle, Info, Star, CheckCircle, Search,
  ChevronRight, Loader2, Package, Users, BarChart2,
  ArrowRight, Shield, Activity, ExternalLink, Clock,
} from 'lucide-react';
import { getProductInvestigation } from '../services/api';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error('Investigation error boundary caught:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-red-900/20 border border-red-700/50 rounded-2xl p-6 text-center space-y-3">
          <AlertTriangle className="w-8 h-8 text-red-400 mx-auto" />
          <p className="text-red-300 font-semibold">An error occurred displaying this section.</p>
          <p className="text-xs text-gray-400 font-mono">{this.state.error?.message || 'Render error'}</p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="px-4 py-1.5 bg-red-800/60 hover:bg-red-700 border border-red-600 rounded-lg text-xs text-white"
          >
            Retry Section
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v ?? 0));

function scoreColor(score) {
  if (score >= 0.75) return { ring: '#ef4444', text: 'text-red-400',   label: 'Critical',  badge: 'bg-red-900/40 text-red-300 border-red-700/50'    };
  if (score >= 0.50) return { ring: '#f97316', text: 'text-orange-400', label: 'High',      badge: 'bg-orange-900/40 text-orange-300 border-orange-700/50' };
  if (score >= 0.25) return { ring: '#eab308', text: 'text-yellow-400', label: 'Medium',    badge: 'bg-yellow-900/40 text-yellow-300 border-yellow-700/50' };
  return                   { ring: '#22c55e', text: 'text-green-400',  label: 'Low',       badge: 'bg-green-900/40 text-green-300 border-green-700/50'   };
}

function StarRating({ rating, count }) {
  const full = Math.round(clamp(rating, 0, 5));
  return (
    <span className="flex items-center gap-1">
      <span className="text-yellow-400 tracking-tight">
        {'★'.repeat(full)}{'☆'.repeat(5 - full)}
      </span>
      <span className="text-gray-400 text-sm">{Number(rating ?? 0).toFixed(1)}</span>
      {count !== undefined && <span className="text-gray-500 text-xs">({count} reviews)</span>}
    </span>
  );
}

// ─── Suspiciousness Gauge (SVG circle) ───────────────────────────────────────

function SuspiciousnessGauge({ score }) {
  const pct   = clamp(score, 0, 1);
  const info  = scoreColor(pct);
  const R     = 72;
  const cx    = 96;
  const cy    = 96;
  const circ  = 2 * Math.PI * R;
  const dash  = circ * pct;

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative w-48 h-48">
        <svg viewBox="0 0 192 192" className="w-full h-full -rotate-90">
          {/* track */}
          <circle cx={cx} cy={cy} r={R} fill="none" stroke="#1f2937" strokeWidth="14" />
          {/* fill */}
          <circle
            cx={cx} cy={cy} r={R} fill="none"
            stroke={info.ring}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circ}`}
            style={{ transition: 'stroke-dasharray 1s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <p className={`text-4xl font-black ${info.text}`}>
            {Math.round(pct * 100)}
            <span className="text-xl font-bold">%</span>
          </p>
          <p className="text-xs text-gray-500 mt-0.5">suspiciousness</p>
        </div>
      </div>
      <span className={`px-4 py-1 rounded-full text-sm font-bold border ${info.badge}`}>
        {info.label} Risk
      </span>
    </div>
  );
}

// ─── skeleton sections ────────────────────────────────────────────────────────

function Skeleton({ className = '' }) {
  return <div className={`bg-gray-700/60 rounded animate-pulse ${className}`} />;
}

function SkeletonSection() {
  return (
    <div className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-6 space-y-3">
      <Skeleton className="h-5 w-40" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-4 w-5/6" />
    </div>
  );
}

// ─── entity pill ─────────────────────────────────────────────────────────────

function EntityPill({ type, label }) {
  const styles = {
    product:  'bg-blue-900/50 text-blue-300 border-blue-700/60',
    review:   'bg-yellow-900/50 text-yellow-300 border-yellow-700/60',
    reviewer: 'bg-green-900/50 text-green-300 border-green-700/60',
    signal:   'bg-red-900/50 text-red-300 border-red-700/60',
    relation: 'bg-gray-700/60 text-gray-300 border-gray-600/60',
  };
  const displayLabel = typeof label === 'object'
    ? (label?.label || label?.id || label?.name || JSON.stringify(label))
    : String(label ?? '');
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold border ${styles[type] ?? styles.relation}`}>
      {displayLabel}
    </span>
  );
}

// ─── graph path renderer ──────────────────────────────────────────────────────

function parseStringPath(str) {
  if (!str || typeof str !== 'string') return [];
  const parts = str.split(/\s+[→←]\s+|->|<-/).map(s => s.trim()).filter(Boolean);
  return parts.map(part => {
    const lower = part.toLowerCase();
    let type = 'relation';
    if (lower.startsWith('product')) type = 'product';
    else if (lower.startsWith('review') && !lower.startsWith('reviewer')) type = 'review';
    else if (lower.startsWith('reviewer')) type = 'reviewer';
    else if (lower.includes('signal')) type = 'signal';
    return { type, label: part };
  });
}

function GraphPath({ path, index }) {
  let nodes = [];
  let conf = undefined;
  let basis = null;

  if (typeof path === 'string') {
    nodes = parseStringPath(path);
  } else if (path && typeof path === 'object') {
    if (Array.isArray(path.nodes) && path.nodes.length > 0) {
      nodes = path.nodes;
    } else if (Array.isArray(path.entities) && path.entities.length > 0) {
      nodes = path.entities;
    } else if (typeof path.path_text === 'string') {
      nodes = parseStringPath(path.path_text);
    } else if (typeof path.path === 'string') {
      nodes = parseStringPath(path.path);
    }
    conf = path.confidence ?? path.confidence_score;
    basis = path.basis ?? path.explanation;
  }

  return (
    <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <p className="text-xs text-gray-500 font-mono">Path {index + 1}</p>
        {conf !== undefined && (
          <span className="text-xs bg-indigo-900/40 text-indigo-300 border border-indigo-700/50 px-2 py-0.5 rounded-full">
            Confidence: {Math.round(conf * 100)}%
          </span>
        )}
      </div>

      {/* node chain */}
      <div className="flex flex-wrap items-center gap-1.5">
        {nodes.map((node, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <EntityPill type={node.type ?? 'relation'} label={node.label ?? node.id ?? String(node)} />
            {i < nodes.length - 1 && (
              <ArrowRight className="w-3.5 h-3.5 text-gray-600 flex-shrink-0" />
            )}
          </div>
        ))}
      </div>

      {basis && (
        <p className="text-xs text-gray-400 italic leading-relaxed border-l-2 border-indigo-700/50 pl-3">
          {basis}
        </p>
      )}
    </div>
  );
}

// ─── evidence signal card ─────────────────────────────────────────────────────

function SignalCard({ signal }) {
  const val = signal.value !== undefined ? signal.value : signal.signal_value;
  const isWarning = signal.is_warning ?? (
    (typeof val === 'number' && val > 0.5) ||
    String(signal.signal_type || signal.type || '').includes('HIGH') ||
    String(signal.signal_type || signal.type || '').includes('EXTREME')
  );
  return (
    <div className={`border rounded-xl p-4 space-y-2 ${
      isWarning
        ? 'bg-red-900/10 border-red-700/40'
        : 'bg-gray-800/40 border-gray-700/50'
    }`}>
      <div className="flex items-start gap-2">
        {isWarning
          ? <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
          : <Info className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
        }
        <div className="flex-1 min-w-0">
          <span className={`inline-block px-2 py-0.5 rounded-md text-xs font-bold mb-1 ${
            isWarning ? 'bg-red-800/50 text-red-300' : 'bg-blue-800/50 text-blue-300'
          }`}>
            {signal.type ?? signal.signal_type ?? 'Signal'}
          </span>
          <p className="text-sm font-semibold text-gray-200">
            {val !== undefined ? String(val) : '—'}
          </p>
          {signal.description && (
            <p className="text-xs text-gray-400 mt-0.5 leading-relaxed">
              {typeof signal.description === 'string' ? signal.description : JSON.stringify(signal.description)}
            </p>
          )}
          {signal.computation_method && (
            <p className="text-[10px] text-gray-600 mt-1 font-mono">
              {String(signal.computation_method)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── review row ───────────────────────────────────────────────────────────────

function ReviewRow({ review }) {
  const score   = review.model_score ?? review.suspiciousness_score ?? 0;
  const pct     = Math.min(100, Math.max(0, Math.round(score * 100)));
  const cls     = review.predicted_class ?? (review.is_fake ? 'Suspicious' : (score > 0.5 ? 'Suspicious' : 'Genuine'));
  const rawText = review.text ?? review.review_text ?? '';
  const excerpt = rawText ? rawText.slice(0, 100) + (rawText.length > 100 ? '…' : '') : '—';

  return (
    <div className="grid grid-cols-12 gap-3 items-center py-3 border-b border-gray-700/40 last:border-0 text-sm">
      <div className="col-span-2 text-gray-500 font-mono text-xs truncate" title={review.review_id}>
        {(review.review_id ?? review.id ?? '—').slice(0, 12)}…
      </div>
      <div className="col-span-1">
        <StarRating rating={review.rating} />
      </div>
      <div className="col-span-3">
        <div className="flex items-center gap-1.5">
          <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div className="h-full bg-red-500 rounded-full" style={{ width: `${pct}%` }} />
          </div>
          <span className="text-xs text-red-300 w-8 text-right">{pct}%</span>
        </div>
      </div>
      <div className="col-span-2">
        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${
          String(cls).toLowerCase().includes('sus')
            ? 'bg-red-900/40 text-red-300 border-red-700/50'
            : 'bg-green-900/40 text-green-300 border-green-700/50'
        }`}>{cls}</span>
      </div>
      <div className="col-span-3 text-xs text-gray-400 truncate" title={rawText}>{excerpt}</div>
      <div className="col-span-1 text-right">
        <a
          href={`/reviews?id=${review.review_id ?? review.id}`}
          className="text-indigo-400 hover:text-indigo-300 transition"
          title="View details"
        >
          <ExternalLink className="w-3.5 h-3.5 inline" />
        </a>
      </div>
    </div>
  );
}

// ─── main component ───────────────────────────────────────────────────────────

const RECENT_KEY = 'rg_recent_investigations';

export default function Investigation() {
  // read product_id from URL query params
  const urlParams = new URLSearchParams(window.location.search);
  const initialId = urlParams.get('product_id') ?? '';

  const [productId,    setProductId]    = useState(initialId);
  const [inputId,      setInputId]      = useState(initialId);
  const [data,         setData]         = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [recentIds,    setRecentIds]    = useState(() => {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]'); }
    catch { return []; }
  });

  const investigate = useCallback(async (id) => {
    if (!id?.trim()) return;
    setLoading(true);
    setError(null);
    setData(null);
    setProductId(id.trim());
    try {
      const result = await getProductInvestigation(id.trim());
      setData(result);
      // update recent list
      setRecentIds((prev) => {
        const updated = [id.trim(), ...prev.filter((r) => r !== id.trim())].slice(0, 8);
        localStorage.setItem(RECENT_KEY, JSON.stringify(updated));
        return updated;
      });
    } catch (err) {
      const status = err?.status ?? err?.response?.status;
      setError(status === 404
        ? `Product "${id.trim()}" was not found in the knowledge graph.`
        : err?.message ?? 'Failed to load investigation');
    } finally {
      setLoading(false);
    }
  }, []);

  // auto-investigate if URL has product_id
  useEffect(() => { if (initialId) investigate(initialId); }, []); // eslint-disable-line

  const handleSubmit = (e) => {
    e.preventDefault();
    investigate(inputId);
  };

  // derived data
  const product   = data?.product_info ?? data?.product ?? {};
  const score     = data?.suspiciousness?.score ?? data?.suspiciousness_score ?? data?.product?.suspiciousness_score ?? 0;
  const signals   = data?.evidence_signals ?? data?.signals ?? [];
  const susReviews = data?.suspicious_reviews ?? (data?.reviews?.filter(r => (r.predicted_class || '').toLowerCase().includes('sus') || (r.model_score && r.model_score > 0.5) || r.is_fake)) ?? [];
  const paths     = data?.reasoning_paths ?? data?.graph_paths ?? [];
  const connected = data?.connected_entities ?? {
    products: (data?.connected_products ?? []).map(p => ({
      product_id: p.other_product_id || p.product_id || p.id || (typeof p === 'string' ? p : 'Unknown'),
      title: p.other_product_name || p.name || p.title,
      avg_rating: p.avg_rating ?? p.other_avg_rating,
    })),
    reviewers: (data?.reviewer_overlaps ?? []).reduce((acc, o) => {
      if (o?.reviewer_id && !acc.some(r => r.reviewer_id === o.reviewer_id)) {
        acc.push({
          reviewer_id: o.reviewer_id,
          review_count: o.total_reviews ?? 1,
          suspiciousness: o.overlap_score ?? 0.5,
        });
      }
      return acc;
    }, []),
  };
  const limitations = data?.limitations ?? (Array.isArray(data?.explanation?.limitations) ? data.explanation.limitations : []);
  const explanation = typeof data?.explanation === 'string'
    ? data.explanation
    : (data?.explanation?.explanation_summary ?? data?.explanation_summary ?? data?.summary ?? '');

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-5xl mx-auto px-4 py-10 space-y-10">

        {/* ── header ── */}
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white flex items-center gap-2">
            <Shield className="w-7 h-7 text-indigo-400" />
            Product Investigation
          </h1>
          <p className="text-gray-400 mt-1 text-sm">
            Analyse a product's review network for fake review signals and graph-based evidence.
          </p>
        </div>

        {/* ── 1. PRODUCT SELECTOR ── */}
        <section className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-6 space-y-4">
          <form onSubmit={handleSubmit} className="flex gap-3 flex-wrap">
            <div className="relative flex-1 min-w-[220px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={inputId}
                onChange={(e) => setInputId(e.target.value)}
                placeholder="Enter product_id (e.g. B08N5WRWNW)"
                className="w-full bg-gray-800 border border-gray-700 text-white placeholder-gray-500
                           rounded-xl py-2.5 pl-10 pr-4 text-sm focus:outline-none focus:ring-2
                           focus:ring-indigo-500 transition"
              />
            </div>
            <button
              type="submit"
              disabled={loading || !inputId.trim()}
              className="flex items-center gap-1.5 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500
                         disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-sm font-semibold transition"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Shield className="w-4 h-4" />}
              Investigate
            </button>
          </form>

          {/* sample / recent products */}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs text-gray-500 uppercase tracking-wider">Samples:</span>
            {['P001', 'P002', 'P003', 'P004', 'P005'].map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => { setInputId(id); investigate(id); }}
                className="px-3 py-1 bg-indigo-950/60 hover:bg-indigo-900 border border-indigo-700/60 rounded-full text-xs text-indigo-300 transition font-mono"
              >
                {id}
              </button>
            ))}
            {recentIds.length > 0 && (
              <>
                <span className="text-xs text-gray-500 uppercase tracking-wider ml-2">Recent:</span>
                {recentIds.map((id) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => { setInputId(id); investigate(id); }}
                    className="px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-full text-xs text-gray-300 transition font-mono"
                  >
                    {id}
                  </button>
                ))}
              </>
            )}
          </div>
        </section>

        {/* ── loading ── */}
        {loading && (
          <div className="space-y-6">
            {[...Array(4)].map((_, i) => <SkeletonSection key={i} />)}
          </div>
        )}

        {/* ── error ── */}
        {!loading && error && (
          <div className="bg-red-900/20 border border-red-700/40 rounded-2xl p-8 text-center space-y-3">
            <AlertTriangle className="w-10 h-10 text-red-400 mx-auto" />
            <p className="text-red-300 font-medium">{error}</p>
            <p className="text-gray-500 text-sm">Try a different product ID, or check that the knowledge graph is populated.</p>
          </div>
        )}

        {/* ── results ── */}
        {!loading && data && (
          <ErrorBoundary>
            {/* ── 2. SUSPICIOUSNESS METER ── */}
            <section className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-gray-100 mb-6 flex items-center gap-2">
                <Activity className="w-5 h-5 text-red-400" /> Suspiciousness Score
              </h2>
              <div className="flex flex-col sm:flex-row items-center gap-8">
                <SuspiciousnessGauge score={score} />
                <div className="flex-1 space-y-3">
                  <p className="text-sm text-gray-400 leading-relaxed">
                    This score aggregates multiple graph-structural and ML signals to estimate the
                    probability that this product has been subjected to organised fake review activity.
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: 'Score',    value: (score * 100).toFixed(1) + '%' },
                      { label: 'Reviews',  value: (product.review_count ?? data.review_count ?? '—').toLocaleString() },
                      { label: 'Avg Rating', value: Number(product.avg_rating ?? product.average_rating ?? 0).toFixed(2) + ' ★' },
                      { label: 'Suspicious Reviews', value: susReviews.length },
                    ].map((s) => (
                      <div key={s.label} className="bg-gray-800/60 rounded-xl p-3">
                        <p className="text-xs text-gray-500">{s.label}</p>
                        <p className="text-lg font-bold text-white">{s.value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            {/* ── 3. PRODUCT INFO CARD ── */}
            {Object.keys(product).length > 0 && (
              <section className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-6 space-y-4">
                <h2 className="text-lg font-bold text-gray-100 flex items-center gap-2">
                  <Package className="w-5 h-5 text-blue-400" /> Product Information
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Product ID</p>
                    <p className="text-gray-200 font-mono">{product.product_id ?? productId}</p>
                  </div>
                  {product.title && (
                    <div className="sm:col-span-2">
                      <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Title</p>
                      <p className="text-gray-100 font-medium">{product.title}</p>
                    </div>
                  )}
                  {product.category && (
                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Category</p>
                      <p className="text-gray-300">{product.category}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wider mb-0.5">Average Rating</p>
                    <StarRating
                      rating={product.avg_rating ?? product.average_rating}
                      count={product.review_count}
                    />
                  </div>
                </div>

                {/* rating distribution bar chart */}
                {product.rating_distribution && (
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                      <BarChart2 className="w-3.5 h-3.5" /> Rating Distribution
                    </p>
                    <div className="space-y-1">
                      {[5, 4, 3, 2, 1].map((star) => {
                        const cnt = product.rating_distribution[star] ?? 0;
                        const total = Object.values(product.rating_distribution).reduce((a, b) => a + b, 0) || 1;
                        const pct = Math.round((cnt / total) * 100);
                        return (
                          <div key={star} className="flex items-center gap-2 text-xs">
                            <span className="text-yellow-400 w-3">{star}★</span>
                            <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                              <div className="h-full bg-yellow-500 rounded-full" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="text-gray-500 w-8 text-right">{pct}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* ── 4. EVIDENCE SIGNALS ── */}
            {signals.length > 0 && (
              <section className="space-y-4">
                <h2 className="text-lg font-bold text-gray-100 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-400" /> Evidence Signals
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {signals.map((sig, i) => <SignalCard key={i} signal={sig} />)}
                </div>
              </section>
            )}

            {/* ── 5. SUSPICIOUS REVIEWS ── */}
            {susReviews.length > 0 && (
              <section className="space-y-4">
                <h2 className="text-lg font-bold text-gray-100 flex items-center gap-2">
                  <Star className="w-5 h-5 text-red-400" /> Suspicious Reviews
                  <span className="text-sm font-normal text-gray-500">({susReviews.length})</span>
                </h2>
                <div className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-4 overflow-x-auto">
                  {/* table header */}
                  <div className="grid grid-cols-12 gap-3 text-[10px] uppercase tracking-wider text-gray-600 pb-2 border-b border-gray-700/40">
                    <span className="col-span-2">Review ID</span>
                    <span className="col-span-1">Rating</span>
                    <span className="col-span-3">ML Score</span>
                    <span className="col-span-2">Class</span>
                    <span className="col-span-3">Excerpt</span>
                    <span className="col-span-1 text-right">Link</span>
                  </div>
                  {susReviews.map((r, i) => <ReviewRow key={r.review_id ?? r.id ?? i} review={r} />)}
                </div>
              </section>
            )}

            {/* ── 6. GRAPH PATHS ── */}
            {paths.length > 0 && (
              <section className="space-y-4">
                <h2 className="text-lg font-bold text-gray-100 flex items-center gap-2">
                  <ChevronRight className="w-5 h-5 text-indigo-400" /> Reasoning Paths
                  <span className="text-xs text-indigo-400 bg-indigo-900/30 border border-indigo-700/40 px-2 py-0.5 rounded-full font-normal">
                    Graph Evidence
                  </span>
                </h2>
                <p className="text-sm text-gray-500">
                  Each path below shows a chain of graph relationships that contributed to the suspiciousness score.
                </p>
                <div className="space-y-3">
                  {paths.map((path, i) => <GraphPath key={i} path={path} index={i} />)}
                </div>
              </section>
            )}

            {/* ── 7. CONNECTED ENTITIES ── */}
            {(connected.products?.length > 0 || connected.reviewers?.length > 0) && (
              <section className="space-y-4">
                <h2 className="text-lg font-bold text-gray-100 flex items-center gap-2">
                  <Users className="w-5 h-5 text-green-400" /> Connected Entities
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">

                  {/* connected products */}
                  {connected.products?.length > 0 && (
                    <div className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-5 space-y-3">
                      <p className="text-sm font-semibold text-blue-300 flex items-center gap-1.5">
                        <Package className="w-4 h-4" /> Connected Products
                      </p>
                      <div className="space-y-2">
                        {connected.products.map((p, i) => {
                          const pid = typeof p === 'object' ? (p.product_id || p.other_product_id || p.id || 'Unknown') : String(p);
                          const title = typeof p === 'object' ? (p.title || p.name || p.other_product_name || '') : '';
                          return (
                            <div key={i} className="flex items-center justify-between text-xs text-gray-300 border-b border-gray-700/30 pb-2 last:border-0">
                              <div className="flex items-center gap-1.5 truncate">
                                <span className="font-mono text-gray-400">{pid}</span>
                                {title && <span className="text-gray-500 truncate max-w-[120px]">({title})</span>}
                              </div>
                              {p?.avg_rating !== undefined && (
                                <span className="text-yellow-400">{Number(p.avg_rating).toFixed(1)} ★</span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* connected reviewers */}
                  {connected.reviewers?.length > 0 && (
                    <div className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-5 space-y-3">
                      <p className="text-sm font-semibold text-green-300 flex items-center gap-1.5">
                        <Users className="w-4 h-4" /> Connected Reviewers
                      </p>
                      <div className="space-y-2">
                        {connected.reviewers.map((r, i) => {
                          const rid = typeof r === 'object' ? (r.reviewer_id || r.id || 'Unknown') : String(r);
                          return (
                            <div key={i} className="flex items-center justify-between text-xs text-gray-300 border-b border-gray-700/30 pb-2 last:border-0">
                              <span className="font-mono text-gray-400 truncate">{rid}</span>
                              <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                                {r?.review_count !== undefined && (
                                  <span className="text-gray-500">{r.review_count} reviews</span>
                                )}
                                {r?.suspiciousness !== undefined && (
                                  <span className={`${r.suspiciousness > 0.5 ? 'text-red-400' : 'text-green-400'}`}>
                                    {Math.round(r.suspiciousness * 100)}%
                                  </span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                </div>
              </section>
            )}

            {/* ── 8. LIMITATIONS ── */}
            <section>
              <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-2xl p-6 space-y-3">
                <h2 className="text-base font-bold text-yellow-300 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" /> Limitations & Alternative Interpretations
                </h2>
                <p className="text-xs text-yellow-400/70">
                  This system provides probabilistic evidence — it does not definitively prove fake review
                  activity. The following caveats apply:
                </p>
                <ul className="space-y-1.5">
                  {(limitations.length > 0 ? limitations : [
                    'ML classifier may produce false positives, especially for niche product categories.',
                    'Graph-structural signals (e.g. shared reviewers) do not conclusively indicate coordination.',
                    'Burstiness in reviews may result from legitimate product launches or promotions.',
                    'Suspiciousness score is a heuristic aggregate and should not be used as sole evidence.',
                    'The model was trained on historical data and may not generalise to new manipulation patterns.',
                  ]).map((lim, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-yellow-200/80">
                      <span className="text-yellow-500 mt-0.5 flex-shrink-0">•</span>
                      {typeof lim === 'string' ? lim : (lim.text || lim.description || JSON.stringify(lim))}
                    </li>
                  ))}
                </ul>
              </div>
            </section>

            {/* ── 9. EXPLANATION SUMMARY ── */}
            {explanation && (
              <section className="bg-gray-900/60 border border-indigo-700/40 rounded-2xl p-6 space-y-3">
                <h2 className="text-base font-bold text-indigo-300 flex items-center gap-2">
                  <CheckCircle className="w-5 h-5" /> System-Generated Explanation
                  <span className="text-xs font-normal text-gray-500 ml-1">(Evidence-Based)</span>
                </h2>
                <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                  {typeof explanation === 'string' ? explanation : JSON.stringify(explanation, null, 2)}
                </p>
              </section>
            )}
          </ErrorBoundary>
        )}

        {/* ── initial empty state ── */}
        {!loading && !data && !error && (
          <div className="text-center py-20 space-y-4">
            <Shield className="w-16 h-16 text-gray-700 mx-auto" />
            <p className="text-gray-500 font-medium">Enter a product ID above to begin investigation</p>
            <p className="text-gray-600 text-sm">
              The system will analyse the product's review network, apply ML scoring, and generate graph-based explanations.
            </p>
          </div>
        )}

      </div>
    </div>
  );
}
