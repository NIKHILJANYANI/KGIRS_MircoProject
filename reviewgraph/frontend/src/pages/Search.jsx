import { useState, useCallback } from 'react';
import {
  Search as SearchIcon, Type, Brain, Network, Zap,
  Package, Star, AlertTriangle, ChevronLeft, ChevronRight,
  Clock, ArrowRight, Loader2,
} from 'lucide-react';
import { searchReviews } from '../services/api';

// ─── constants ───────────────────────────────────────────────────────────────

const MODES = [
  {
    id: 'keyword',
    label: 'Keyword',
    icon: Type,
    color: 'blue',
    banner: 'bg-blue-900/30 border-blue-700/50 text-blue-300',
    active: 'bg-blue-600 text-white',
    inactive: 'text-blue-400 border-blue-700/40 hover:bg-blue-900/30',
    description: 'Searches review text and product titles using full-text BM25 search over indexed terms.',
  },
  {
    id: 'semantic',
    label: 'Semantic',
    icon: Brain,
    color: 'purple',
    banner: 'bg-purple-900/30 border-purple-700/50 text-purple-300',
    active: 'bg-purple-600 text-white',
    inactive: 'text-purple-400 border-purple-700/40 hover:bg-purple-900/30',
    description: 'Uses sentence embeddings (SBERT) and FAISS to find semantically similar reviews, even when exact words differ.',
  },
  {
    id: 'graph',
    label: 'Graph',
    icon: Network,
    color: 'indigo',
    banner: 'bg-indigo-900/30 border-indigo-700/50 text-indigo-300',
    active: 'bg-indigo-600 text-white',
    inactive: 'text-indigo-400 border-indigo-700/40 hover:bg-indigo-900/30',
    description: 'Traverses the Neo4j knowledge graph to surface connected entities — products, reviewers, and signals — related to your query.',
  },
  {
    id: 'hybrid',
    label: 'Hybrid',
    icon: Zap,
    color: 'gradient',
    banner: 'bg-gradient-to-r from-indigo-900/30 to-purple-900/30 border-indigo-600/50 text-indigo-200',
    active: 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white',
    inactive: 'text-indigo-300 border-indigo-600/40 hover:bg-indigo-900/20',
    description: 'Combines keyword BM25, semantic SBERT, and graph traversal signals and re-ranks with reciprocal rank fusion for best overall results.',
  },
];

const EXAMPLE_SEARCHES = [
  'wireless earbuds noise cancelling',
  'suspicious five star review pattern',
  'duplicate reviews same product',
  'fake reviewer network',
  'battery life laptop complaints',
];

// ─── helpers ─────────────────────────────────────────────────────────────────

function scoreBar(value, color) {
  const norm = value > 1 ? value / (1 + value) : Math.max(0, Math.min(1, value ?? 0));
  const pct = Math.round(norm * 100);
  const bg = {
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    indigo: 'bg-indigo-500',
  }[color] ?? 'bg-gray-500';
  return (
    <div className="flex items-center gap-1.5 w-full">
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${bg} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-gray-500 w-7 text-right">{pct}%</span>
    </div>
  );
}

function starRating(rating) {
  const full = Math.round(rating ?? 0);
  return (
    <span className="text-yellow-400 text-xs">
      {'★'.repeat(Math.min(full, 5))}{'☆'.repeat(Math.max(0, 5 - full))}
    </span>
  );
}

// ─── skeleton cards ───────────────────────────────────────────────────────────

function SkeletonResultCard() {
  return (
    <div className="bg-gray-800/60 border border-gray-700/50 rounded-xl p-5 animate-pulse space-y-3">
      <div className="flex items-center gap-3">
        <div className="h-5 w-16 bg-gray-700 rounded-full" />
        <div className="h-5 w-48 bg-gray-700 rounded" />
      </div>
      <div className="h-4 w-full bg-gray-700 rounded" />
      <div className="h-4 w-3/4 bg-gray-700 rounded" />
      <div className="flex gap-4">
        <div className="h-3 w-24 bg-gray-700 rounded" />
        <div className="h-3 w-24 bg-gray-700 rounded" />
        <div className="h-3 w-24 bg-gray-700 rounded" />
      </div>
    </div>
  );
}

// ─── result card ──────────────────────────────────────────────────────────────

function ResultCard({ result, rank }) {
  const isProduct = result.type === 'product';
  const title = result.title
    ?? (result.text ? result.text.slice(0, 80) + (result.text.length > 80 ? '…' : '') : result.id);

  const excerpt = result.text
    ? result.text.slice(0, 150) + (result.text.length > 150 ? '…' : '')
    : null;

  const rawScore = result.combined_score ?? result.score ?? 0;
  const normalized = rawScore > 1 ? rawScore / (1 + rawScore) : Math.max(0, Math.min(1, rawScore));
  const relevancePct = Math.round(normalized * 100);
  const modelScore = result.model_score;

  const modelBadgeColor = modelScore !== undefined
    ? modelScore > 0.7
      ? 'bg-red-800/50 text-red-300 border-red-700/50'
      : modelScore > 0.4
        ? 'bg-yellow-800/50 text-yellow-300 border-yellow-700/50'
        : 'bg-green-800/50 text-green-300 border-green-700/50'
    : null;

  const modelLabel = modelScore !== undefined
    ? modelScore > 0.7 ? 'Suspicious' : modelScore > 0.4 ? 'Uncertain' : 'Genuine'
    : null;

  return (
    <div className="bg-gray-900/70 border border-gray-700/50 rounded-xl p-5 hover:border-indigo-600/50 transition group space-y-3">
      {/* header */}
      <div className="flex items-start gap-3 justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-gray-500 font-mono">#{rank}</span>
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold border ${
            isProduct
              ? 'bg-blue-900/40 text-blue-300 border-blue-700/50'
              : 'bg-yellow-900/40 text-yellow-300 border-yellow-700/50'
          }`}>
            {isProduct ? <Package className="w-3 h-3" /> : <Star className="w-3 h-3" />}
            {isProduct ? 'Product' : 'Review'}
          </span>
          {modelBadgeColor && (
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold border ${modelBadgeColor}`}>
              <AlertTriangle className="w-3 h-3" />
              {modelLabel}
            </span>
          )}
        </div>
        {/* combined score */}
        <div className="text-right flex-shrink-0">
          <p className="text-2xl font-black text-indigo-400 leading-none">
            {relevancePct}
            <span className="text-sm font-normal text-gray-500">%</span>
          </p>
          <p className="text-[10px] text-gray-500 uppercase tracking-wide">relevance</p>
        </div>
      </div>

      {/* title */}
      <p className="text-sm font-semibold text-gray-100 group-hover:text-white transition leading-snug">
        {title}
      </p>

      {/* excerpt */}
      {excerpt && (
        <p className="text-xs text-gray-400 leading-relaxed">{excerpt}</p>
      )}

      {/* sub-scores */}
      <div className="space-y-1 pt-1">
        {result.keyword_score  !== undefined && <div className="flex items-center gap-2"><span className="text-[10px] text-gray-500 w-16 flex-shrink-0">Keyword</span>{scoreBar(result.keyword_score, 'blue')}</div>}
        {result.semantic_score !== undefined && <div className="flex items-center gap-2"><span className="text-[10px] text-gray-500 w-16 flex-shrink-0">Semantic</span>{scoreBar(result.semantic_score, 'purple')}</div>}
        {result.graph_score    !== undefined && <div className="flex items-center gap-2"><span className="text-[10px] text-gray-500 w-16 flex-shrink-0">Graph</span>{scoreBar(result.graph_score, 'indigo')}</div>}
      </div>

      {/* model score row */}
      {modelScore !== undefined && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-500 w-16 flex-shrink-0">ML Score</span>
          <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-red-500 rounded-full transition-all"
              style={{ width: `${Math.round(modelScore * 100)}%` }}
            />
          </div>
          <span className="text-[10px] text-red-300 w-7 text-right">{Math.round(modelScore * 100)}%</span>
        </div>
      )}

      {/* footer link */}
      <div className="pt-1">
        <a
          href={isProduct ? `/investigation?product_id=${result.id}` : `/reviews?id=${result.id}`}
          className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition font-medium"
        >
          {isProduct ? 'Investigate product' : 'View review details'}
          <ArrowRight className="w-3 h-3" />
        </a>
      </div>
    </div>
  );
}

// ─── main component ───────────────────────────────────────────────────────────

export default function Search() {
  const [query,       setQuery]       = useState('');
  const [mode,        setMode]        = useState('hybrid');
  const [results,     setResults]     = useState([]);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);
  const [page,        setPage]        = useState(1);
  const [totalPages,  setTotalPages]  = useState(1);
  const [totalCount,  setTotalCount]  = useState(0);
  const [latencyMs,   setLatencyMs]   = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  const PAGE_SIZE = 10;

  const doSearch = useCallback(async (q = query, p = 1, m = mode) => {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setHasSearched(true);
    const t0 = performance.now();
    try {
      const data = await searchReviews({ query: q, mode: m, page: p, page_size: PAGE_SIZE });
      const elapsed = performance.now() - t0;
      setResults(data.results ?? []);
      setTotalCount(data.total ?? data.results?.length ?? 0);
      setTotalPages(Math.max(1, Math.ceil((data.total ?? data.results?.length ?? 0) / PAGE_SIZE)));
      setPage(p);
      setLatencyMs(Math.round(elapsed));
    } catch (err) {
      setError(err?.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  }, [query, mode]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    doSearch(query, 1, mode);
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    if (hasSearched && query.trim()) doSearch(query, 1, newMode);
  };

  const handleExample = (ex) => {
    setQuery(ex);
    doSearch(ex, 1, mode);
  };

  const activeModeObj = MODES.find((m) => m.id === mode);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-4xl mx-auto px-4 py-10 space-y-8">

        {/* ── header ── */}
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Review Search</h1>
          <p className="text-gray-400 mt-1 text-sm">Find products and reviews using keyword, semantic, or graph-based retrieval.</p>
        </div>

        {/* ── 1. SEARCH BAR ── */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative flex items-center">
            <SearchIcon className="absolute left-4 w-5 h-5 text-gray-400 pointer-events-none" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search reviews, products, reviewers…"
              className="w-full bg-gray-800/80 border border-gray-700 text-white placeholder-gray-500
                         rounded-xl py-3.5 pl-12 pr-32 text-sm focus:outline-none focus:ring-2
                         focus:ring-indigo-500 focus:border-transparent transition"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-2 flex items-center gap-1.5 px-4 py-2 bg-indigo-600 hover:bg-indigo-500
                         disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm font-semibold
                         transition text-white"
            >
              {loading
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <SearchIcon className="w-4 h-4" />}
              {loading ? 'Searching' : 'Search'}
            </button>
          </div>

          {/* mode selector */}
          <div className="flex flex-wrap gap-2">
            {MODES.map((m) => {
              const Icon = m.icon;
              const isActive = mode === m.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => handleModeChange(m.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium border transition
                    ${isActive ? m.active + ' border-transparent' : m.inactive + ' bg-transparent border'}`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {m.label}
                </button>
              );
            })}
          </div>
        </form>

        {/* ── 2. MODE BANNER ── */}
        {activeModeObj && (
          <div className={`flex items-start gap-3 border rounded-xl px-4 py-3 text-sm ${activeModeObj.banner}`}>
            <activeModeObj.icon className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <p>{activeModeObj.description}</p>
          </div>
        )}

        {/* ── LATENCY ── */}
        {latencyMs !== null && hasSearched && !loading && (
          <div className="flex items-center gap-1.5 text-xs text-gray-500">
            <Clock className="w-3.5 h-3.5" />
            Retrieved <span className="text-gray-300 font-medium">{totalCount}</span> results in{' '}
            <span className="text-gray-300 font-medium">{latencyMs} ms</span>
          </div>
        )}

        {/* ── ERROR ── */}
        {error && (
          <div className="bg-red-900/20 border border-red-700/40 rounded-xl p-4 flex items-center gap-3 text-sm text-red-300">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* ── 3. RESULTS / STATES ── */}
        {loading ? (
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => <SkeletonResultCard key={i} />)}
          </div>
        ) : !hasSearched ? (
          /* empty / example state */
          <div className="text-center py-16 space-y-6">
            <SearchIcon className="w-14 h-14 text-gray-700 mx-auto" />
            <p className="text-gray-500 text-sm">Start typing to search, or try one of these examples:</p>
            <div className="flex flex-wrap justify-center gap-2">
              {EXAMPLE_SEARCHES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => handleExample(ex)}
                  className="px-4 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-full text-sm text-gray-300 transition"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : results.length === 0 ? (
          <div className="text-center py-16 space-y-3">
            <SearchIcon className="w-14 h-14 text-gray-700 mx-auto" />
            <p className="text-gray-400 font-medium">No results found for "{query}"</p>
            <p className="text-gray-600 text-sm">Try a different query or switch modes.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {results.map((r, i) => (
              <ResultCard key={r.id ?? i} result={r} rank={(page - 1) * PAGE_SIZE + i + 1} />
            ))}
          </div>
        )}

        {/* ── 4. PAGINATION ── */}
        {!loading && results.length > 0 && totalPages > 1 && (
          <div className="flex items-center justify-center gap-4 pt-2">
            <button
              onClick={() => doSearch(query, page - 1, mode)}
              disabled={page <= 1}
              className="flex items-center gap-1.5 px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-40
                         disabled:cursor-not-allowed border border-gray-700 rounded-lg text-sm font-medium transition"
            >
              <ChevronLeft className="w-4 h-4" /> Previous
            </button>
            <span className="text-sm text-gray-400">
              Page <span className="text-white font-semibold">{page}</span> of{' '}
              <span className="text-white font-semibold">{totalPages}</span>
            </span>
            <button
              onClick={() => doSearch(query, page + 1, mode)}
              disabled={page >= totalPages}
              className="flex items-center gap-1.5 px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-40
                         disabled:cursor-not-allowed border border-gray-700 rounded-lg text-sm font-medium transition"
            >
              Next <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
