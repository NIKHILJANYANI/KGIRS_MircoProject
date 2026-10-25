import { useEffect, useState } from 'react';
import { useSearchParams, useParams, Link } from 'react-router-dom';
import { getReview, getRelatedReviews } from '../services/api';
import {
  FileText,
  Star,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  ExternalLink,
  Loader2,
  Clock,
  Hash,
  Database,
  ShieldAlert,
  BarChart2,
  Layers,
  Search,
} from 'lucide-react';
import { normalizeReviewText } from '../utils/reviewText';

const REVIEW_RECOMMENDATIONS = ['AMZ23_ELEC_0001778'];

// ─── Helpers ─────────────────────────────────────────────────────────────────
function StarRating({ rating }) {
  const stars = Math.round(Number(rating) || 0);
  return (
    <div className="flex gap-0.5" aria-label={`${stars} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star
          key={i}
          className={`w-4 h-4 ${i <= stars ? 'text-amber-400 fill-amber-400' : 'text-gray-600'}`}
        />
      ))}
    </div>
  );
}

function LabelBadge({ label }) {
  if (!label) return null;
  const styles = {
    Suspicious: 'bg-red-900 text-red-300 border-red-700',
    Genuine: 'bg-emerald-900 text-emerald-300 border-emerald-700',
    Unlabeled: 'bg-gray-800 text-gray-400 border-gray-600',
    __OR__: 'bg-orange-900 text-orange-300 border-orange-700',
    __CG__: 'bg-emerald-900 text-emerald-300 border-emerald-700',
  };
  const cls = styles[label] || styles['Unlabeled'];
  return (
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${cls}`}>
      {label}
    </span>
  );
}

function PredictedBadge({ cls }) {
  if (!cls) return null;
  const isFraud =
    cls === 'Suspicious' ||
    cls === '__OR__' ||
    cls?.toLowerCase().includes('spam') ||
    cls?.toLowerCase().includes('fake');
  return (
    <span
      className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
        isFraud
          ? 'bg-red-900 text-red-300 border-red-700'
          : 'bg-emerald-900 text-emerald-300 border-emerald-700'
      }`}
    >
      {isFraud ? '⚠ ' : '✓ '}
      {cls}
    </span>
  );
}

function ScoreBar({ score }) {
  const pct = Math.min(1, Math.max(0, Number(score) || 0)) * 100;
  const color =
    pct > 70 ? 'from-red-600 to-red-400' : pct > 40 ? 'from-orange-500 to-amber-400' : 'from-emerald-600 to-emerald-400';
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>Suspicion Score</span>
        <span className="font-mono font-bold text-gray-200">{Number(score || 0).toFixed(4)}</span>
      </div>
      <div className="h-3 w-full bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full bg-gradient-to-r ${color} rounded-full transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function SkeletonBlock({ h = 'h-6', w = 'w-full' }) {
  return <div className={`${h} ${w} bg-gray-800 animate-pulse rounded`} />;
}

function Skeleton() {
  return (
    <div className="space-y-6 p-6">
      <SkeletonBlock h="h-8" w="w-64" />
      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-4">
          <SkeletonBlock h="h-48" />
          <SkeletonBlock h="h-28" />
        </div>
        <div className="space-y-4">
          <SkeletonBlock h="h-40" />
          <SkeletonBlock h="h-32" />
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value, mono = false }) {
  return (
    <div className="flex justify-between items-start gap-4 py-1.5 border-b border-gray-800 last:border-0 text-sm">
      <span className="text-gray-500 shrink-0">{label}</span>
      <span className={`text-gray-200 text-right break-all ${mono ? 'font-mono text-xs' : ''}`}>
        {value ?? '—'}
      </span>
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function ReviewDetails() {
  const [params, setParams] = useSearchParams();
  const routeParams = useParams();

  const urlId = params.get('id') || params.get('review_id') || routeParams.id || '';
  const [reviewId, setReviewId] = useState(urlId);
  const [inputId, setInputId] = useState(urlId);

  const [review, setReview] = useState(null);
  const [related, setRelated] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Sync if URL param changes
  useEffect(() => {
    if (urlId && urlId !== reviewId) {
      setReviewId(urlId);
      setInputId(urlId);
    }
  }, [urlId]);

  const loadReview = async (id) => {
    const targetId = (id || reviewId || '').trim();
    if (!targetId) return;
    setLoading(true);
    setError(null);
    try {
      const [rev, rel] = await Promise.all([
        getReview(targetId),
        getRelatedReviews(targetId).catch(() => []),
      ]);
      const rawReview = rev?.review || rev;
      const livePred = rev?.live_prediction || rawReview?.live_prediction;
      const sigs = rev?.signals || rawReview?.signals || [];

      const unwrapped = {
        ...rawReview,
        review_id: rawReview?.review_id || targetId,
        reviewer_id: rawReview?.reviewer_id,
        product_id: rawReview?.product_id,
        rating: rawReview?.rating,
        review_text: rawReview?.review_text || rawReview?.text || '',
        dataset_label: rawReview?.dataset_label || (rawReview?.is_synthetic ? 'Synthetic' : 'Unlabeled'),
        is_synthetic: rawReview?.is_synthetic,
        ml_prediction: livePred || {
          score: rawReview?.model_score,
          predicted_class: rawReview?.predicted_class,
          model_version: rawReview?.model_version || '1.0.0',
        },
        evidence_signals: sigs,
      };
      setReview(unwrapped);
      setRelated(rel?.items || (Array.isArray(rel) ? rel : []));
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || `No review with ID "${targetId}" exists.`);
      setReview(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReview(reviewId);
  }, [reviewId]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputId.trim()) return;
    setReviewId(inputId.trim());
    setParams({ id: inputId.trim() });
  };

  const handleSelectSample = (id) => {
    setInputId(id);
    setReviewId(id);
    setParams({ id });
  };

  // Destructure fields (handles both flat and nested API shapes)
  const currentReview = review || {};
  const {
    review_id = reviewId,
    reviewer_id,
    product_id,
    rating,
    review_text,
    review_title,
    timestamp,
    dataset_label,
    is_synthetic,
    source_dataset,
    ml_prediction,
    evidence_signals,
    graph_paths,
  } = currentReview;

  const score = ml_prediction?.score ?? ml_prediction?.model_score ?? currentReview.model_score ?? currentReview.fraud_score ?? currentReview.suspicion_score ?? 0;
  const predicted_class =
    ml_prediction?.predicted_class ??
    currentReview.predicted_class ??
    currentReview.predicted_label;
  const model_version = ml_prediction?.model_version ?? currentReview.model_version;
  const feature_contributions = ml_prediction?.feature_contributions ?? currentReview.feature_contributions;

  const displayReviewText = normalizeReviewText(review_text);
  const wordCount = displayReviewText ? displayReviewText.split(/\s+/).filter(Boolean).length : 0;
  const charCount = displayReviewText.length;

  const formattedDate = timestamp
    ? new Date(timestamp).toLocaleString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : null;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6 space-y-8 max-w-7xl mx-auto">
      {/* ── Search / Selector Toolbar ── */}
      <section className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-4">
        <form onSubmit={handleSubmit} className="flex gap-3 flex-wrap">
          <div className="relative flex-1 min-w-[220px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={inputId}
              onChange={(e) => setInputId(e.target.value)}
              placeholder="Enter a review ID from search results"
              className="w-full bg-gray-800 border border-gray-700 text-white placeholder-gray-500 rounded-xl py-2.5 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !inputId.trim()}
            className="flex items-center gap-1.5 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-sm font-semibold transition"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            Inspect Review
          </button>
        </form>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-gray-500">Try a current review:</span>
          {REVIEW_RECOMMENDATIONS.map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => handleSelectSample(id)}
              className="px-2.5 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-full text-xs text-gray-300 font-mono transition"
            >
              {id}
            </button>
          ))}
        </div>

      </section>

      {loading && <Skeleton />}

      {!loading && error && (
        <div className="bg-red-900/20 border border-red-700/40 rounded-2xl p-8 text-center space-y-3">
          <AlertTriangle className="w-10 h-10 text-red-400 mx-auto" />
          <p className="text-xl text-gray-200 font-semibold">Review Not Found</p>
          <p className="text-red-300 text-sm">{error}</p>
          <p className="text-gray-500 text-xs">Open a review from search results or enter its current review ID.</p>
        </div>
      )}

      {!loading && !error && review && (
        <>
          {/* ── Header ── */}
          <header className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-indigo-900/40 border border-indigo-700">
                  <FileText className="w-5 h-5 text-indigo-400" />
                </div>
                <h1 className="text-2xl font-bold text-gray-100">Review Details</h1>
              </div>
              <p className="text-gray-500 font-mono text-sm ml-11">{review_id}</p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <LabelBadge label={dataset_label} />
              <PredictedBadge cls={predicted_class} />
            </div>
          </header>

      {/* ── Main 2-column grid ─────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── LEFT COLUMN ─────────────────────────────────────────────── */}
        <div className="space-y-6">

          {/* Review Text Card */}
          <section className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-gray-200 flex items-center gap-2">
                <FileText className="w-4 h-4 text-indigo-400" /> Review Text
              </h2>
              <StarRating rating={rating} />
            </div>

            {review_title && (
              <p className="text-gray-200 font-medium">{review_title}</p>
            )}

            <blockquote className="relative border-l-4 border-indigo-600 bg-gray-800/60 rounded-r-lg px-4 py-3 text-gray-300 text-sm leading-relaxed italic whitespace-pre-line">
              {displayReviewText || <span className="text-gray-600">No review text available.</span>}
            </blockquote>

            <div className="flex flex-wrap gap-4 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <Hash className="w-3.5 h-3.5" /> {wordCount} words
              </span>
              <span className="flex items-center gap-1">
                <Hash className="w-3.5 h-3.5" /> {charCount} chars
              </span>
              {formattedDate && (
                <span className="flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5" /> {formattedDate}
                </span>
              )}
            </div>
          </section>

          {/* Provenance Card */}
          <section className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
            <h2 className="font-semibold text-gray-200 flex items-center gap-2">
              <Database className="w-4 h-4 text-emerald-400" /> Provenance
            </h2>

            {is_synthetic && (
              <div className="flex items-start gap-2 bg-orange-950 border border-orange-700 rounded-lg px-3 py-2 text-orange-300 text-xs">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>SYNTHETIC DATA — NOT FROM A REAL DATASET</span>
              </div>
            )}

            <InfoRow label="Dataset Label" value={<LabelBadge label={dataset_label} />} />
            <InfoRow label="Source Dataset" value={source_dataset} />
            <InfoRow label="Review ID" value={review_id} mono />
            <InfoRow label="Reviewer ID" value={reviewer_id} mono />
            <InfoRow label="Product ID" value={product_id} mono />
          </section>
        </div>

        {/* ── RIGHT COLUMN ─────────────────────────────────────────────── */}
        <div className="space-y-6">

          {/* ML Prediction Card */}
          <section className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
            <h2 className="font-semibold text-gray-200 flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-red-400" /> ML Prediction
            </h2>

            {score != null ? (
              <ScoreBar score={score} />
            ) : (
              <p className="text-xs text-gray-500">No model score available.</p>
            )}

            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Predicted class:</span>
              <PredictedBadge cls={predicted_class} />
            </div>

            {/* Disclaimer */}
            <div className="flex items-start gap-2 bg-yellow-950/40 border border-yellow-800/60 rounded-lg px-3 py-2 text-yellow-300/80 text-xs">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>
                <strong>Important:</strong> This is a model score, not a verified probability of fraud.
                Treat it as a risk signal to guide further investigation.
              </span>
            </div>

            {model_version && (
              <p className="text-xs text-gray-600">Model version: {model_version}</p>
            )}

            {/* Feature Contributions mini bar chart */}
            {feature_contributions && Object.keys(feature_contributions).length > 0 && (
              <div className="space-y-1.5">
                <p className="text-xs text-gray-400 font-medium">Feature Contributions</p>
                {Object.entries(feature_contributions)
                  .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                  .slice(0, 6)
                  .map(([feat, val]) => {
                    const absVal = Math.abs(Number(val));
                    const maxVal = 1;
                    const pct = Math.min(100, (absVal / maxVal) * 100);
                    const positive = Number(val) >= 0;
                    return (
                      <div key={feat} className="space-y-0.5">
                        <div className="flex justify-between text-xs text-gray-500">
                          <span className="truncate max-w-[60%]">{feat}</span>
                          <span className={positive ? 'text-red-400' : 'text-emerald-400'}>
                            {positive ? '+' : ''}{Number(val).toFixed(3)}
                          </span>
                        </div>
                        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${positive ? 'bg-red-500' : 'bg-emerald-500'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            )}
          </section>

          {/* Evidence Signals Card */}
          <section className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
            <h2 className="font-semibold text-gray-200 flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-amber-400" /> Evidence Signals
            </h2>

            {evidence_signals && evidence_signals.length > 0 ? (
              <ul className="space-y-2">
                {evidence_signals.map((sig, i) => (
                  <li
                    key={sig.signal_id || i}
                    className="flex items-start gap-3 bg-gray-800 rounded-lg px-3 py-2.5"
                  >
                    <span className="mt-0.5 px-1.5 py-0.5 rounded text-xs font-bold bg-red-900 text-red-300 shrink-0">
                      {sig.signal_type || '?'}
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm text-gray-200 font-mono">
                        {sig.value != null ? String(sig.value) : ''}
                      </p>
                      {sig.description && (
                        <p className="text-xs text-gray-500 mt-0.5">{sig.description}</p>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="flex items-center gap-2 text-gray-500 text-sm">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                No evidence signals detected for this review.
              </div>
            )}
          </section>
        </div>
      </div>

      {/* ── Related Reviews ────────────────────────────────────────────── */}
      {related.length > 0 && (
        <section className="space-y-4">
          <h2 className="font-semibold text-lg text-gray-200 flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-400" /> Related Reviews
            <span className="text-sm text-gray-500 font-normal">({related.length})</span>
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {related.map((rel, i) => {
              const relId = rel.review_id || rel.id;
              const relScore = rel.similarity_score ?? rel.score;
              const relFraud = rel.ml_prediction?.score ?? rel.fraud_score;
              return (
                <div
                  key={relId || i}
                  className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3 hover:border-gray-700 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-mono text-gray-400 truncate">{relId}</p>
                    {relScore != null && (
                      <span className="text-xs bg-indigo-900 text-indigo-300 border border-indigo-700 rounded-full px-2 py-0.5 shrink-0 ml-2">
                        {Number(relScore).toFixed(2)} sim
                      </span>
                    )}
                  </div>

                  {rel.review_text && (
                    <p className="text-sm text-gray-400 line-clamp-3 leading-relaxed whitespace-pre-line">
                      {normalizeReviewText(rel.review_text)}
                    </p>
                  )}

                  <div className="flex items-center justify-between">
                    <StarRating rating={rel.rating} />
                    {relFraud != null && (
                      <span className="text-xs text-gray-500">
                        score: <span className="text-red-400 font-mono">{Number(relFraud).toFixed(3)}</span>
                      </span>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={() => handleSelectSample(relId)}
                    className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                  >
                    View Details <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ── Graph Paths ────────────────────────────────────────────────── */}
      {graph_paths && graph_paths.length > 0 && (
        <section className="space-y-4">
          <h2 className="font-semibold text-lg text-gray-200 flex items-center gap-2">
            <FileText className="w-5 h-5 text-amber-400" /> Reasoning Paths
          </h2>

          <div className="space-y-3">
            {graph_paths.map((path, pi) => (
              <div
                key={pi}
                className="bg-gray-900 border border-gray-800 rounded-xl p-4"
              >
                <p className="text-xs text-gray-500 mb-3">Path {pi + 1}</p>
                <div className="flex flex-wrap items-center gap-1">
                  {(path.nodes || []).map((node, ni) => (
                    <span key={ni} className="flex items-center gap-1">
                      <span className="px-2 py-1 bg-gray-800 rounded text-xs font-mono text-gray-300">
                        {node.id || node}
                      </span>
                      {ni < (path.nodes || []).length - 1 && (
                        <span className="text-gray-600 text-xs">
                          {path.rels?.[ni] ? `—[${path.rels[ni]}]→` : '→'}
                        </span>
                      )}
                    </span>
                  ))}
                </div>
                {path.explanation && (
                  <p className="text-xs text-gray-500 mt-2 italic">{path.explanation}</p>
                )}
              </div>
            ))}
          </div>
        </section>
      )}
        </>
      )}
    </div>
  );
}
