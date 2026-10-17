import { useEffect, useState } from 'react';
import { getEvaluation } from '../services/api';
import {
  FlaskConical,
  Download,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  BarChart2,
  Clock,
  ShieldCheck,
  BookOpen,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

// ─── Helpers ─────────────────────────────────────────────────────────────────
function pct(val) {
  if (val == null) return '—';
  const n = Number(val);
  return isNaN(n) ? '—' : `${(n * 100).toFixed(1)}%`;
}

function fmt(val, decimals = 4) {
  if (val == null) return '—';
  const n = Number(val);
  return isNaN(n) ? String(val) : n.toFixed(decimals);
}

function MetricCard({ label, value, color = 'text-indigo-400', sub }) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 space-y-1">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className={`text-3xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-xs text-gray-600">{sub}</p>}
    </div>
  );
}

function SectionHeader({ icon: Icon, title, color = 'text-indigo-400' }) {
  return (
    <div className="flex items-center gap-3">
      <div className="p-2 rounded-lg bg-gray-800 border border-gray-700">
        <Icon className={`w-5 h-5 ${color}`} />
      </div>
      <h2 className="text-lg font-semibold text-gray-100">{title}</h2>
    </div>
  );
}

// Confusion matrix cell
function CMCell({ label, value, bg, text }) {
  return (
    <div className={`flex flex-col items-center justify-center rounded-xl p-4 ${bg}`}>
      <p className={`text-2xl font-bold ${text}`}>{value ?? '—'}</p>
      <p className={`text-xs font-medium ${text} opacity-80`}>{label}</p>
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function Evaluation() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await getEvaluation();
        if (!cancelled) setData(result);
      } catch (err) {
        if (!cancelled)
          setError(err?.response?.data?.detail || err.message || 'Failed to load evaluation data');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // ── Download report ──────────────────────────────────────────────────────
  const downloadReport = () => {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'evaluation_report.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Guards ───────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <Loader2 className="w-10 h-10 text-indigo-400 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
        <div className="text-center space-y-3 max-w-md">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto" />
          <p className="text-xl text-gray-200 font-semibold">Error Loading Evaluation</p>
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // ── Not-run state ─────────────────────────────────────────────────────────
  if (!data || data.status === 'not_run' || !data.classification) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 p-8 flex items-center justify-center">
        <div className="max-w-lg w-full space-y-6">
          <div className="text-center space-y-2">
            <FlaskConical className="w-16 h-16 text-gray-600 mx-auto" />
            <h1 className="text-2xl font-bold text-gray-200">Evaluation Not Run Yet</h1>
            <p className="text-gray-500">
              Run the evaluation script to generate metrics and populate this dashboard.
            </p>
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
            <p className="text-sm text-gray-400">Run in your project root:</p>
            <pre className="bg-gray-950 border border-gray-700 rounded-lg p-4 text-emerald-400 font-mono text-sm overflow-x-auto">
              python scripts/evaluate_system.py
            </pre>
            <p className="text-xs text-gray-600">
              Results will be persisted to the database and shown here automatically.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ── Extract data ──────────────────────────────────────────────────────────
  const cls = data.classification || {};
  const retrieval = data.retrieval || {};
  const explainability = data.explainability || {};
  const latency = data.latency || {};
  const discussion = data.discussion || {};

  // Models: prefer LR + RF keys or fall back to first two keys
  const modelKeys = Object.keys(cls.models || {});
  const lrKey = modelKeys.find((k) => k.toLowerCase().includes('logistic')) || modelKeys[0];
  const rfKey = modelKeys.find((k) => k.toLowerCase().includes('random') || k.toLowerCase().includes('forest')) || modelKeys[1];
  const models = cls.models || {};

  // Retrieval rows
  const retrievalMethods = ['keyword', 'semantic', 'graph', 'hybrid'];
  const retrievalData = retrievalMethods.map((m) => ({
    method: m.charAt(0).toUpperCase() + m.slice(1),
    'Recall@5': Number(retrieval[m]?.recall_at_5 ?? 0),
    'Recall@10': Number(retrieval[m]?.recall_at_10 ?? 0),
    MRR: Number(retrieval[m]?.mrr ?? 0),
    'nDCG@10': Number(retrieval[m]?.ndcg_at_10 ?? 0),
    'Latency (ms)': Number(retrieval[m]?.avg_latency_ms ?? 0),
  }));

  // Best values per column
  const retrievalCols = ['Recall@5', 'Recall@10', 'MRR', 'nDCG@10', 'Latency (ms)'];
  const bestVals = {};
  retrievalCols.forEach((col) => {
    const vals = retrievalData.map((r) => r[col]);
    bestVals[col] = col === 'Latency (ms)' ? Math.min(...vals) : Math.max(...vals);
  });

  // Latency chart data
  const latencyChartData = Object.entries(latency).map(([endpoint, ms]) => ({
    endpoint: endpoint.replace(/_/g, ' '),
    'Avg Latency (ms)': Number(ms),
  }));

  // Classification report rows
  const report = cls.classification_report || {};
  const reportRows = ['Genuine', 'Suspicious', 'macro avg', 'weighted avg'].filter(
    (k) => report[lrKey]?.[k] || report[rfKey]?.[k]
  );

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6 space-y-10 max-w-7xl mx-auto">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-indigo-900/40 border border-indigo-700">
              <FlaskConical className="w-5 h-5 text-indigo-400" />
            </div>
            <h1 className="text-2xl font-bold text-gray-100">Evaluation &amp; Research Metrics</h1>
          </div>
          <p className="text-gray-500 text-sm ml-11">Results generated from actual experiments</p>
        </div>

        <button
          onClick={downloadReport}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-700 hover:bg-indigo-600 text-sm font-medium transition-colors"
        >
          <Download className="w-4 h-4" /> Download Report
        </button>
      </header>

      {/* ── Classification Metrics ───────────────────────────────────────── */}
      <section className="space-y-6">
        <SectionHeader icon={BarChart2} title="Classification Metrics" color="text-indigo-400" />

        {[lrKey, rfKey].filter(Boolean).map((modelKey) => {
          const m = models[modelKey] || {};
          return (
            <div key={modelKey} className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-5">
              <h3 className="font-semibold text-gray-200 flex items-center gap-2">
                <span className="px-2 py-0.5 bg-indigo-900 text-indigo-300 rounded text-xs">{modelKey}</span>
              </h3>

              {/* Metric cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Precision" value={fmt(m.precision, 3)} color="text-indigo-400" />
                <MetricCard label="Recall" value={fmt(m.recall, 3)} color="text-emerald-400" />
                <MetricCard label="F1-Score" value={fmt(m.f1_score, 3)} color="text-amber-400" />
                <MetricCard label="PR-AUC" value={fmt(m.pr_auc, 3)} color="text-rose-400" />
              </div>

              {/* Confusion Matrix */}
              {m.confusion_matrix && (
                <div className="space-y-2">
                  <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Confusion Matrix</p>
                  <div className="grid grid-cols-2 gap-2 max-w-xs">
                    <CMCell
                      label="True Positive"
                      value={m.confusion_matrix.tp}
                      bg="bg-emerald-950 border border-emerald-800"
                      text="text-emerald-300"
                    />
                    <CMCell
                      label="False Positive"
                      value={m.confusion_matrix.fp}
                      bg="bg-red-950 border border-red-800"
                      text="text-red-300"
                    />
                    <CMCell
                      label="False Negative"
                      value={m.confusion_matrix.fn}
                      bg="bg-orange-950 border border-orange-800"
                      text="text-orange-300"
                    />
                    <CMCell
                      label="True Negative"
                      value={m.confusion_matrix.tn}
                      bg="bg-gray-800 border border-gray-700"
                      text="text-gray-300"
                    />
                  </div>
                </div>
              )}

              {/* Classification Report Table */}
              {report[modelKey] && reportRows.length > 0 && (
                <div className="space-y-2 overflow-x-auto">
                  <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Classification Report</p>
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-800">
                        <th className="text-left py-1.5 pr-4 text-gray-500 font-medium">Class</th>
                        {['precision', 'recall', 'f1-score', 'support'].map((col) => (
                          <th key={col} className="text-right py-1.5 px-2 text-gray-500 font-medium capitalize">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {reportRows.map((row) => {
                        const r = report[modelKey][row] || {};
                        return (
                          <tr key={row} className="border-b border-gray-800/50 hover:bg-gray-800/40">
                            <td className="py-1.5 pr-4 text-gray-300 font-medium">{row}</td>
                            {['precision', 'recall', 'f1-score', 'support'].map((col) => (
                              <td key={col} className="text-right py-1.5 px-2 font-mono text-gray-400">
                                {r[col] != null
                                  ? col === 'support'
                                    ? Math.round(r[col])
                                    : Number(r[col]).toFixed(3)
                                  : '—'}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          );
        })}
      </section>

      {/* ── Retrieval Comparison ─────────────────────────────────────────── */}
      <section className="space-y-6">
        <SectionHeader icon={BarChart2} title="Retrieval Method Comparison" color="text-amber-400" />

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-5">
          {/* Table */}
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-2 pr-4 text-gray-500 font-medium">Method</th>
                  {retrievalCols.map((col) => (
                    <th key={col} className="text-right py-2 px-2 text-gray-500 font-medium">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {retrievalData.map((row) => (
                  <tr key={row.method} className="border-b border-gray-800/50 hover:bg-gray-800/40">
                    <td className="py-2 pr-4 text-gray-200 font-medium">{row.method}</td>
                    {retrievalCols.map((col) => {
                      const val = row[col];
                      const isBest = val === bestVals[col];
                      return (
                        <td
                          key={col}
                          className={`text-right py-2 px-2 font-mono ${
                            isBest ? 'text-indigo-300 font-bold' : 'text-gray-400'
                          }`}
                        >
                          {col === 'Latency (ms)' ? val.toFixed(1) : val.toFixed(3)}
                          {isBest && <span className="ml-1 text-indigo-500 text-xs">★</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Bar Chart */}
          <div>
            <p className="text-xs text-gray-500 font-medium uppercase tracking-wide mb-3">
              Visual Comparison (Recall@5 · Recall@10 · MRR · nDCG@10)
            </p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={retrievalData} margin={{ top: 0, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="method" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                  labelStyle={{ color: '#f3f4f6' }}
                  itemStyle={{ color: '#d1d5db' }}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: '#9ca3af' }} />
                <Bar dataKey="Recall@5" fill="#6366f1" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Recall@10" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="MRR" fill="#10b981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="nDCG@10" fill="#f59e0b" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* ── Explainability Metrics ───────────────────────────────────────── */}
      <section className="space-y-4">
        <SectionHeader icon={ShieldCheck} title="Explainability Metrics" color="text-emerald-400" />

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="Explanation Validity"
            value={pct(explainability.explanation_validity)}
            color="text-emerald-400"
            sub="Edges exist in graph"
          />
          <MetricCard
            label="Source Coverage"
            value={pct(explainability.source_coverage)}
            color="text-indigo-400"
            sub="Sources found in DB"
          />
          <MetricCard
            label="Path Accuracy"
            value={pct(explainability.path_accuracy)}
            color="text-amber-400"
            sub="Paths verified correct"
          />
          <MetricCard
            label="Hallucination Rate"
            value={pct(explainability.hallucination_rate ?? 0)}
            color="text-emerald-400"
            sub="Only retrieved evidence"
          />
        </div>
      </section>

      {/* ── Latency Metrics ─────────────────────────────────────────────── */}
      {latencyChartData.length > 0 && (
        <section className="space-y-4">
          <SectionHeader icon={Clock} title="Latency per Endpoint" color="text-sky-400" />

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={latencyChartData}
                layout="vertical"
                margin={{ top: 0, right: 30, left: 80, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  label={{ value: 'ms', position: 'insideRight', offset: 10, fill: '#6b7280', fontSize: 11 }}
                />
                <YAxis
                  type="category"
                  dataKey="endpoint"
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  width={78}
                />
                <Tooltip
                  contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                  labelStyle={{ color: '#f3f4f6' }}
                  itemStyle={{ color: '#d1d5db' }}
                  formatter={(v) => [`${Number(v).toFixed(1)} ms`, 'Avg Latency']}
                />
                <Bar dataKey="Avg Latency (ms)" fill="#38bdf8" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* ── Research Discussion ──────────────────────────────────────────── */}
      {(discussion.findings || discussion.limitations) && (
        <section className="space-y-4">
          <SectionHeader icon={BookOpen} title="Research Discussion" color="text-violet-400" />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Findings */}
            {discussion.findings && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <h3 className="font-semibold text-gray-200">Findings &amp; Observations</h3>
                </div>
                <p className="text-sm text-gray-400 leading-relaxed">{discussion.findings}</p>
              </div>
            )}

            {/* Limitations */}
            {discussion.limitations && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  <h3 className="font-semibold text-gray-200">Limitations of the Proposed Method</h3>
                </div>
                {Array.isArray(discussion.limitations) ? (
                  <ul className="space-y-2">
                    {discussion.limitations.map((lim, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-400">
                        <span className="text-amber-500 mt-1 shrink-0">•</span>
                        {lim}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-400 leading-relaxed">{discussion.limitations}</p>
                )}
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
