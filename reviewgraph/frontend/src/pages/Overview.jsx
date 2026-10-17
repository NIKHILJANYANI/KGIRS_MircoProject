import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import {
  Database, Users, Star, Shield, AlertTriangle, CheckCircle,
  Activity, Network, TrendingUp, BookOpen,
} from 'lucide-react';
import { getHealth, getStats } from '../services/api';

// ─── helpers ────────────────────────────────────────────────────────────────

const fmt = (n) => (n === undefined || n === null ? '—' : Number(n).toLocaleString());

const SCORE_COLORS = {
  suspicious: '#ef4444',
  genuine: '#22c55e',
  unlabeled: '#6b7280',
};

// ─── sub-components ─────────────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, color, trend }) {
  const colorMap = {
    blue: 'from-blue-600/20 to-blue-900/10 border-blue-700/40 text-blue-400',
    yellow: 'from-yellow-600/20 to-yellow-900/10 border-yellow-700/40 text-yellow-400',
    green: 'from-green-600/20 to-green-900/10 border-green-700/40 text-green-400',
    purple: 'from-purple-600/20 to-purple-900/10 border-purple-700/40 text-purple-400',
  };
  return (
    <div className={`bg-gradient-to-br ${colorMap[color]} border rounded-2xl p-6 flex flex-col gap-3`}>
      <div className="flex items-center justify-between">
        <Icon className={`w-7 h-7 ${colorMap[color].split(' ').find(c => c.startsWith('text-'))}`} />
        <TrendingUp className="w-4 h-4 text-gray-500" />
      </div>
      <p className="text-4xl font-extrabold text-white tracking-tight">{value}</p>
      <p className="text-sm text-gray-400 font-medium">{label}</p>
    </div>
  );
}

function StatusDot({ ok }) {
  return (
    <span className={`inline-block w-2.5 h-2.5 rounded-full mr-2 ${ok ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`} />
  );
}

function ServiceCard({ title, ok, detail }) {
  return (
    <div className="bg-gray-800/60 border border-gray-700/50 rounded-xl p-4 flex-1 min-w-[160px]">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{title}</p>
      <p className={`text-sm font-semibold flex items-center ${ok ? 'text-green-400' : 'text-red-400'}`}>
        <StatusDot ok={ok} />{ok ? 'Online' : 'Unavailable'}
      </p>
      {detail && <p className="text-xs text-gray-400 mt-1">{detail}</p>}
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="bg-gray-800/60 border border-gray-700/50 rounded-2xl p-6 animate-pulse">
      <div className="h-7 w-7 bg-gray-700 rounded-full mb-4" />
      <div className="h-10 w-24 bg-gray-700 rounded mb-2" />
      <div className="h-4 w-32 bg-gray-700 rounded" />
    </div>
  );
}

const PIPELINE_STEPS = [
  { icon: Database,   label: 'Dataset',          desc: 'Amazon reviews corpus'        },
  { icon: Activity,   label: 'Preprocessing',    desc: 'Cleaning & feature extraction' },
  { icon: Shield,     label: 'ML Classifier',    desc: 'Fake-review detection model'   },
  { icon: Network,    label: 'Knowledge Graph',  desc: 'Neo4j entity relationships'    },
  { icon: BookOpen,   label: 'Retrieval Engine', desc: 'Multi-modal IR pipeline'       },
  { icon: CheckCircle,label: 'Explanation Engine', desc: 'Path-based reasoning'       },
  { icon: TrendingUp, label: 'Dashboard',        desc: 'Interactive analytics UI'      },
];

const OBJECTIVES = [
  'Detect fake reviews using graph-structural and ML signals',
  'Build an explainable knowledge graph for product reviews',
  'Implement multi-modal retrieval (keyword, semantic, graph)',
  'Generate human-readable, evidence-based explanations',
  'Visualise reviewer networks and suspiciousness scores',
  'Evaluate retrieval quality with academic IR metrics (MRR, MAP, NDCG)',
];

const TECH_STACK = {
  Backend:    ['FastAPI (Python)', 'Pydantic', 'Uvicorn'],
  'Graph DB': ['Neo4j', 'Cypher Query Language'],
  ML:         ['Scikit-learn / XGBoost', 'Sentence-Transformers (SBERT)', 'FAISS vector index'],
  Retrieval:  ['BM25 (keyword)', 'Cosine similarity (semantic)', 'Graph traversal (Cypher)'],
  Frontend:   ['React 18', 'Tailwind CSS', 'Recharts', 'Lucide icons'],
};

// ─── CustomTooltip ───────────────────────────────────────────────────────────

function PieTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white shadow-xl">
      <p>{payload[0].name}: <span className="font-bold">{payload[0].value.toLocaleString()}</span></p>
    </div>
  );
}

function BarTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white shadow-xl">
      <p className="text-gray-400 mb-1">Score range: {label}</p>
      <p>Count: <span className="font-bold">{payload[0].value.toLocaleString()}</span></p>
    </div>
  );
}

// ─── main component ──────────────────────────────────────────────────────────

export default function Overview() {
  const [health, setHealth]   = useState(null);
  const [stats,  setStats]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, s] = await Promise.all([getHealth(), getStats()]);
      setHealth(h);
      setStats(s);
    } catch (err) {
      setError(err?.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  // ── derive chart data from stats ──
  const labelDist = stats
    ? [
        { name: 'Suspicious', value: stats.label_distribution?.suspicious ?? 0 },
        { name: 'Genuine',    value: stats.label_distribution?.genuine    ?? 0 },
        { name: 'Unlabeled',  value: stats.label_distribution?.unlabeled  ?? 0 },
      ]
    : [];

  const scoreDist = stats?.model_score_distribution
    ? Object.entries(stats.model_score_distribution).map(([range, count]) => ({
        range,
        count,
      }))
    : [];

  const isHealthy = health?.status === 'healthy' || health?.status === 'ok';

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 space-y-12">

        {/* ── 1. HERO ── */}
        <section className="text-center space-y-4">
          <div className="inline-flex items-center gap-2 bg-indigo-900/40 border border-indigo-700/50 rounded-full px-4 py-1.5 text-xs text-indigo-300 font-semibold tracking-wide uppercase mb-2">
            <BookOpen className="w-3.5 h-3.5" />
            KGIRS Academic Project
          </div>

          <h1 className="text-5xl sm:text-6xl font-extrabold tracking-tight">
            <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              ReviewGraph
            </span>
          </h1>

          <p className="text-lg text-gray-400 max-w-2xl mx-auto leading-relaxed">
            Explainable Graph-Based Information Retrieval for Fake Review Detection
          </p>

          {/* system status */}
          <div className="flex items-center justify-center gap-2 mt-2">
            {loading ? (
              <span className="text-sm text-gray-500">Checking system status…</span>
            ) : error ? (
              <span className="flex items-center gap-1.5 text-sm text-red-400">
                <AlertTriangle className="w-4 h-4" /> System unreachable
              </span>
            ) : (
              <span className={`flex items-center gap-1.5 text-sm font-medium ${isHealthy ? 'text-green-400' : 'text-red-400'}`}>
                {isHealthy
                  ? <><CheckCircle className="w-4 h-4" /> All systems operational</>
                  : <><AlertTriangle className="w-4 h-4" /> Degraded — check service status</>
                }
              </span>
            )}
          </div>
        </section>

        {/* ── 2. STATS GRID ── */}
        <section>
          <h2 className="text-xl font-bold text-gray-200 mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-400" /> Dataset Statistics
          </h2>
          {loading ? (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[...Array(4)].map((_, i) => <SkeletonCard key={i} />)}
            </div>
          ) : error ? (
            <div className="bg-red-900/20 border border-red-700/40 rounded-2xl p-8 text-center space-y-3">
              <AlertTriangle className="w-10 h-10 text-red-400 mx-auto" />
              <p className="text-red-300 font-medium">{error}</p>
              <button
                onClick={fetchData}
                className="mt-2 px-5 py-2 bg-red-700 hover:bg-red-600 rounded-lg text-sm font-semibold transition"
              >
                Retry
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard icon={Database} label="Total Products"   value={fmt(stats?.total_products ?? stats?.graph?.node_counts?.products)}  color="blue"   />
              <StatCard icon={Star}     label="Total Reviews"    value={fmt(stats?.total_reviews ?? stats?.graph?.node_counts?.reviews)}   color="yellow" />
              <StatCard icon={Users}    label="Total Reviewers"  value={fmt(stats?.total_reviewers ?? stats?.graph?.node_counts?.reviewers)} color="green"  />
              <StatCard icon={Shield}   label="Labeled Reviews"  value={fmt(stats?.labeled_reviews ?? stats?.total_reviews ?? 100)} color="purple" />
            </div>
          )}
        </section>

        {/* ── 3. SERVICE STATUS ── */}
        {!loading && !error && (
          <section>
            <h2 className="text-xl font-bold text-gray-200 mb-4 flex items-center gap-2">
              <Network className="w-5 h-5 text-purple-400" /> Service Status
            </h2>
            <div className="flex flex-wrap gap-4">
              <ServiceCard
                title="Neo4j Knowledge Graph"
                ok={health?.neo4j === 'ok' || health?.services?.neo4j === 'connected' || stats?.graph?.status === 'ok'}
                detail={health?.neo4j === 'ok' || stats?.graph?.status === 'ok' ? 'Connected (Aura DB)' : 'Disconnected'}
              />
              <ServiceCard
                title="ML Classifier"
                ok={health?.model === 'ok' || health?.services?.model === 'loaded' || stats?.model?.status === 'ok'}
                detail={stats?.model_version ? `v${stats.model_version}` : 'Loaded'}
              />
              <ServiceCard
                title="Embedding Index"
                ok={health?.embeddings === 'ok' || health?.services?.faiss === 'loaded' || stats?.embeddings?.status === 'ok'}
                detail={stats?.total_reviews ? `${fmt(stats.total_reviews)} vectors` : 'all-MiniLM-L6-v2'}
              />
              <ServiceCard
                title="Dataset"
                ok={health?.dataset === 'ok' || !!stats?.total_reviews}
                detail={stats?.dataset_size ? `${fmt(stats.dataset_size)} records` : '100 records'}
              />
            </div>
          </section>
        )}

        {/* ── 4. CHARTS ── */}
        {!loading && !error && (labelDist.length > 0 || scoreDist.length > 0) && (
          <section>
            <h2 className="text-xl font-bold text-gray-200 mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-green-400" /> Label & Score Distribution
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              {/* Pie chart */}
              <div className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-6">
                <p className="text-sm font-semibold text-gray-300 mb-4">Review Label Distribution</p>
                {labelDist.every(d => d.value === 0) ? (
                  <p className="text-gray-500 text-sm text-center py-10">No label data available</p>
                ) : (
                  <ResponsiveContainer width="100%" height={240}>
                    <PieChart>
                      <Pie
                        data={labelDist}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={95}
                        paddingAngle={3}
                        dataKey="value"
                      >
                        {labelDist.map((entry, i) => (
                          <Cell
                            key={i}
                            fill={SCORE_COLORS[entry.name.toLowerCase()] ?? '#6b7280'}
                          />
                        ))}
                      </Pie>
                      <Tooltip content={<PieTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                )}
                <div className="flex justify-center gap-5 mt-2">
                  {labelDist.map((d) => (
                    <span key={d.name} className="flex items-center gap-1.5 text-xs text-gray-400">
                      <span
                        className="inline-block w-3 h-3 rounded-full"
                        style={{ background: SCORE_COLORS[d.name.toLowerCase()] ?? '#6b7280' }}
                      />
                      {d.name}
                    </span>
                  ))}
                </div>
              </div>

              {/* Bar chart */}
              <div className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-6">
                <p className="text-sm font-semibold text-gray-300 mb-4">Model Score Distribution (histogram)</p>
                {scoreDist.length === 0 ? (
                  <p className="text-gray-500 text-sm text-center py-10">No score distribution data available</p>
                ) : (
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={scoreDist} barCategoryGap="20%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="range" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                      <Tooltip content={<BarTooltip />} />
                      <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>

            </div>
          </section>
        )}

        {/* ── 5. ARCHITECTURE PIPELINE ── */}
        <section>
          <h2 className="text-xl font-bold text-gray-200 mb-6 flex items-center gap-2">
            <Network className="w-5 h-5 text-indigo-400" /> System Architecture
          </h2>
          <div className="flex flex-wrap items-start justify-center gap-2">
            {PIPELINE_STEPS.map((step, i) => {
              const Icon = step.icon;
              return (
                <div key={step.label} className="flex items-center gap-2">
                  <div className="flex flex-col items-center bg-gray-800/70 border border-gray-700/60 rounded-xl px-4 py-3 w-[130px] text-center hover:border-indigo-600/60 transition group">
                    <div className="w-9 h-9 rounded-full bg-indigo-900/50 flex items-center justify-center mb-2 group-hover:bg-indigo-700/50 transition">
                      <Icon className="w-5 h-5 text-indigo-400" />
                    </div>
                    <p className="text-xs font-semibold text-gray-200">{step.label}</p>
                    <p className="text-[10px] text-gray-500 mt-0.5 leading-snug">{step.desc}</p>
                  </div>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <span className="text-gray-600 text-xl font-light select-none">→</span>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* ── 6. ACADEMIC INFO ── */}
        <section>
          <h2 className="text-xl font-bold text-gray-200 mb-6 flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-yellow-400" /> Academic Details
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Objectives */}
            <div className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-6">
              <p className="text-base font-bold text-gray-100 mb-4">Project Objectives</p>
              <ul className="space-y-2">
                {OBJECTIVES.map((obj, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-gray-400">
                    <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                    {obj}
                  </li>
                ))}
              </ul>
            </div>

            {/* Tech stack */}
            <div className="bg-gray-900/60 border border-gray-700/50 rounded-2xl p-6">
              <p className="text-base font-bold text-gray-100 mb-4">Technology Stack</p>
              <div className="space-y-3">
                {Object.entries(TECH_STACK).map(([group, items]) => (
                  <div key={group}>
                    <p className="text-xs uppercase tracking-widest text-indigo-400 font-semibold mb-1">{group}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {items.map((item) => (
                        <span
                          key={item}
                          className="bg-gray-800 border border-gray-700 rounded-md px-2 py-0.5 text-xs text-gray-300"
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </section>

      </div>
    </div>
  );
}
