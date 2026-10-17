import React, { useEffect, useState } from 'react'
import {
  BarChart3,
  Search,
  Shield,
  Network,
  FileText,
  FlaskConical,
  Activity,
  AlertCircle,
  GitBranch,
} from 'lucide-react'
import {
  BrowserRouter as Router,
  Routes,
  Route,
  NavLink,
} from 'react-router-dom'
import { getHealth } from './services/api'

// ── Page imports ──────────────────────────────────────────────────────────────
import Overview      from './pages/Overview'
import SearchPage    from './pages/Search'
import Investigation from './pages/Investigation'
import GraphView     from './pages/GraphView'
import ReviewDetails from './pages/ReviewDetails'
import Evaluation    from './pages/Evaluation'

// ── Nav config ────────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { label: 'Overview',         icon: BarChart3,    path: '/'            },
  { label: 'Search',           icon: Search,       path: '/search'      },
  { label: 'Investigation',    icon: Shield,       path: '/investigation'},
  { label: 'Knowledge Graph',  icon: Network,      path: '/graph'       },
  { label: 'Review Details',   icon: FileText,     path: '/reviews'     },
  { label: 'Evaluation',       icon: FlaskConical, path: '/evaluation'  },
]

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ health }) {
  const isUp = health?.status === 'ok'

  return (
    <aside className="fixed top-0 left-0 h-screen w-60 bg-gray-950 border-r border-gray-800 flex flex-col z-20">
      {/* Logo */}
      <div className="px-5 py-6 border-b border-gray-800">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center flex-shrink-0">
            <GitBranch size={18} className="text-white" />
          </div>
          <span className="text-lg font-bold text-gray-100 leading-none">ReviewGraph</span>
        </div>
        <p className="text-[10px] text-gray-500 pl-11 font-medium tracking-wide uppercase">
          KGIRS Academic Project
        </p>
      </div>

      {/* Nav items */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ label, icon: Icon, path }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-600/30'
                  : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'
              }`
            }
          >
            <Icon size={17} className="flex-shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Health indicator */}
      <div className="px-5 py-4 border-t border-gray-800">
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`w-2 h-2 rounded-full flex-shrink-0 ${
              isUp ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'
            }`}
          />
          <span className={isUp ? 'text-emerald-400' : 'text-red-400'}>
            {isUp ? 'API Online' : 'API Offline'}
          </span>
          {health?.graph_nodes != null && (
            <span className="ml-auto text-gray-500">
              {health.graph_nodes.toLocaleString()} nodes
            </span>
          )}
        </div>
      </div>
    </aside>
  )
}

// ── Top status bar ────────────────────────────────────────────────────────────
function TopBar({ health, healthError }) {
  const isUp = health?.status === 'ok'

  return (
    <header className="fixed top-0 left-60 right-0 h-10 bg-gray-950 border-b border-gray-800 flex items-center px-6 gap-3 z-10">
      <Activity size={13} className={isUp ? 'text-emerald-400' : 'text-red-400'} />
      <span className="text-xs text-gray-500">
        {healthError
          ? `Backend unreachable — ${healthError}`
          : isUp
          ? `ReviewGraph API v${health?.version ?? '—'} · ${health?.graph_nodes?.toLocaleString() ?? '—'} nodes · ${health?.graph_edges?.toLocaleString() ?? '—'} edges`
          : 'Connecting to backend…'}
      </span>
      {healthError && (
        <span className="ml-auto flex items-center gap-1.5 text-xs text-red-400">
          <AlertCircle size={12} />
          Start the FastAPI backend on port 8000
        </span>
      )}
    </header>
  )
}

// ── App root ──────────────────────────────────────────────────────────────────
export default function App() {
  const [health, setHealth]           = useState(null)
  const [healthError, setHealthError] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function fetchHealth() {
      try {
        const data = await getHealth()
        if (!cancelled) {
          setHealth(data)
          setHealthError(null)
        }
      } catch (err) {
        if (!cancelled) setHealthError(err.message)
      }
    }

    fetchHealth()
    const interval = setInterval(fetchHealth, 30_000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return (
    <Router>
      <Sidebar health={health} />
      <TopBar health={health} healthError={healthError} />

      {/* Main content — offset for sidebar + topbar */}
      <main className="ml-60 mt-10 min-h-screen bg-gray-950">
        <div className="p-6">
          <Routes>
            <Route path="/"             element={<Overview />}      />
            <Route path="/search"       element={<SearchPage />}    />
            <Route path="/investigation" element={<Investigation />} />
            <Route path="/graph"        element={<GraphView />}     />
            <Route path="/reviews"      element={<ReviewDetails />} />
            <Route path="/reviews/:id"  element={<ReviewDetails />} />
            <Route path="/review/:id"   element={<ReviewDetails />} />
            <Route path="/evaluation"   element={<Evaluation />}    />
          </Routes>
        </div>
      </main>
    </Router>
  )
}
