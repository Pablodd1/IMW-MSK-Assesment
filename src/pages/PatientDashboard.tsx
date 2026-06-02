/**
 * PatientDashboard - Comprehensive patient dashboard page
 * Server-rendered via Hono JSX with Supabase data
 */

import type { FC } from 'hono/jsx'
import { StatsCard } from '../components/StatsCard.js'
import { PatientCard } from '../components/PatientCard.js'

interface DashboardStats {
  total_patients: number
  active_patients: number
  assessments_this_month: number
  avg_fms_score: number
}

interface DashboardPatient {
  id: string
  first_name: string
  last_name: string
  date_of_birth?: string
  gender?: string
  condition?: string
  last_visit?: string
  assessment_count: number
  pain_level?: number
  status: string
}

interface PatientDashboardProps {
  stats: DashboardStats
  patients: DashboardPatient[]
  error?: string
}

function Layout({ children }: { children: any }) {
  return (
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Patient Dashboard - PhysioMotion</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet" />
        <link href="/static/styles.css" rel="stylesheet" />
        <link href="/static/mobile-responsive.css" rel="stylesheet" />
        <style>{`
          body { background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
          .metric-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); }
          .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); }
          .animate-fade-in { animation: fadeIn 0.3s ease-out; }
          @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
          .pulse-dot { animation: pulse 2s ease-in-out infinite; }
          @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        `}</style>
      </head>
      <body class="text-white min-h-screen">
        {/* Top Navigation Bar */}
        <nav class="glass border-b border-gray-700/50 px-6 py-3 flex items-center justify-between sticky top-0 z-50">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center">
              <i class="fas fa-heartbeat text-white text-lg"></i>
            </div>
            <div>
              <span class="font-bold text-lg text-white">PhysioMotion</span>
              <span class="text-xs text-amber-400 ml-2">Medical Dashboard</span>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <a href="/dashboard" class="text-sm text-gray-400 hover:text-white transition-colors"><i class="fas fa-video mr-1"></i>Live Tracking</a>
            <a href="/patient-dashboard" class="text-sm text-amber-400 font-semibold"><i class="fas fa-users mr-1"></i>Patients</a>
            <a href="/exercises" class="text-sm text-gray-400 hover:text-white transition-colors"><i class="fas fa-dumbbell mr-1"></i>Exercises</a>
            <a href="/assessment" class="text-sm text-gray-400 hover:text-white transition-colors"><i class="fas fa-clipboard-list mr-1"></i>Assess</a>
          </div>
        </nav>

        {children}
      </body>
    </html>
  )
}

export const PatientDashboard: FC<PatientDashboardProps> = ({ stats, patients, error }) => {
  return (
    <Layout>
      <div class="max-w-7xl mx-auto px-4 md:px-6 py-6 animate-fade-in">
        {/* Header */}
        <div class="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
          <div>
            <h1 class="text-2xl md:text-3xl font-bold text-white">
              <i class="fas fa-users text-amber-400 mr-3"></i>
              Patient Dashboard
            </h1>
            <p class="text-gray-400 mt-1 text-sm">Manage and monitor your patients</p>
          </div>
          <div class="flex gap-3">
            <a href="/intake" class="px-4 py-2.5 bg-amber-500 hover:bg-amber-600 text-gray-900 font-semibold rounded-lg flex items-center gap-2 transition-colors text-sm">
              <i class="fas fa-plus"></i> New Patient
            </a>
            <button class="px-4 py-2.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg flex items-center gap-2 transition-colors text-sm">
              <i class="fas fa-file-export"></i> Export
            </button>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div class="bg-rose-900/30 border border-rose-500/30 rounded-xl p-4 mb-6 text-rose-300 text-sm flex items-center gap-2">
            <i class="fas fa-exclamation-triangle"></i>
            {error}
          </div>
        )}

        {/* Stats Grid */}
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatsCard
            title="Total Patients"
            value={stats.total_patients}
            icon="fas fa-users"
            color="teal"
            trend={{ value: 12, label: 'vs last month' }}
          />
          <StatsCard
            title="Active Patients"
            value={stats.active_patients}
            icon="fas fa-user-check"
            color="gold"
            trend={{ value: 8, label: 'this month' }}
          />
          <StatsCard
            title="Assessments"
            value={stats.assessments_this_month}
            icon="fas fa-clipboard-check"
            color="purple"
            trend={{ value: 15, label: 'vs last month' }}
          />
          <StatsCard
            title="Avg FMS Score"
            value={stats.avg_fms_score}
            icon="fas fa-chart-line"
            color="rose"
            trend={{ value: -3, label: 'vs last month' }}
          />
        </div>

        {/* Search & Filter Bar */}
        <div class="glass rounded-xl p-3 mb-6 flex flex-col sm:flex-row gap-3 border border-gray-700/50">
          <div class="flex-1 relative">
            <i class="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm"></i>
            <input
              type="text"
              placeholder="Search patients by name, condition..."
              class="w-full pl-10 pr-4 py-2.5 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/30"
              id="patient-search"
            />
          </div>
          <select class="px-4 py-2.5 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-amber-500/50" id="status-filter">
            <option value="">All Status</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="completed">Completed</option>
          </select>
          <select class="px-4 py-2.5 bg-gray-700/50 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:border-amber-500/50" id="sort-by">
            <option value="recent">Recent</option>
            <option value="name">Name A-Z</option>
            <option value="assessments">Most Assessments</option>
          </select>
        </div>

        {/* Patient Cards Grid */}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="patient-grid">
          {patients.length === 0 && !error && (
            <div class="col-span-2 text-center py-16 text-gray-500">
              <i class="fas fa-user-slash text-5xl mb-4 block"></i>
              <p class="text-lg font-medium">No patients found</p>
              <p class="text-sm mt-1">Add your first patient to get started</p>
              <a href="/intake" class="inline-block mt-4 px-6 py-2 bg-amber-500 text-gray-900 font-semibold rounded-lg hover:bg-amber-600 transition-colors">
                Add Patient
              </a>
            </div>
          )}
          {patients.map((p) => (
            <PatientCard
              id={p.id}
              firstName={p.first_name}
              lastName={p.last_name}
              dateOfBirth={p.date_of_birth}
              gender={p.gender}
              condition={p.condition}
              lastVisit={p.last_visit}
              assessmentCount={p.assessment_count}
              painLevel={p.pain_level}
              status={p.status as any}
            />
          ))}
        </div>

        {/* Recent Activity Section */}
        <div class="mt-8">
          <h2 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <i class="fas fa-clock text-amber-400"></i>
            Recent Activity
          </h2>
          <div class="glass rounded-xl border border-gray-700/50 overflow-hidden">
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b border-gray-700/50 text-left">
                    <th class="px-6 py-3 text-gray-400 font-medium">Patient</th>
                    <th class="px-6 py-3 text-gray-400 font-medium">Action</th>
                    <th class="px-6 py-3 text-gray-400 font-medium">Date</th>
                    <th class="px-6 py-3 text-gray-400 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody id="activity-table" class="divide-y divide-gray-700/30">
                  {patients.slice(0, 5).map((p) => (
                    <tr class="hover:bg-gray-700/30 transition-colors">
                      <td class="px-6 py-3 text-white font-medium">{p.first_name} {p.last_name}</td>
                      <td class="px-6 py-3 text-gray-400">
                        {p.assessment_count > 0 ? `${p.assessment_count} assessments` : 'New patient'}
                      </td>
                      <td class="px-6 py-3 text-gray-500">{p.last_visit ? new Date(p.last_visit).toLocaleDateString() : 'N/A'}</td>
                      <td class="px-6 py-3">
                        <span class={`px-2 py-1 rounded-full text-xs font-medium border ${
                          p.status === 'active' ? 'bg-green-900/40 text-green-400 border-green-500/30' :
                          p.status === 'inactive' ? 'bg-gray-700/40 text-gray-400 border-gray-500/30' :
                          'bg-blue-900/40 text-blue-400 border-blue-500/30'
                        }`}>
                          {p.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {patients.length === 0 && (
                    <tr>
                      <td colspan={4} class="px-6 py-8 text-center text-gray-500">No recent activity</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div class="mt-8 pt-6 border-t border-gray-700/30 flex flex-col sm:flex-row justify-between items-center gap-2 text-xs text-gray-500">
          <span>PhysioMotion v2.0 — Medical Movement Assessment Platform</span>
          <span class="flex items-center gap-1">
            <span class="w-2 h-2 rounded-full bg-green-400 pulse-dot"></span>
            Supabase connected
          </span>
        </div>
      </div>

      {/* Client-side JS for search/filter */}
      <script dangerouslySetInnerHTML={{ __html: `
        document.getElementById('patient-search')?.addEventListener('input', function() {
          const query = this.value.toLowerCase();
          document.querySelectorAll('#patient-grid > div').forEach(card => {
            const text = card.textContent?.toLowerCase() || '';
            card.style.display = text.includes(query) ? '' : 'none';
          });
        });
        document.getElementById('status-filter')?.addEventListener('change', function() {
          const status = this.value.toLowerCase();
          document.querySelectorAll('#patient-grid > div').forEach(card => {
            if (!status) { card.style.display = ''; return; }
            const hasStatus = card.querySelector('[class*="rounded-full"]')?.textContent?.toLowerCase().includes(status);
            card.style.display = hasStatus ? '' : 'none';
          });
        });
      `}} />
    </Layout>
  )
}

export default PatientDashboard
