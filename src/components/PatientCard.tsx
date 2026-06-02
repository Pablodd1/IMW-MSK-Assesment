/**
 * PatientCard - Reusable patient summary card
 * Shows: name, DOB, last visit, condition, assessments count
 */

import type { FC } from 'hono/jsx'

export interface PatientCardProps {
  id: string
  firstName: string
  lastName: string
  dateOfBirth?: string
  gender?: string
  condition?: string
  lastVisit?: string
  assessmentCount: number
  painLevel?: number
  status?: 'active' | 'inactive' | 'completed'
}

function calculateAge(dob?: string): string {
  if (!dob) return 'N/A'
  const diff = Date.now() - new Date(dob).getTime()
  return String(Math.floor(diff / (365.25 * 24 * 60 * 60 * 1000)))
}

function formatDate(d?: string): string {
  if (!d) return 'N/A'
  return new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function getInitials(first: string, last: string): string {
  return `${first?.[0] || ''}${last?.[0] || ''}`.toUpperCase()
}

const statusStyles: Record<string, string> = {
  active:    'bg-green-900/40 text-green-400 border-green-500/30',
  inactive:  'bg-gray-700/40 text-gray-400 border-gray-500/30',
  completed: 'bg-blue-900/40 text-blue-400 border-blue-500/30',
}

export const PatientCard: FC<PatientCardProps> = ({
  id, firstName, lastName, dateOfBirth, gender, condition,
  lastVisit, assessmentCount, painLevel, status = 'active'
}) => {
  const age = calculateAge(dateOfBirth)
  const initials = getInitials(firstName, lastName)

  return (
    <div class="bg-gray-800/70 border border-gray-700/50 rounded-xl p-4 hover:border-amber-500/40 hover:shadow-lg hover:shadow-amber-500/5 transition-all duration-200 cursor-pointer group">
      <div class="flex items-start gap-4">
        {/* Avatar */}
        <div class="w-12 h-12 rounded-full bg-amber-500/20 border border-amber-500/30 flex items-center justify-center flex-shrink-0">
          <span class="text-amber-400 font-semibold text-sm">{initials}</span>
        </div>

        {/* Info */}
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 mb-1">
            <h3 class="text-white font-semibold truncate group-hover:text-amber-300 transition-colors">
              {firstName} {lastName}
            </h3>
            <span class={`text-xs px-2 py-0.5 rounded-full border ${statusStyles[status]}`}>
              {status}
            </span>
          </div>

          <div class="flex flex-wrap gap-x-3 gap-y-1 text-xs text-gray-400">
            <span>{age} yrs</span>
            {gender && <span>• {gender}</span>}
            {condition && <span class="truncate max-w-[200px]">• {condition}</span>}
          </div>

          {/* Metrics row */}
          <div class="flex items-center gap-4 mt-3 text-xs">
            <div class="flex items-center gap-1 text-gray-500">
              <i class="fas fa-calendar-alt text-gray-600"></i>
              <span title={lastVisit ? formatDate(lastVisit) : ''}>
                {lastVisit ? formatDate(lastVisit) : 'No visits'}
              </span>
            </div>
            <div class="flex items-center gap-1 text-gray-500">
              <i class="fas fa-clipboard-list text-gray-600"></i>
              <span>{assessmentCount} assessments</span>
            </div>
            {painLevel !== undefined && (
              <div class="flex items-center gap-1 text-gray-500">
                <i class="fas fa-heart text-rose-500/60"></i>
                <span>Pain: {painLevel}/10</span>
              </div>
            )}
          </div>
        </div>

        {/* Quick actions */}
        <div class="flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <a href={`/patients/${id}`} class="text-xs px-3 py-1.5 bg-amber-500/20 text-amber-400 rounded-lg hover:bg-amber-500/30 flex items-center gap-1" title="View Details">
            <i class="fas fa-eye text-[10px]"></i>
          </a>
          <a href={`/assessment?patient=${id}`} class="text-xs px-3 py-1.5 bg-teal-500/20 text-teal-400 rounded-lg hover:bg-teal-500/30 flex items-center gap-1" title="New Assessment">
            <i class="fas fa-plus text-[10px]"></i>
          </a>
        </div>
      </div>
    </div>
  )
}
