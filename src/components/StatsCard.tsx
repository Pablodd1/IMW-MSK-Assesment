/**
 * StatsCard - Metric card with icon, value, trend indicator
 * Used in patient dashboard for KPIs (Total Patients, Active, etc.)
 */

import type { FC } from 'hono/jsx'

export interface StatsCardProps {
  title: string
  value: string | number
  icon: string       // FontAwesome icon class, e.g. 'fas fa-users'
  trend?: {
    value: number    // positive = up, negative = down
    label: string    // e.g. "vs last month"
  }
  color?: 'teal' | 'gold' | 'purple' | 'rose' | 'amber'
}

const colorMap: Record<string, { bg: string; border: string; iconBg: string; iconColor: string }> = {
  teal:    { bg: 'bg-teal-900/30',     border: 'border-teal-500/30',  iconBg: 'bg-teal-500/20',  iconColor: 'text-teal-400' },
  gold:    { bg: 'bg-amber-900/20',     border: 'border-amber-500/30', iconBg: 'bg-amber-500/20', iconColor: 'text-amber-400' },
  purple:  { bg: 'bg-purple-900/30',   border: 'border-purple-500/30',iconBg: 'bg-purple-500/20',iconColor: 'text-purple-400' },
  rose:    { bg: 'bg-rose-900/30',     border: 'border-rose-500/30',  iconBg: 'bg-rose-500/20',  iconColor: 'text-rose-400' },
  amber:   { bg: 'bg-amber-900/20',    border: 'border-amber-500/30', iconBg: 'bg-amber-500/20', iconColor: 'text-amber-400' },
}

export const StatsCard: FC<StatsCardProps> = ({ title, value, icon, trend, color = 'teal' }) => {
  const c = colorMap[color] || colorMap.teal

  return (
    <div class={`rounded-xl p-4 border ${c.border} ${c.bg} backdrop-blur-sm`}>
      <div class="flex items-start justify-between mb-3">
        <p class="text-sm text-gray-400 font-medium">{title}</p>
        <div class={`w-10 h-10 rounded-lg ${c.iconBg} flex items-center justify-center`}>
          <i class={`${icon} ${c.iconColor} text-lg`}></i>
        </div>
      </div>
      <p class="text-3xl font-bold text-white mb-1">{value}</p>
      {trend && (
        <div class="flex items-center gap-1">
          <i class={`fas fa-arrow-${trend.value >= 0 ? 'up' : 'down'} text-xs ${trend.value >= 0 ? 'text-green-400' : 'text-red-400'}`}></i>
          <span class={`text-xs ${trend.value >= 0 ? 'text-green-400' : 'text-red-400'} font-medium`}>
            {Math.abs(trend.value)}%
          </span>
          <span class="text-xs text-gray-500 ml-1">{trend.label}</span>
        </div>
      )}
    </div>
  )
}
