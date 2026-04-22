import { z } from 'zod'

const envSchema = z.object({
  DATABASE_URL: z.string().url().optional().default('postgresql://localhost:5432/imw_msk_demo'),
  JWT_SECRET: z.string().min(1).optional().default('dev-secret-change-in-production'),
  AUTH_SECRET: z.string().min(32).optional(),
  ALLOWED_ORIGINS: z.string().optional(),
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  SESSION_TIMEOUT_MINUTES: z.coerce.number().min(5).max(60).default(15),
  RATE_LIMIT_WINDOW_MS: z.coerce.number().default(60000),
  RATE_LIMIT_MAX_REQUESTS: z.coerce.number().default(100),
  LOG_AUDIT: z.coerce.boolean().default(true),
  DEMO_MODE: z.coerce.boolean().default(true),
})

function validateEnv() {
  const result = envSchema.safeParse(process.env)
  
  if (!result.success) {
    const errors = result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`).join(', ')
    console.warn(`Environment validation warning: ${errors} — running in DEMO_MODE`)
  }
  
  return result.data ?? envSchema.parse({})
}

export const env = validateEnv()
export { envSchema }
