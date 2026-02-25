// Centralized error handling for API
import type { Middleware } from 'hono/factory'
import { safeLog } from './hipaa'

// Simple wrapper to catch errors in route handlers
export const errorHandler: Middleware = (next) => async (c) => {
  try {
    return await next()
  } catch (err: any) {
    safeLog.error('Unhandled server error', err as Error, {
      path: c.req?.path,
      method: c.req?.method
    })
    return c.json({ success: false, error: 'Internal server error' }, 500)
  }
}
