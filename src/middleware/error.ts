// Centralized error handling for API
import type { MiddlewareHandler } from 'hono'
import { hipaaLogger } from './hipaa.js'

// Simple wrapper to catch errors in route handlers
export const errorHandler: MiddlewareHandler = async (c, next) => {
  try {
    return await next()
  } catch (err: any) {
    hipaaLogger.error('Unhandled server error', err as Error, {
      path: c.req?.path,
      method: c.req?.method
    })
    return c.json({ success: false, error: 'Internal server error' }, 500)
  }
}
