// HIPAA Compliance Middleware
// Audit Logging & PHI Protection

import { createMiddleware } from 'hono/factory'
import type { Bindings, Variables } from '../types.js'
import { getPool } from '../database.js'

// Audit Action Types
export type AuditAction =
  | 'LOGIN' | 'LOGOUT' | 'LOGIN_FAILED' | 'PASSWORD_CHANGED' | 'REGISTER' | 'REGISTER_FAILED'
  | 'PROFILE_UPDATE'
  | 'PATIENT_VIEW' | 'PATIENT_CREATE' | 'PATIENT_UPDATE' | 'PATIENT_DELETE'
  | 'ASSESSMENT_VIEW' | 'ASSESSMENT_CREATE' | 'ASSESSMENT_UPDATE' | 'ASSESSMENT_COMPLETE'
  | 'TEST_VIEW' | 'TEST_CREATE' | 'TEST_ANALYZE'
  | 'PRESCRIPTION_VIEW' | 'PRESCRIPTION_CREATE' | 'PRESCRIPTION_UPDATE'
  | 'SESSION_VIEW' | 'SESSION_CREATE'
  | 'BILLING_VIEW' | 'BILLING_CREATE'
  | 'VIDEO_VIEW' | 'VIDEO_UPLOAD' | 'VIDEO_DELETE'
  | 'REPORT_GENERATE' | 'EXPORT_DATA'
  | 'SETTINGS_CHANGE' | 'USER_CREATE' | 'USER_UPDATE' | 'USER_DELETE'

// Resource Types for Audit
export type ResourceType =
  | 'auth'
  | 'patient'
  | 'assessment'
  | 'movement_test'
  | 'exercise'
  | 'prescription'
  | 'session'
  | 'billing'
  | 'video'
  | 'clinician'
  | 'report'

// Sensitive fields that should be redacted
const SENSITIVE_FIELDS = [
  'first_name', 'last_name', 'email', 'phone', 'date_of_birth',
  'ssn', 'social_security', 'address', 'insurance', 'policy_number',
  'medical_history', 'medications', 'allergies', 'diagnosis',
  'prescription', 'notes', 'pain_location', 'emergency_contact',
  'password', 'password_hash', 'token'
]

// Redact sensitive data for logging
function redactSensitiveData(data: any): any {
  if (!data) return {}
  if (typeof data === 'string') {
    // Check if string contains sensitive keywords
    for (const field of SENSITIVE_FIELDS) {
      if (data.toLowerCase().includes(field)) {
        return '[REDACTED]'
      }
    }
    return data
  }

  if (Array.isArray(data)) {
    return data.map(item => redactSensitiveData(item))
  }

  if (typeof data === 'object') {
    const redacted: any = {}
    for (const [key, value] of Object.entries(data)) {
      const lowerKey = key.toLowerCase()
      if (SENSITIVE_FIELDS.some(f => lowerKey.includes(f))) {
        redacted[key] = '[REDACTED]'
      } else {
        redacted[key] = redactSensitiveData(value)
      }
    }
    return redacted
  }

  return data
}

// HIPAA-compliant logger - no PHI in console logs
export const hipaaLogger = {
  info: (message: string, meta?: Record<string, any>) => {
    const safeMeta = meta ? redactSensitiveData(meta) : undefined
    // Only log in development, use audit database in production
    if (process.env.NODE_ENV === 'development') {
      console.log(`[INFO] ${new Date().toISOString()} - ${message}`, safeMeta || '')
    }
  },

  warn: (message: string, meta?: Record<string, any>) => {
    const safeMeta = meta ? redactSensitiveData(meta) : undefined
    if (process.env.NODE_ENV === 'development') {
      console.warn(`[WARN] ${new Date().toISOString()} - ${message}`, safeMeta || '')
    }
  },

  error: (message: string, error?: Error, meta?: Record<string, any>) => {
    const safeMeta = meta ? redactSensitiveData(meta) : undefined
    // Error logging without PHI
    const errorInfo = {
      message: error?.message || message,
      // Never log stack traces with potential PHI
      code: (error as any)?.code,
      ...safeMeta
    }
    console.error(`[ERROR] ${new Date().toISOString()} - ${message}`, errorInfo)
  },

  // Simple safe log for non-sensitive info
  safeLog: (message: string) => {
    console.log(`[LOG] ${new Date().toISOString()} - ${message}`)
  },

  // Audit logging - goes to database, not console
  audit: async (action: AuditAction, resourceType: ResourceType, details: Record<string, any>) => {
    const pool = getPool()
    if (!pool) return

    try {
      // Always redact for audit logs
      const safeDetails = redactSensitiveData(details)
      
      await pool.query(
        `INSERT INTO audit_logs (
          user_id, user_email, action, resource_type, resource_id,
          ip_address, user_agent, http_method, http_path, 
          request_data, success, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())`,
        [
          safeDetails.userId || null,
          safeDetails.userEmail || null,
          action,
          resourceType,
          safeDetails.resourceId || null,
          safeDetails.ipAddress || null,
          safeDetails.userAgent || null,
          safeDetails.httpMethod || null,
          safeDetails.httpPath || null,
          JSON.stringify(safeDetails),
          safeDetails.success !== false
        ]
      )
    } catch (e) {
      // Silent fail - don't block operations due to audit logging failures
      // But do log to console that audit logging failed (without PHI)
      console.error('[AUDIT] Failed to write audit log:', { action, resourceType })
    }
  }
}

// Audit Log Middleware - Records all PHI access to database
export const auditLog = (action: AuditAction, resourceType: ResourceType) =>
  createMiddleware<{ Bindings: Bindings, Variables: Variables }>(async (c, next) => {
    const startTime = Date.now()
    const clinician = c.get('clinician')

    // Get client info
    const ipAddress = c.req.header('X-Forwarded-For') ||
      c.req.header('CF-Connecting-IP') ||
      c.req.header('X-Real-IP') ||
      'unknown'
    const userAgent = c.req.header('User-Agent') || 'unknown'

    // Get resource ID from params if available
    const resourceId = c.req.param('id') || 
                       c.req.param('patientId') || 
                       c.req.param('patient_id') ||
                       null

    await next()

    const duration = Date.now() - startTime
    const status = c.res.status
    const success = status >= 200 && status < 400

    // Log to database (async, don't await to avoid blocking)
    hipaaLogger.audit(action, resourceType, {
      userId: clinician?.id,
      userEmail: clinician?.email,
      resourceId,
      ipAddress,
      userAgent,
      httpMethod: c.req.method,
      httpPath: c.req.path,
      status,
      duration,
      success
    }).catch(() => {
      // Silent fail
    })
  })

// PHI Access Audit - Extra logging for sensitive operations
export const phiAudit = (action: 'VIEW' | 'CREATE' | 'UPDATE' | 'DELETE', resourceType: ResourceType) =>
  createMiddleware<{ Bindings: Bindings, Variables: Variables }>(async (c, next) => {
    const clinician = c.get('clinician')

    // Get request body (redacted)
    let requestData: any = {}
    try {
      const body = await c.req.raw.clone().json()
      requestData = redactSensitiveData(body)
    } catch (e) {
      // Not JSON or empty
    }

    // Get patient ID if related
    const patientId = c.req.param('patientId') ||
      c.req.param('patient_id') ||
      requestData.patient_id ||
      null

    // Log to database
    hipaaLogger.audit(
      `PHI_${action}` as AuditAction,
      resourceType,
      {
        userId: clinician?.id,
        userEmail: clinician?.email,
        patientId,
        action,
        resourceType,
        ipAddress: c.req.header('X-Forwarded-For') || 'unknown',
        httpPath: c.req.path,
        httpMethod: c.req.method,
        timestamp: new Date().toISOString()
      }
    ).catch(() => {})

    await next()
  })

// Session Timeout Middleware (HIPAA requirement)
const SESSION_TIMEOUT_MS = 15 * 60 * 1000 // 15 minutes

export const sessionTimeout = createMiddleware<{ Bindings: Bindings, Variables: Variables }>(async (c, next) => {
  const pool = getPool()
  const clinician = c.get('clinician')

  if (clinician?.id && pool) {
    // Check last activity
    const result = await pool.query(
      'SELECT last_activity_at FROM users WHERE id = $1',
      [clinician.id]
    )

    if (result.rows.length > 0) {
      const lastActivity = result.rows[0].last_activity_at
      if (lastActivity) {
        const lastActiveTime = new Date(lastActivity).getTime()
        const now = Date.now()

        if (now - lastActiveTime > SESSION_TIMEOUT_MS) {
          // Log timeout
          hipaaLogger.audit('LOGOUT', 'auth', {
            userId: clinician.id,
            userEmail: clinician.email,
            reason: 'session_timeout',
            success: true
          }).catch(() => {})

          return c.json({
            success: false,
            error: 'Session expired due to inactivity',
            code: 'SESSION_TIMEOUT'
          }, 401)
        }
      }

      // Update last activity
      await pool.query(
        'UPDATE users SET last_activity_at = NOW() WHERE id = $1',
        [clinician.id]
      ).catch(() => {})
    }
  }

  await next()
})

// Request Size Limit (prevent DoS)
export const requestSizeLimit = (maxSize: number = 50 * 1024 * 1024) =>
  createMiddleware<{ Bindings: Bindings, Variables: Variables }>(async (c, next) => {
    const contentLength = c.req.header('Content-Length')

    if (contentLength && parseInt(contentLength) > maxSize) {
      return c.json({
        success: false,
        error: 'Request too large',
        code: 'PAYLOAD_TOO_LARGE'
      }, 413)
    }

    await next()
  })

// Security Headers Middleware
export const securityHeaders = createMiddleware<{ Bindings: Bindings, Variables: Variables }>(async (c, next) => {
  await next()

  c.res.headers.set('X-Content-Type-Options', 'nosniff')
  c.res.headers.set('X-Frame-Options', 'DENY')
  c.res.headers.set('X-XSS-Protection', '1; mode=block')
  c.res.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin')
  c.res.headers.set('Permissions-Policy', 'camera=(self), microphone=(self), geolocation=()')

  // HSTS (only over HTTPS)
  if (c.req.url.startsWith('https://')) {
    c.res.headers.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
  }
})

// Auto-logout warning middleware
export const autoLogoutWarning = createMiddleware<{ Bindings: Bindings, Variables: Variables }>(async (c, next) => {
  await next()

  const pool = getPool()
  const clinician = c.get('clinician')

  if (clinician?.id && pool) {
    const result = await pool.query(
      'SELECT last_activity_at FROM users WHERE id = $1',
      [clinician.id]
    )

    if (result.rows.length > 0) {
      const lastActivity = result.rows[0].last_activity_at
      if (lastActivity) {
        const lastActiveTime = new Date(lastActivity).getTime()
        const now = Date.now()
        const timeRemaining = SESSION_TIMEOUT_MS - (now - lastActiveTime)

        // Add warning header if session expires in less than 60 seconds
        if (timeRemaining < 60000 && timeRemaining > 0) {
          c.res.headers.set('X-Session-Warning', 'true')
          c.res.headers.set('X-Session-Time-Remaining', Math.floor(timeRemaining / 1000).toString())
        }
      }
    }
  }
})

// Export audit actions for use in routes
export { SENSITIVE_FIELDS, redactSensitiveData }

// Convenience export for safe logging
export const safeLog = hipaaLogger.safeLog
