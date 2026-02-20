// HIPAA Compliance Middleware
// Audit Logging & Security

import { createMiddleware } from 'hono/factory'
import type { Bindings } from '../types'

// Audit Action Types
export type AuditAction = 
  | 'LOGIN' | 'LOGOUT' | 'LOGIN_FAILED'
  | 'PATIENT_VIEW' | 'PATIENT_CREATE' | 'PATIENT_UPDATE' | 'PATIENT_DELETE'
  | 'ASSESSMENT_VIEW' | 'ASSESSMENT_CREATE' | 'ASSESSMENT_UPDATE' | 'ASSESSMENT_COMPLETE'
  | 'TEST_VIEW' | 'TEST_CREATE' | 'TEST_ANALYZE'
  | 'PRESCRIPTION_VIEW' | 'PRESCRIPTION_CREATE' | 'PRESCRIPTION_UPDATE'
  | 'SESSION_VIEW' | 'SESSION_CREATE'
  | 'BILLING_VIEW' | 'BILLING_CREATE'
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
  | 'clinician' 
  | 'report'

// Sensitive fields that should be redacted
const SENSITIVE_FIELDS = [
  'first_name', 'last_name', 'email', 'phone', 'date_of_birth',
  'ssn', 'social_security', 'address', 'insurance', 'policy_number',
  'medical_history', 'medications', 'allergies', 'diagnosis',
  'prescription', 'notes', 'pain_location', 'emergency_contact'
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

// Safe logging - prevents PHI in logs
export const safeLog = {
  info: (message: string, meta?: Record<string, any>) => {
    const safeMeta = meta ? redactSensitiveData(meta) : undefined
    console.log(`[INFO] ${new Date().toISOString()} - ${message}`, safeMeta || '')
  },
  
  warn: (message: string, meta?: Record<string, any>) => {
    const safeMeta = meta ? redactSensitiveData(meta) : undefined
    console.warn(`[WARN] ${new Date().toISOString()} - ${message}`, safeMeta || '')
  },
  
  error: (message: string, error?: Error, meta?: Record<string, any>) => {
    const safeMeta = meta ? redactSensitiveData(meta) : undefined
    console.error(`[ERROR] ${new Date().toISOString()} - ${message}`, {
      message: error?.message || message,
      stack: process.env.NODE_ENV === 'development' ? error?.stack : undefined,
      ...safeMeta
    })
  },
  
  audit: (action: AuditAction, resourceType: ResourceType, details: Record<string, any>) => {
    // Always redact for audit logs
    const safeDetails = redactSensitiveData(details)
    console.log(`[AUDIT] ${new Date().toISOString()} - ${action} on ${resourceType}`, safeDetails)
  }
}

// Audit Log Middleware
export const auditLog = (action: AuditAction, resourceType: ResourceType) => 
  createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
    const startTime = Date.now()
    const clinician = c.get('clinician')
    const clinicianId = clinician?.id || null
    
    // Get client info
    const ipAddress = c.req.header('CF-Connecting-IP') || 
                      c.req.header('X-Forwarded-For') || 
                      'unknown'
    const userAgent = c.req.header('User-Agent') || 'unknown'
    
    // Get resource ID from params if available
    const resourceId = c.req.param('id') || null
    
    // Capture response
    const originalResponse = c.res.clone()
    
    await next()
    
    const duration = Date.now() - startTime
    const status = c.res.status
    
    // Determine if action was successful
    const success = status >= 200 && status < 400
    
    // Log to database
    try {
      await c.env.DB.prepare(`
        INSERT INTO audit_logs (
          clinician_id, 
          action, 
          resource_type, 
          resource_id, 
          ip_address, 
          user_agent,
          http_method,
          http_status,
          duration_ms,
          success,
          created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
      `).bind(
        clinicianId,
        action,
        resourceType,
        resourceId,
        ipAddress,
        userAgent,
        c.req.method,
        status,
        duration,
        success ? 1 : 0
      ).run()
    } catch (error) {
      // Log to console but don't fail the request
      console.error('Failed to write audit log:', error)
    }
    
    // Also log to console in development
    if (process.env.NODE_ENV === 'development') {
      safeLog.audit(action, resourceType, {
        clinicianId,
        resourceId,
        ipAddress,
        httpMethod: c.req.method,
        status,
        duration
      })
    }
  })

// PHI Access Audit - Extra logging for sensitive operations
export const phiAudit = (action: 'VIEW' | 'CREATE' | 'UPDATE' | 'DELETE', resourceType: ResourceType) =>
  createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
    const clinician = c.get('clinician')
    
    // Get request body (redacted)
    let requestData: any = {}
    try {
      const body = await c.req.clone().json()
      requestData = redactSensitiveData(body)
    } catch (e) {
      // Not JSON or empty
    }
    
    // Get patient ID if related
    const patientId = c.req.param('patientId') || 
                      c.req.param('patient_id') ||
                      requestData.patient_id ||
                      null
    
    safeLog.audit(
      `PHI_${action}` as AuditAction,
      resourceType,
      {
        clinicianId: clinician?.id,
        clinicianEmail: clinician?.email,
        patientId,
        action,
        resourceType,
        ipAddress: c.req.header('CF-Connecting-IP'),
        timestamp: new Date().toISOString()
      }
    )
    
    await next()
  })

// Session Timeout Middleware (HIPAA requirement)
const SESSION_TIMEOUT_MS = 15 * 60 * 1000 // 15 minutes

export const sessionTimeout = createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
  const clinician = c.get('clinician')
  
  if (clinician?.id) {
    // Check last activity
    const lastActivity = await c.env.DB.prepare(`
      SELECT last_activity FROM clinicians WHERE id = ?
    `).bind(clinician.id).first<{ last_activity: string }>()
    
    if (lastActivity?.last_activity) {
      const lastActiveTime = new Date(lastActivity.last_activity).getTime()
      const now = Date.now()
      
      if (now - lastActiveTime > SESSION_TIMEOUT_MS) {
        return c.json({
          success: false,
          error: 'Session expired due to inactivity',
          code: 'SESSION_TIMEOUT'
        }, 401)
      }
    }
  }
  
  await next()
})

// Request Size Limit (prevent DoS)
export const requestSizeLimit = (maxSize: number = 1024 * 1024) => 
  createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
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
export const securityHeaders = createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
  await next()
  
  c.res.headers.set('X-Content-Type-Options', 'nosniff')
  c.res.headers.set('X-Frame-Options', 'DENY')
  c.res.headers.set('X-XSS-Protection', '1; mode=block')
  c.res.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin')
  c.res.headers.set('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
  
  // HSTS (only over HTTPS)
  if (c.req.url.startsWith('https://')) {
    c.res.headers.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
  }
})

// Create audit logs table migration
export const AUDIT_LOG_MIGRATION = `
CREATE TABLE IF NOT EXISTS audit_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clinician_id INTEGER,
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id TEXT,
  ip_address TEXT,
  user_agent TEXT,
  http_method TEXT,
  http_status INTEGER,
  duration_ms INTEGER,
  success INTEGER DEFAULT 1,
  details TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (clinician_id) REFERENCES clinicians(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_clinician ON audit_logs(clinician_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);
`
