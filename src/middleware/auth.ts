// Security Middleware - Authentication & Authorization
// Medical-Grade Implementation

import { createMiddleware } from 'hono/factory'
import type { Bindings } from '../types'

// JWT Configuration
const JWT_CONFIG = {
  algorithm: 'HS256',
  expiresIn: '24h',
  refreshExpiresIn: '7d'
}

// Password hashing (using Web Crypto API - available in Cloudflare Workers)
export async function hashPassword(password: string, salt: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(password + salt)
  const hashBuffer = await crypto.subtle.digest('SHA-384', data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

export async function verifyPassword(password: string, salt: string, hash: string): Promise<boolean> {
  const computedHash = await hashPassword(password, salt)
  return computedHash === hash
}

// Generate JWT Token
export async function generateToken(payload: { 
  id: number; 
  email: string; 
  role: string;
  clinic_id?: number 
}, secret: string): Promise<string> {
  const header = btoa(JSON.stringify({ alg: 'HS384', typ: 'JWT' }))
  const payloadB64 = btoa(JSON.stringify({
    ...payload,
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + 86400 // 24 hours
  }))
  
  const encoder = new TextEncoder()
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-384' },
    false,
    ['sign']
  )
  
  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    encoder.encode(`${header}.${payloadB64}`)
  )
  
  const signatureB64 = btoa(String.fromCharCode(...new Uint8Array(signature)))
  return `${header}.${payloadB64}.${signatureB64}`
}

// Verify JWT Token
export async function verifyToken(token: string, secret: string): Promise<{
  id: number;
  email: string;
  role: string;
  clinic_id?: number;
} | null> {
  try {
    const parts = token.split('.')
    if (parts.length !== 3) return null
    
    const [header, payloadB64, signature] = parts
    const encoder = new TextEncoder()
    
    const key = await crypto.subtle.importKey(
      'raw',
      encoder.encode(secret),
      { name: 'HMAC', hash: 'SHA-384' },
      false,
      ['verify']
    )
    
    const signatureValid = await crypto.subtle.verify(
      'HMAC',
      key,
      encoder.encode(`${header}.${payloadB64}`),
      btoa(signature)
    )
    
    if (!signatureValid) return null
    
    const payload = JSON.parse(atob(payloadB64))
    
    // Check expiration
    if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) {
      return null
    }
    
    return {
      id: payload.id,
      email: payload.email,
      role: payload.role,
      clinic_id: payload.clinic_id
    }
  } catch (e) {
    return null
  }
}

// Auth Middleware - Protects routes
export const authMiddleware = createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
  const token = c.req.header('Authorization')?.replace('Bearer ', '')
  
  if (!token) {
    return c.json({ 
      success: false, 
      error: 'Authentication required',
      code: 'NO_TOKEN'
    }, 401)
  }
  
  const secret = c.env.JWT_SECRET || c.env.AUTH_SECRET
  if (!secret) {
    console.error('JWT_SECRET not configured')
    return c.json({ 
      success: false, 
      error: 'Server configuration error',
      code: 'CONFIG_ERROR'
    }, 500)
  }
  
  const payload = await verifyToken(token, secret)
  
  if (!payload) {
    return c.json({ 
      success: false, 
      error: 'Invalid or expired token',
      code: 'INVALID_TOKEN'
    }, 401)
  }
  
  // Attach user to context
  c.set('clinician', payload)
  c.set('clinicianId', payload.id)
  
  await next()
})

// Role-based Access Control
export const requireRole = (...allowedRoles: string[]) => 
  createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
    const clinician = c.get('clinician')
    
    if (!clinician) {
      return c.json({ 
        success: false, 
        error: 'Authentication required',
        code: 'NO_AUTH'
      }, 401)
    }
    
    if (!allowedRoles.includes(clinician.role)) {
      return c.json({ 
        success: false, 
        error: 'Insufficient permissions',
        code: 'FORBIDDEN'
      }, 403)
    }
    
    await next()
  })

// Session Activity Update Middleware
export const sessionActivity = createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
  const clinician = c.get('clinician')
  
  if (clinician?.id) {
    try {
      await c.env.DB.prepare(`
        UPDATE clinicians 
        SET last_activity = CURRENT_TIMESTAMP,
            last_login = COALESCE(last_login, CURRENT_TIMESTAMP)
        WHERE id = ?
      `).bind(clinician.id).run()
    } catch (e) {
      // Non-critical - don't block request
    }
  }
  
  await next()
})

// Combined auth middleware with activity tracking
export const secureAuth = createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
  // First check auth
  const token = c.req.header('Authorization')?.replace('Bearer ', '')
  
  if (!token) {
    return c.json({ 
      success: false, 
      error: 'Authentication required',
      code: 'NO_TOKEN'
    }, 401)
  }
  
  const secret = c.env.JWT_SECRET || c.env.AUTH_SECRET
  if (!secret) {
    console.error('JWT_SECRET not configured')
    return c.json({ 
      success: false, 
      error: 'Server configuration error',
      code: 'CONFIG_ERROR'
    }, 500)
  }
  
  const payload = await verifyToken(token, secret)
  
  if (!payload) {
    return c.json({ 
      success: false, 
      error: 'Invalid or expired token',
      code: 'INVALID_TOKEN'
    }, 401)
  }
  
  c.set('clinician', payload)
  c.set('clinicianId', payload.id)
  
  // Update last activity
  if (payload.id) {
    c.env.DB.prepare(`
      UPDATE clinicians SET last_activity = CURRENT_TIMESTAMP WHERE id = ?
    `).bind(payload.id).run().catch(() => {})
  }
  
  await next()
})
