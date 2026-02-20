// Rate Limiting Middleware
// Medical-Grade Implementation for API Protection

import { createMiddleware } from 'hono/factory'
import type { Bindings } from '../types'

// In-memory rate limit store (for single worker)
// In production, use Redis or Cloudflare KV
interface RateLimitRecord {
  count: number
  resetTime: number
  firstRequest: number
}

const rateLimits = new Map<string, RateLimitRecord>()

// Cleanup old entries periodically
setInterval(() => {
  const now = Date.now()
  for (const [key, record] of rateLimits.entries()) {
    if (now > record.resetTime) {
      rateLimits.delete(key)
    }
  }
}, 60000) // Clean every minute

// Rate limit configuration
interface RateLimitConfig {
  windowMs: number      // Time window in milliseconds
  maxRequests: number   // Maximum requests per window
  keyGenerator?: (c: any) => string // Custom key generator
  skipSuccessfulRequests?: boolean
  skipFailedRequests?: boolean
  message?: string
  statusCode?: number
}

// Default configs for different endpoints
export const rateLimitConfigs: Record<string, RateLimitConfig> = {
  // Strict limits for auth endpoints
  auth: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 5,           // 5 attempts
    message: 'Too many login attempts. Please try again later.'
  },
  
  // Standard API limits
  api: {
    windowMs: 60 * 1000,     // 1 minute
    maxRequests: 100,         // 100 requests per minute
    message: 'Too many requests. Please slow down.'
  },
  
  // Stricter for data creation
  write: {
    windowMs: 60 * 1000,     // 1 minute
    maxRequests: 30,         // 30 writes per minute
    message: 'Too many write requests.'
  },
  
  // Very strict for analysis endpoints (computationally expensive)
  analysis: {
    windowMs: 60 * 1000,    // 1 minute
    maxRequests: 10,        // 10 analyses per minute
    message: 'Analysis rate limit exceeded. Please wait before trying again.'
  },
  
  // Upload limits
  upload: {
    windowMs: 60 * 60 * 1000, // 1 hour
    maxRequests: 10,          // 10 uploads per hour
    message: 'Upload limit exceeded. Please try again later.'
  }
}

// Create rate limiter middleware
export function createRateLimiter(config: RateLimitConfig) {
  return createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
    // Generate key - use IP + clinician ID if authenticated
    const clinician = c.get('clinician')
    const ip = c.req.header('CF-Connecting-IP') || 
               c.req.header('X-Forwarded-For') || 
               'unknown'
    
    const key = config.keyGenerator 
      ? config.keyGenerator(c)
      : clinician?.id 
        ? `clinician:${clinician.id}`
        : `ip:${ip}`
    
    const now = Date.now()
    let record = rateLimits.get(key)
    
    // Initialize or reset if window expired
    if (!record || now > record.resetTime) {
      record = {
        count: 0,
        resetTime: now + config.windowMs,
        firstRequest: now
      }
    }
    
    // Increment counter
    record.count++
    rateLimits.set(key, record)
    
    // Set response headers
    const remaining = Math.max(0, config.maxRequests - record.count)
    const resetTime = Math.ceil((record.resetTime - now) / 1000)
    
    c.res.headers.set('X-RateLimit-Limit', config.maxRequests.toString())
    c.res.headers.set('X-RateLimit-Remaining', remaining.toString())
    c.res.headers.set('X-RateLimit-Reset', resetTime.toString())
    
    // Check if limit exceeded
    if (record.count > config.maxRequests) {
      return c.json({
        success: false,
        error: config.message || 'Rate limit exceeded',
        code: 'RATE_LIMITED',
        retryAfter: resetTime
      }, config.statusCode || 429)
    }
    
    await next()
  })
}

// Pre-configured rate limiters
export const authRateLimit = createRateLimiter(rateLimitConfigs.auth)
export const apiRateLimit = createRateLimiter(rateLimitConfigs.api)
export const writeRateLimit = createRateLimiter(rateLimitConfigs.write)
export const analysisRateLimit = createRateLimiter(rateLimitConfigs.analysis)

// Custom rate limiter for specific use cases
export function rateLimit(windowMs: number, maxRequests: number, message?: string) {
  return createRateLimiter({
    windowMs,
    maxRequests,
    message
  })
}

// IP-based rate limiter (no auth required)
export const ipRateLimit = createRateLimiter({
  windowMs: 60 * 1000,
  maxRequests: 50,
  keyGenerator: (c) => {
    return `ip:${c.req.header('CF-Connecting-IP') || c.req.header('X-Forwarded-For') || 'unknown'}`
  }
})

// Sliding window rate limiter (more accurate)
// This is a simplified version - production should use Redis
interface SlidingWindowRecord {
  timestamps: number[]
}

const slidingWindows = new Map<string, SlidingWindowRecord>()

export function slidingRateLimit(windowMs: number, maxRequests: number) {
  return createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
    const ip = c.req.header('CF-Connecting-IP') || 
               c.req.header('X-Forwarded-For') || 
               'unknown'
    const key = `sliding:${ip}`
    
    const now = Date.now()
    const windowStart = now - windowMs
    
    let record = slidingWindows.get(key)
    
    // Clean old timestamps
    if (record) {
      record.timestamps = record.timestamps.filter(ts => ts > windowStart)
    } else {
      record = { timestamps: [] }
    }
    
    // Check limit
    if (record.timestamps.length >= maxRequests) {
      const oldestInWindow = record.timestamps[0]
      const retryAfter = Math.ceil((oldestInWindow + windowMs - now) / 1000)
      
      c.res.headers.set('X-RateLimit-Limit', maxRequests.toString())
      c.res.headers.set('X-RateLimit-Remaining', '0')
      c.res.headers.set('X-RateLimit-Reset', retryAfter.toString())
      
      return c.json({
        success: false,
        error: 'Rate limit exceeded',
        code: 'RATE_LIMITED',
        retryAfter
      }, 429)
    }
    
    // Add current request
    record.timestamps.push(now)
    slidingWindows.set(key, record)
    
    // Set headers
    const remaining = maxRequests - record.timestamps.length
    c.res.headers.set('X-RateLimit-Limit', maxRequests.toString())
    c.res.headers.set('X-RateLimit-Remaining', remaining.toString())
    c.res.headers.set('X-RateLimit-Reset', Math.ceil(windowMs / 1000).toString())
    
    await next()
  })
}
