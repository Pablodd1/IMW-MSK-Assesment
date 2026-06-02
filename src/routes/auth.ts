// Authentication Routes - Secure Implementation
// Uses bcrypt for password hashing and proper JWT token generation

import { Hono } from 'hono'
import bcrypt from 'bcryptjs'
import type { Bindings, Variables } from '../types.js'
import { generateToken, hashPassword, verifyPassword } from '../middleware/auth.js'
import { validate, loginSchema, clinicianRegisterSchema } from '../middleware/validation.js'
import { getPool, withTransaction } from '../database.js'
import { auditLog } from '../middleware/hipaa.js'

const auth = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Constants
const MAX_LOGIN_ATTEMPTS = 5
const LOCKOUT_DURATION_MINUTES = 30
const SALT_ROUNDS = 12

// Helper: Get client IP
function getClientIP(c: any): string {
  return c.req.header('X-Forwarded-For') || 
         c.req.header('X-Real-IP') || 
         'unknown'
}

// Helper: Log audit event
async function logAudit(
  c: any, 
  action: string, 
  userId: string | number | null = null, 
  email: string | null = null,
  success: boolean = true,
  details: any = {}
) {
  try {
    const pool = getPool()
    if (!pool) return

    await pool.query(
      `INSERT INTO audit_logs (
        user_id, user_email, action, resource_type, resource_id,
        ip_address, user_agent, http_method, http_path, http_status, success, request_data
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`,
      [
        userId,
        email,
        action,
        'auth',
        userId?.toString() || null,
        getClientIP(c),
        c.req.header('User-Agent') || 'unknown',
        c.req.method,
        c.req.path,
        success ? 200 : 401,
        success,
        JSON.stringify(details)
      ]
    )
  } catch (e) {
    // Silent fail - don't block auth due to logging issues
  }
}

// POST /api/auth/register - Register new user
auth.post('/register', validate(clinicianRegisterSchema), async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ 
      success: false, 
      error: 'Database not available',
      code: 'DB_UNAVAILABLE'
    }, 503)
  }

  try {
    const data = c.get('validatedData')
    
    // Check if email already exists
    const existingUser = await pool.query(
      'SELECT id FROM users WHERE email = $1',
      [data.email.toLowerCase()]
    )
    
    if (existingUser.rows.length > 0) {
      await logAudit(c, 'REGISTER_FAILED', null, data.email, false, { reason: 'email_exists' })
      return c.json({ 
        success: false, 
        error: 'Email already registered',
        code: 'EMAIL_EXISTS'
      }, 409)
    }

    // Hash password
    const passwordHash = await bcrypt.hash(data.password, SALT_ROUNDS)

    // Create user
    const result = await pool.query(
      `INSERT INTO users (
        email, password_hash, first_name, last_name, role, title,
        license_number, license_state, npi_number, phone,
        account_status, email_verified, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW())
      RETURNING id, email, first_name, last_name, role`,
      [
        data.email.toLowerCase(),
        passwordHash,
        data.first_name,
        data.last_name,
        data.role || 'clinician',
        data.title || null,
        data.license_number || null,
        data.license_state || null,
        data.npi_number || null,
        data.phone || null,
        'active',
        false // email not verified yet
      ]
    )

    const user = result.rows[0]

    // Log successful registration
    await logAudit(c, 'REGISTER', user.id, user.email, true, { role: user.role })

    return c.json({
      success: true,
      data: {
        id: user.id,
        email: user.email,
        first_name: user.first_name,
        last_name: user.last_name,
        role: user.role
      },
      message: 'Registration successful. Please check your email to verify your account.'
    })

  } catch (error: any) {
    console.error('[AUTH] Registration error:', error.message)
    return c.json({ 
      success: false, 
      error: 'Registration failed',
      code: 'REGISTRATION_ERROR'
    }, 500)
  }
})

// POST /api/auth/login - Authenticate user and return JWT
auth.post('/login', validate(loginSchema), async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ 
      success: false, 
      error: 'Database not available',
      code: 'DB_UNAVAILABLE'
    }, 503)
  }

  try {
    const { email, password } = c.get('validatedData')
    const clientIP = getClientIP(c)

    // Get user by email
    const userResult = await pool.query(
      `SELECT id, email, password_hash, first_name, last_name, role, 
              clinic_id, title, account_status, failed_login_attempts, 
              locked_until, email_verified, last_login_at
       FROM users WHERE email = $1`,
      [email.toLowerCase()]
    )

    if (userResult.rows.length === 0) {
      await logAudit(c, 'LOGIN_FAILED', null, email, false, { reason: 'user_not_found' })
      return c.json({ 
        success: false, 
        error: 'Invalid email or password',
        code: 'INVALID_CREDENTIALS'
      }, 401)
    }

    const user = userResult.rows[0]

    // Check if account is locked
    if (user.locked_until && new Date(user.locked_until) > new Date()) {
      const remainingMinutes = Math.ceil((new Date(user.locked_until).getTime() - Date.now()) / 60000)
      await logAudit(c, 'LOGIN_FAILED', user.id, email, false, { reason: 'account_locked' })
      return c.json({ 
        success: false, 
        error: `Account locked. Try again in ${remainingMinutes} minutes.`,
        code: 'ACCOUNT_LOCKED'
      }, 403)
    }

    // Check if account is active
    if (user.account_status !== 'active') {
      await logAudit(c, 'LOGIN_FAILED', user.id, email, false, { reason: 'account_inactive', status: user.account_status })
      return c.json({ 
        success: false, 
        error: 'Account is not active. Please contact support.',
        code: 'ACCOUNT_INACTIVE'
      }, 403)
    }

    // Verify password
    const passwordValid = await bcrypt.compare(password, user.password_hash)

    if (!passwordValid) {
      // Increment failed attempts
      const newAttempts = (user.failed_login_attempts || 0) + 1
      const shouldLock = newAttempts >= MAX_LOGIN_ATTEMPTS

      await pool.query(
        `UPDATE users SET 
          failed_login_attempts = $1,
          locked_until = $2,
          updated_at = NOW()
         WHERE id = $3`,
        [
          newAttempts,
          shouldLock ? new Date(Date.now() + LOCKOUT_DURATION_MINUTES * 60000) : null,
          user.id
        ]
      )

      await logAudit(c, 'LOGIN_FAILED', user.id, email, false, { 
        reason: 'invalid_password', 
        attempts: newAttempts,
        locked: shouldLock 
      })

      if (shouldLock) {
        return c.json({ 
          success: false, 
          error: `Too many failed attempts. Account locked for ${LOCKOUT_DURATION_MINUTES} minutes.`,
          code: 'ACCOUNT_LOCKED'
        }, 403)
      }

      return c.json({ 
        success: false, 
        error: 'Invalid email or password',
        code: 'INVALID_CREDENTIALS'
      }, 401)
    }

    // Generate JWT token
    const secret = process.env.JWT_SECRET || process.env.AUTH_SECRET
    if (!secret) {
      console.error('[AUTH] JWT_SECRET not configured')
      return c.json({ 
        success: false, 
        error: 'Server configuration error',
        code: 'CONFIG_ERROR'
      }, 500)
    }

    const token = await generateToken({
      id: user.id,
      email: user.email,
      role: user.role,
      clinic_id: user.clinic_id
    }, secret)

    // Update last login and reset failed attempts
    await pool.query(
      `UPDATE users SET 
        last_login_at = NOW(),
        last_activity_at = NOW(),
        failed_login_attempts = 0,
        locked_until = NULL,
        updated_at = NOW()
       WHERE id = $1`,
      [user.id]
    )

    // Log successful login
    await logAudit(c, 'LOGIN', user.id, email, true, { ip: clientIP })

    return c.json({
      success: true,
      data: {
        id: user.id,
        email: user.email,
        first_name: user.first_name,
        last_name: user.last_name,
        role: user.role,
        title: user.title,
        clinic_id: user.clinic_id,
        email_verified: user.email_verified,
        token
      }
    })

  } catch (error: any) {
    console.error('[AUTH] Login error:', error.message)
    return c.json({ 
      success: false, 
      error: 'Login failed',
      code: 'LOGIN_ERROR'
    }, 500)
  }
})

// POST /api/auth/logout - Logout user (client should clear token)
auth.post('/logout', async (c) => {
  // Get user from context if available (token already validated by middleware)
  const userId = c.get('clinicianId')
  const email = c.get('clinician')?.email

  if (userId) {
    await logAudit(c, 'LOGOUT', userId, email || null, true)
  }

  return c.json({
    success: true,
    message: 'Logged out successfully'
  })
})

// GET /api/auth/profile - Get current user profile
auth.get('/profile', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ 
      success: false, 
      error: 'Database not available',
      code: 'DB_UNAVAILABLE'
    }, 503)
  }

  const userId = c.get('clinicianId')
  if (!userId) {
    return c.json({ 
      success: false, 
      error: 'Not authenticated',
      code: 'NO_AUTH'
    }, 401)
  }

  try {
    const result = await pool.query(
      `SELECT id, email, first_name, last_name, role, title,
              clinic_id, license_number, license_state, npi_number, phone,
              specialty, avatar_url, email_verified, last_login_at,
              created_at
       FROM users WHERE id = $1`,
      [userId]
    )

    if (result.rows.length === 0) {
      return c.json({ 
        success: false, 
        error: 'User not found',
        code: 'USER_NOT_FOUND'
      }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0]
    })

  } catch (error: any) {
    console.error('[AUTH] Profile error:', error.message)
    return c.json({ 
      success: false, 
      error: 'Failed to load profile',
      code: 'PROFILE_ERROR'
    }, 500)
  }
})

// PUT /api/auth/profile - Update user profile
auth.put('/profile', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ 
      success: false, 
      error: 'Database not available' 
    }, 503)
  }

  const userId = c.get('clinicianId')
  if (!userId) {
    return c.json({ 
      success: false, 
      error: 'Not authenticated' 
    }, 401)
  }

  try {
    const updates = await c.req.json()
    
    // Whitelist allowed fields
    const allowedFields = ['first_name', 'last_name', 'phone', 'title', 'specialty', 'avatar_url']
    const setFields: string[] = []
    const values: any[] = []
    
    for (const field of allowedFields) {
      if (updates[field] !== undefined) {
        setFields.push(`${field} = $${values.length + 1}`)
        values.push(updates[field])
      }
    }

    if (setFields.length === 0) {
      return c.json({ 
        success: false, 
        error: 'No valid fields to update' 
      }, 400)
    }

    values.push(userId)
    
    const result = await pool.query(
      `UPDATE users SET ${setFields.join(', ')}, updated_at = NOW()
       WHERE id = $${values.length}
       RETURNING id, email, first_name, last_name, role, title, phone, specialty, avatar_url`,
      values
    )

    await logAudit(c, 'PROFILE_UPDATE', userId, null, true, { fields: allowedFields.filter(f => updates[f] !== undefined) })

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Profile updated successfully'
    })

  } catch (error: any) {
    console.error('[AUTH] Profile update error:', error.message)
    return c.json({ 
      success: false, 
      error: 'Failed to update profile' 
    }, 500)
  }
})

// POST /api/auth/change-password - Change password
auth.post('/change-password', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ 
      success: false, 
      error: 'Database not available' 
    }, 503)
  }

  const userId = c.get('clinicianId')
  if (!userId) {
    return c.json({ 
      success: false, 
      error: 'Not authenticated' 
    }, 401)
  }

  try {
    const { current_password, new_password } = await c.req.json()

    if (!current_password || !new_password) {
      return c.json({ 
        success: false, 
        error: 'Current password and new password required' 
      }, 400)
    }

    if (new_password.length < 8) {
      return c.json({ 
        success: false, 
        error: 'New password must be at least 8 characters' 
      }, 400)
    }

    // Get current password hash
    const userResult = await pool.query(
      'SELECT password_hash FROM users WHERE id = $1',
      [userId]
    )

    if (userResult.rows.length === 0) {
      return c.json({ 
        success: false, 
        error: 'User not found' 
      }, 404)
    }

    // Verify current password
    const currentHash = userResult.rows[0].password_hash
    const valid = await bcrypt.compare(current_password, currentHash)

    if (!valid) {
      await logAudit(c, 'PASSWORD_CHANGE_FAILED', userId, null, false, { reason: 'invalid_current_password' })
      return c.json({ 
        success: false, 
        error: 'Current password is incorrect' 
      }, 401)
    }

    // Hash new password
    const newHash = await bcrypt.hash(new_password, SALT_ROUNDS)

    // Update password
    await pool.query(
      `UPDATE users SET 
        password_hash = $1,
        password_changed_at = NOW(),
        updated_at = NOW()
       WHERE id = $2`,
      [newHash, userId]
    )

    await logAudit(c, 'PASSWORD_CHANGED', userId, null, true)

    return c.json({
      success: true,
      message: 'Password changed successfully'
    })

  } catch (error: any) {
    console.error('[AUTH] Password change error:', error.message)
    return c.json({ 
      success: false, 
      error: 'Failed to change password' 
    }, 500)
  }
})

export default auth
