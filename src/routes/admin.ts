// Admin Routes - System Management and User Administration

import { Hono } from 'hono'
import type { Bindings, Variables } from '../types.js'
import { authMiddleware, requireRole } from '../middleware/auth.js'
import { getPool } from '../database.js'
import bcrypt from 'bcryptjs'

const admin = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Apply auth and admin role middleware
admin.use('*', authMiddleware)
admin.use('*', requireRole('admin'))

// GET /api/admin/users - List all users
admin.get('/users', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const url = new URL(c.req.url)
    const page = parseInt(url.searchParams.get('page') || '1')
    const limit = parseInt(url.searchParams.get('limit') || '20')
    const role = url.searchParams.get('role')
    const status = url.searchParams.get('status')
    const offset = (page - 1) * limit

    let whereClause = 'WHERE 1=1'
    const params: any[] = []

    if (role) {
      params.push(role)
      whereClause += ` AND u.role = $${params.length}`
    }

    if (status) {
      params.push(status)
      whereClause += ` AND u.account_status = $${params.length}`
    }

    // Get total count
    const countResult = await pool.query(
      `SELECT COUNT(*) as total FROM users u ${whereClause}`,
      params
    )
    const total = parseInt(countResult.rows[0].total)

    // Get users
    params.push(limit, offset)
    const result = await pool.query(
      `SELECT u.id, u.email, u.first_name, u.last_name, u.role, u.title,
              u.account_status, u.email_verified, u.last_login_at, u.created_at,
              c.name as clinic_name
       FROM users u
       LEFT JOIN clinics c ON u.clinic_id = c.id
       ${whereClause}
       ORDER BY u.created_at DESC
       LIMIT $${params.length - 1} OFFSET $${params.length}`,
      params
    )

    return c.json({
      success: true,
      data: result.rows,
      pagination: {
        page,
        limit,
        total,
        total_pages: Math.ceil(total / limit)
      }
    })

  } catch (error: any) {
    console.error('[ADMIN] Users list error:', error.message)
    return c.json({ success: false, error: 'Failed to load users' }, 500)
  }
})

// GET /api/admin/users/:id - Get user details
admin.get('/users/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid user ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT u.*, c.name as clinic_name
       FROM users u
       LEFT JOIN clinics c ON u.clinic_id = c.id
       WHERE u.id = $1`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'User not found' }, 404)
    }

    // Don't return password_hash
    const user = result.rows[0]
    delete user.password_hash
    delete user.mfa_secret

    return c.json({
      success: true,
      data: user
    })

  } catch (error: any) {
    console.error('[ADMIN] User get error:', error.message)
    return c.json({ success: false, error: 'Failed to load user' }, 500)
  }
})

// POST /api/admin/users - Create new user
admin.post('/users', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const data = await c.req.json()

    // Validate required fields
    if (!data.email || !data.password || !data.first_name || !data.last_name) {
      return c.json({ success: false, error: 'Email, password, first_name, and last_name are required' }, 400)
    }

    // Check if email exists
    const existing = await pool.query(
      'SELECT id FROM users WHERE email = $1',
      [data.email.toLowerCase()]
    )

    if (existing.rows.length > 0) {
      return c.json({ success: false, error: 'Email already exists' }, 409)
    }

    // Hash password
    const passwordHash = await bcrypt.hash(data.password, 12)

    const result = await pool.query(
      `INSERT INTO users (
        email, password_hash, first_name, last_name, role, title,
        clinic_id, license_number, license_state, npi_number, phone,
        account_status, email_verified, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW(), NOW())
      RETURNING id, email, first_name, last_name, role, account_status`,
      [
        data.email.toLowerCase(),
        passwordHash,
        data.first_name,
        data.last_name,
        data.role || 'clinician',
        data.title || null,
        data.clinic_id || null,
        data.license_number || null,
        data.license_state || null,
        data.npi_number || null,
        data.phone || null,
        data.account_status || 'active',
        true // auto-verify when created by admin
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'User created successfully'
    })

  } catch (error: any) {
    console.error('[ADMIN] User create error:', error.message)
    return c.json({ success: false, error: 'Failed to create user' }, 500)
  }
})

// PUT /api/admin/users/:id - Update user
admin.put('/users/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid user ID' }, 400)
  }

  try {
    const updates = await c.req.json()

    const allowedFields = [
      'first_name', 'last_name', 'role', 'title', 'clinic_id',
      'license_number', 'license_state', 'npi_number', 'phone',
      'account_status', 'specialty', 'email_verified'
    ]

    const setFields: string[] = []
    const values: any[] = []

    for (const field of allowedFields) {
      if (updates[field] !== undefined) {
        setFields.push(`${field} = $${values.length + 1}`)
        values.push(updates[field])
      }
    }

    // Handle password reset separately
    if (updates.password) {
      const passwordHash = await bcrypt.hash(updates.password, 12)
      setFields.push(`password_hash = $${values.length + 1}`)
      values.push(passwordHash)
      setFields.push(`password_changed_at = NOW()`)
    }

    if (setFields.length === 0) {
      return c.json({ success: false, error: 'No valid fields to update' }, 400)
    }

    setFields.push('updated_at = NOW()')
    values.push(id)

    const result = await pool.query(
      `UPDATE users SET ${setFields.join(', ')} WHERE id = $${values.length} RETURNING id, email, first_name, last_name, role, account_status`,
      values
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'User not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'User updated successfully'
    })

  } catch (error: any) {
    console.error('[ADMIN] User update error:', error.message)
    return c.json({ success: false, error: 'Failed to update user' }, 500)
  }
})

// DELETE /api/admin/users/:id - Delete user
admin.delete('/users/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid user ID' }, 400)
  }

  // Prevent deleting yourself
  const currentUserId = c.get('clinicianId')
  if (id === currentUserId) {
    return c.json({ success: false, error: 'Cannot delete your own account' }, 400)
  }

  try {
    const result = await pool.query(
      'DELETE FROM users WHERE id = $1 RETURNING id',
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'User not found' }, 404)
    }

    return c.json({
      success: true,
      message: 'User deleted successfully'
    })

  } catch (error: any) {
    console.error('[ADMIN] User delete error:', error.message)
    return c.json({ success: false, error: 'Failed to delete user' }, 500)
  }
})

// GET /api/admin/clinics - List clinics
admin.get('/clinics', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const result = await pool.query(
      `SELECT c.*, COUNT(u.id) as user_count
       FROM clinics c
       LEFT JOIN users u ON c.id = u.clinic_id
       GROUP BY c.id
       ORDER BY c.name`
    )

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[ADMIN] Clinics error:', error.message)
    return c.json({ success: false, error: 'Failed to load clinics' }, 500)
  }
})

// POST /api/admin/clinics - Create clinic
admin.post('/clinics', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const data = await c.req.json()

    if (!data.name) {
      return c.json({ success: false, error: 'Clinic name is required' }, 400)
    }

    const result = await pool.query(
      `INSERT INTO clinics (
        name, address_line1, address_line2, city, state, zip_code,
        phone, email, license_number, npi_number, tax_id, settings,
        created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW())
      RETURNING *`,
      [
        data.name,
        data.address_line1 || null,
        data.address_line2 || null,
        data.city || null,
        data.state || null,
        data.zip_code || null,
        data.phone || null,
        data.email || null,
        data.license_number || null,
        data.npi_number || null,
        data.tax_id || null,
        data.settings ? JSON.stringify(data.settings) : '{}'
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Clinic created successfully'
    })

  } catch (error: any) {
    console.error('[ADMIN] Clinic create error:', error.message)
    return c.json({ success: false, error: 'Failed to create clinic' }, 500)
  }
})

// GET /api/admin/audit-logs - View audit logs
admin.get('/audit-logs', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const url = new URL(c.req.url)
    const page = parseInt(url.searchParams.get('page') || '1')
    const limit = parseInt(url.searchParams.get('limit') || '50')
    const userId = url.searchParams.get('user_id')
    const action = url.searchParams.get('action')
    const startDate = url.searchParams.get('start_date')
    const endDate = url.searchParams.get('end_date')
    const offset = (page - 1) * limit

    let whereClause = 'WHERE 1=1'
    const params: any[] = []

    if (userId) {
      params.push(parseInt(userId))
      whereClause += ` AND user_id = $${params.length}`
    }

    if (action) {
      params.push(action)
      whereClause += ` AND action = $${params.length}`
    }

    if (startDate) {
      params.push(startDate)
      whereClause += ` AND DATE(created_at) >= $${params.length}`
    }

    if (endDate) {
      params.push(endDate)
      whereClause += ` AND DATE(created_at) <= $${params.length}`
    }

    // Get total count
    const countResult = await pool.query(
      `SELECT COUNT(*) as total FROM audit_logs ${whereClause}`,
      params
    )
    const total = parseInt(countResult.rows[0].total)

    // Get logs
    params.push(limit, offset)
    const result = await pool.query(
      `SELECT al.*, 
              u.first_name as user_first_name, 
              u.last_name as user_last_name
       FROM audit_logs al
       LEFT JOIN users u ON al.user_id = u.id
       ${whereClause}
       ORDER BY al.created_at DESC
       LIMIT $${params.length - 1} OFFSET $${params.length}`,
      params
    )

    return c.json({
      success: true,
      data: result.rows,
      pagination: {
        page,
        limit,
        total,
        total_pages: Math.ceil(total / limit)
      }
    })

  } catch (error: any) {
    console.error('[ADMIN] Audit logs error:', error.message)
    return c.json({ success: false, error: 'Failed to load audit logs' }, 500)
  }
})

// GET /api/admin/system-settings - Get system settings
admin.get('/system-settings', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const result = await pool.query('SELECT * FROM system_settings ORDER BY setting_key')

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[ADMIN] Settings error:', error.message)
    return c.json({ success: false, error: 'Failed to load settings' }, 500)
  }
})

// PUT /api/admin/system-settings/:key - Update system setting
admin.put('/system-settings/:key', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const key = c.req.param('key')
  const data = await c.req.json()
  const userId = c.get('clinicianId')

  try {
    const result = await pool.query(
      `INSERT INTO system_settings (setting_key, setting_value, setting_type, description, updated_by, updated_at)
       VALUES ($1, $2, $3, $4, $5, NOW())
       ON CONFLICT (setting_key) DO UPDATE SET
         setting_value = EXCLUDED.setting_value,
         updated_by = EXCLUDED.updated_by,
         updated_at = NOW()
       RETURNING *`,
      [
        key,
        JSON.stringify(data.value),
        data.type || 'string',
        data.description || null,
        userId
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Setting updated successfully'
    })

  } catch (error: any) {
    console.error('[ADMIN] Setting update error:', error.message)
    return c.json({ success: false, error: 'Failed to update setting' }, 500)
  }
})

// GET /api/admin/analytics - Get system analytics
admin.get('/analytics', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    // User statistics
    const userStats = await pool.query(`
      SELECT 
        COUNT(*) as total_users,
        COUNT(CASE WHEN role = 'admin' THEN 1 END) as admins,
        COUNT(CASE WHEN role = 'clinician' THEN 1 END) as clinicians,
        COUNT(CASE WHEN role = 'patient' THEN 1 END) as patients,
        COUNT(CASE WHEN account_status = 'active' THEN 1 END) as active_users,
        COUNT(CASE WHEN last_login_at > NOW() - INTERVAL '30 days' THEN 1 END) as active_last_30_days
      FROM users
    `)

    // Patient statistics
    const patientStats = await pool.query(`
      SELECT 
        COUNT(*) as total_patients,
        COUNT(CASE WHEN patient_status = 'active' THEN 1 END) as active_patients,
        COUNT(CASE WHEN created_at > NOW() - INTERVAL '30 days' THEN 1 END) as new_last_30_days
      FROM patients
    `)

    // Assessment statistics
    const assessmentStats = await pool.query(`
      SELECT 
        COUNT(*) as total_assessments,
        COUNT(CASE WHEN session_date > NOW() - INTERVAL '30 days' THEN 1 END) as last_30_days,
        AVG(duration_minutes) as avg_duration
      FROM assessments
    `)

    // Exercise prescription statistics
    const prescriptionStats = await pool.query(`
      SELECT 
        COUNT(*) as total_prescriptions,
        COUNT(CASE WHEN prescription_status = 'active' THEN 1 END) as active_prescriptions,
        AVG(compliance_percentage) as avg_compliance
      FROM prescribed_exercises
    `)

    // Recent activity
    const recentActivity = await pool.query(`
      SELECT action, resource_type, created_at, user_email
      FROM audit_logs
      ORDER BY created_at DESC
      LIMIT 20
    `)

    return c.json({
      success: true,
      data: {
        users: userStats.rows[0],
        patients: patientStats.rows[0],
        assessments: assessmentStats.rows[0],
        prescriptions: prescriptionStats.rows[0],
        recent_activity: recentActivity.rows
      }
    })

  } catch (error: any) {
    console.error('[ADMIN] Analytics error:', error.message)
    return c.json({ success: false, error: 'Failed to load analytics' }, 500)
  }
})

export default admin
