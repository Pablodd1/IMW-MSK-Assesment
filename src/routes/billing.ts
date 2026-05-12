// Billing Routes - Medicare Compliance and CPT Code Management

import { Hono } from 'hono'
import type { Bindings, Variables } from '../types.js'
import { authMiddleware } from '../middleware/auth.js'
import { validate, billingEventSchema } from '../middleware/validation.js'
import { getPool } from '../database.js'

const billing = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Apply auth middleware
billing.use('*', authMiddleware)

// GET /api/billing/cpt-codes - List all CPT codes
billing.get('/cpt-codes', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const category = c.req.query('category')
    
    let whereClause = ''
    const params: any[] = []
    
    if (category) {
      whereClause = 'WHERE code_category = $1'
      params.push(category)
    }

    const result = await pool.query(
      `SELECT * FROM cpt_codes ${whereClause} ORDER BY cpt_code`,
      params
    )

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[BILLING] CPT codes error:', error.message)
    return c.json({ success: false, error: 'Failed to load CPT codes' }, 500)
  }
})

// GET /api/billing/cpt-codes/:code - Get specific CPT code
billing.get('/cpt-codes/:code', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const code = c.req.param('code')

  try {
    const result = await pool.query(
      'SELECT * FROM cpt_codes WHERE cpt_code = $1',
      [code]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'CPT code not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0]
    })

  } catch (error: any) {
    console.error('[BILLING] CPT code error:', error.message)
    return c.json({ success: false, error: 'Failed to load CPT code' }, 500)
  }
})

// POST /api/billing/calculate-eight-minute - Calculate 8-minute rule units
billing.post('/calculate-eight-minute', async (c) => {
  try {
    const { minutes } = await c.req.json()

    if (typeof minutes !== 'number' || minutes < 0) {
      return c.json({ success: false, error: 'Valid minutes required' }, 400)
    }

    // Medicare 8-minute rule calculation
    // 8-22 min = 1 unit, 23-37 min = 2 units, 38-52 min = 3 units, etc.
    let units = 0
    if (minutes >= 8 && minutes <= 22) {
      units = 1
    } else if (minutes >= 23 && minutes <= 37) {
      units = 2
    } else if (minutes >= 38 && minutes <= 52) {
      units = 3
    } else if (minutes >= 53 && minutes <= 67) {
      units = 4
    } else if (minutes >= 68 && minutes <= 82) {
      units = 5
    } else if (minutes >= 83 && minutes <= 97) {
      units = 6
    } else if (minutes >= 98) {
      units = Math.ceil((minutes - 7) / 15)
    }

    const billable = minutes >= 8

    return c.json({
      success: true,
      data: {
        minutes,
        units,
        billable,
        remainder_minutes: billable ? minutes - (units - 1) * 15 : minutes
      }
    })

  } catch (error: any) {
    console.error('[BILLING] 8-minute calculation error:', error.message)
    return c.json({ success: false, error: 'Calculation failed' }, 500)
  }
})

// POST /api/billing/events - Create billable event
billing.post('/events', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const data = await c.req.json()
    const clinician = c.get('clinician')

    // Validate required fields
    if (!data.patient_id || !data.cpt_code || !data.service_date) {
      return c.json({ 
        success: false, 
        error: 'patient_id, cpt_code, and service_date are required' 
      }, 400)
    }

    // Get CPT code info
    const cptResult = await pool.query(
      'SELECT * FROM cpt_codes WHERE cpt_code = $1',
      [data.cpt_code]
    )

    if (cptResult.rows.length === 0) {
      return c.json({ success: false, error: 'Invalid CPT code' }, 400)
    }

    const cptCode = cptResult.rows[0]
    const durationMinutes = data.duration_minutes || 0

    // Calculate units based on 8-minute rule
    let units = 1
    if (cptCode.minimum_duration_minutes && durationMinutes > 0) {
      if (durationMinutes >= 8 && durationMinutes <= 22) units = 1
      else if (durationMinutes >= 23 && durationMinutes <= 37) units = 2
      else if (durationMinutes >= 38 && durationMinutes <= 52) units = 3
      else if (durationMinutes >= 53 && durationMinutes <= 67) units = 4
      else if (durationMinutes >= 68) units = Math.ceil((durationMinutes - 7) / 15)
    }

    // Insert billable event
    const result = await pool.query(
      `INSERT INTO billable_events (
        patient_id, assessment_id, exercise_session_id, provider_id,
        cpt_code_id, cpt_code, service_date, duration_minutes, units,
        clinical_note, medical_necessity, billing_status, provider_npi, created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW())
      RETURNING *`,
      [
        data.patient_id,
        data.assessment_id || null,
        data.exercise_session_id || null,
        clinician?.id,
        cptCode.id,
        data.cpt_code,
        data.service_date,
        durationMinutes,
        units,
        data.clinical_note || null,
        data.medical_necessity || null,
        'pending',
        clinician?.npi_number || null
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Billable event created successfully'
    })

  } catch (error: any) {
    console.error('[BILLING] Create event error:', error.message)
    return c.json({ success: false, error: 'Failed to create billable event' }, 500)
  }
})

// GET /api/billing/events - List billable events
billing.get('/events', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const url = new URL(c.req.url)
    const patientId = url.searchParams.get('patient_id')
    const status = url.searchParams.get('status')
    const startDate = url.searchParams.get('start_date')
    const endDate = url.searchParams.get('end_date')
    const page = parseInt(url.searchParams.get('page') || '1')
    const limit = parseInt(url.searchParams.get('limit') || '20')
    const offset = (page - 1) * limit

    let whereClause = 'WHERE 1=1'
    const params: any[] = []

    if (patientId) {
      params.push(parseInt(patientId))
      whereClause += ` AND be.patient_id = $${params.length}`
    }

    if (status) {
      params.push(status)
      whereClause += ` AND be.billing_status = $${params.length}`
    }

    if (startDate) {
      params.push(startDate)
      whereClause += ` AND be.service_date >= $${params.length}`
    }

    if (endDate) {
      params.push(endDate)
      whereClause += ` AND be.service_date <= $${params.length}`
    }

    // Get total count
    const countResult = await pool.query(
      `SELECT COUNT(*) as total FROM billable_events be ${whereClause}`,
      params
    )
    const total = parseInt(countResult.rows[0].total)

    // Get events
    params.push(limit, offset)
    const result = await pool.query(
      `SELECT be.*, 
              p.first_name as patient_first_name, 
              p.last_name as patient_last_name,
              p.insurance_provider,
              c.code_description,
              u.first_name as provider_first_name,
              u.last_name as provider_last_name
       FROM billable_events be
       JOIN patients p ON be.patient_id = p.id
       JOIN cpt_codes c ON be.cpt_code_id = c.id
       LEFT JOIN users u ON be.provider_id = u.id
       ${whereClause}
       ORDER BY be.service_date DESC, be.created_at DESC
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
    console.error('[BILLING] Events error:', error.message)
    return c.json({ success: false, error: 'Failed to load billable events' }, 500)
  }
})

// PUT /api/billing/events/:id - Update billable event status
billing.put('/events/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid event ID' }, 400)
  }

  try {
    const updates = await c.req.json()
    const allowedFields = ['billing_status', 'claim_id', 'amount_billed', 'clinical_note']

    const setFields: string[] = []
    const values: any[] = []

    for (const field of allowedFields) {
      if (updates[field] !== undefined) {
        setFields.push(`${field} = $${values.length + 1}`)
        values.push(updates[field])
      }
    }

    if (updates.billing_status === 'submitted') {
      setFields.push('submitted_at = NOW()')
    }

    if (updates.billing_status === 'paid') {
      setFields.push('paid_at = NOW()')
    }

    if (setFields.length === 0) {
      return c.json({ success: false, error: 'No valid fields to update' }, 400)
    }

    values.push(id)
    const result = await pool.query(
      `UPDATE billable_events SET ${setFields.join(', ')} WHERE id = $${values.length} RETURNING *`,
      values
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Billable event not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Billable event updated successfully'
    })

  } catch (error: any) {
    console.error('[BILLING] Update event error:', error.message)
    return c.json({ success: false, error: 'Failed to update billable event' }, 500)
  }
})

// GET /api/billing/patient/:patientId/summary - Get patient billing summary
billing.get('/patient/:patientId/summary', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const patientId = parseInt(c.req.param('patientId'))
  if (isNaN(patientId)) {
    return c.json({ success: false, error: 'Invalid patient ID' }, 400)
  }

  try {
    // Get summary statistics
    const summaryResult = await pool.query(
      `SELECT 
        COUNT(*) as total_events,
        COUNT(CASE WHEN billing_status = 'pending' THEN 1 END) as pending_count,
        COUNT(CASE WHEN billing_status = 'submitted' THEN 1 END) as submitted_count,
        COUNT(CASE WHEN billing_status = 'paid' THEN 1 END) as paid_count,
        COUNT(CASE WHEN billing_status = 'denied' THEN 1 END) as denied_count,
        COALESCE(SUM(amount_billed), 0) as total_billed,
        COALESCE(SUM(amount_paid), 0) as total_paid,
        COALESCE(SUM(units), 0) as total_units
      FROM billable_events
      WHERE patient_id = $1`,
      [patientId]
    )

    // Get recent events
    const recentResult = await pool.query(
      `SELECT be.*, c.code_description
       FROM billable_events be
       JOIN cpt_codes c ON be.cpt_code_id = c.id
       WHERE be.patient_id = $1
       ORDER BY be.service_date DESC
       LIMIT 10`,
      [patientId]
    )

    // Get CPT code breakdown
    const cptBreakdown = await pool.query(
      `SELECT be.cpt_code, c.code_description, COUNT(*) as count, SUM(units) as total_units
       FROM billable_events be
       JOIN cpt_codes c ON be.cpt_code_id = c.id
       WHERE be.patient_id = $1
       GROUP BY be.cpt_code, c.code_description
       ORDER BY count DESC`,
      [patientId]
    )

    return c.json({
      success: true,
      data: {
        summary: summaryResult.rows[0],
        recent_events: recentResult.rows,
        cpt_breakdown: cptBreakdown.rows
      }
    })

  } catch (error: any) {
    console.error('[BILLING] Summary error:', error.message)
    return c.json({ success: false, error: 'Failed to load billing summary' }, 500)
  }
})

// GET /api/billing/dashboard - Get billing dashboard data
billing.get('/dashboard', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const clinician = c.get('clinician')
    const isAdmin = clinician?.role === 'admin'

    // Base query filters
    let providerFilter = ''
    const params: any[] = []

    if (!isAdmin && clinician?.id) {
      providerFilter = 'WHERE provider_id = $1'
      params.push(clinician.id)
    }

    // Monthly statistics
    const monthlyStats = await pool.query(
      `SELECT 
        DATE_TRUNC('month', service_date) as month,
        COUNT(*) as event_count,
        SUM(units) as total_units,
        SUM(amount_billed) as amount_billed,
        SUM(CASE WHEN billing_status = 'paid' THEN amount_paid ELSE 0 END) as amount_collected
      FROM billable_events
      ${providerFilter}
      AND service_date >= NOW() - INTERVAL '12 months'
      GROUP BY DATE_TRUNC('month', service_date)
      ORDER BY month DESC`,
      params
    )

    // Status breakdown
    const statusBreakdown = await pool.query(
      `SELECT billing_status, COUNT(*) as count, SUM(units) as total_units
       FROM billable_events
       ${providerFilter}
       GROUP BY billing_status`,
      params
    )

    // Top CPT codes
    const topCptCodes = await pool.query(
      `SELECT be.cpt_code, c.code_description, COUNT(*) as count
       FROM billable_events be
       JOIN cpt_codes c ON be.cpt_code_id = c.id
       ${providerFilter}
       GROUP BY be.cpt_code, c.code_description
       ORDER BY count DESC
       LIMIT 10`,
      params
    )

    // Pending claims (requires follow-up)
    const pendingClaims = await pool.query(
      `SELECT be.*, p.first_name as patient_first_name, p.last_name as patient_last_name
       FROM billable_events be
       JOIN patients p ON be.patient_id = p.id
       WHERE be.billing_status = 'pending'
       ${!isAdmin ? 'AND be.provider_id = $1' : ''}
       ORDER BY be.service_date DESC
       LIMIT 20`,
      isAdmin ? [] : [clinician?.id]
    )

    return c.json({
      success: true,
      data: {
        monthly_stats: monthlyStats.rows,
        status_breakdown: statusBreakdown.rows,
        top_cpt_codes: topCptCodes.rows,
        pending_claims: pendingClaims.rows
      }
    })

  } catch (error: any) {
    console.error('[BILLING] Dashboard error:', error.message)
    return c.json({ success: false, error: 'Failed to load dashboard' }, 500)
  }
})

export default billing
