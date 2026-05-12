// Patient Portal Routes - Patient-facing API endpoints

import { Hono } from 'hono'
import type { Bindings, Variables } from '../types.js'
import { authMiddleware } from '../middleware/auth.js'
import { getPool } from '../database.js'
import bcrypt from 'bcryptjs'

const portal = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Patient login (separate from clinician login)
portal.post('/login', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const { email, password, patient_id } = await c.req.json()

    // Find patient by email or ID
    let patientQuery = 'SELECT * FROM patients WHERE '
    let params: any[] = []

    if (email) {
      patientQuery += 'email = $1 AND portal_enabled = true'
      params.push(email.toLowerCase())
    } else if (patient_id) {
      patientQuery += 'id = $1 AND portal_enabled = true'
      params.push(parseInt(patient_id))
    } else {
      return c.json({ success: false, error: 'Email or patient_id required' }, 400)
    }

    const result = await pool.query(patientQuery, params)

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Invalid credentials or portal not enabled' }, 401)
    }

    const patient = result.rows[0]

    // Check password if set
    if (patient.portal_password_hash) {
      if (!password) {
        return c.json({ success: false, error: 'Password required' }, 401)
      }

      const valid = await bcrypt.compare(password, patient.portal_password_hash)
      if (!valid) {
        return c.json({ success: false, error: 'Invalid password' }, 401)
      }
    } else if (password) {
      // First time login - set password
      const hash = await bcrypt.hash(password, 12)
      await pool.query(
        'UPDATE patients SET portal_password_hash = $1, portal_last_login = NOW() WHERE id = $2',
        [hash, patient.id]
      )
    } else {
      return c.json({ success: false, error: 'Please set a password' }, 401)
    }

    // Update last login
    await pool.query(
      'UPDATE patients SET portal_last_login = NOW() WHERE id = $1',
      [patient.id]
    )

    // Generate simple token for patient (shorter-lived than clinician tokens)
    const token = btoa(JSON.stringify({
      id: patient.id,
      email: patient.email,
      type: 'patient',
      exp: Date.now() + (24 * 60 * 60 * 1000) // 24 hours
    }))

    return c.json({
      success: true,
      data: {
        id: patient.id,
        first_name: patient.first_name,
        last_name: patient.last_name,
        email: patient.email,
        token
      }
    })

  } catch (error: any) {
    console.error('[PORTAL] Login error:', error.message)
    return c.json({ success: false, error: 'Login failed' }, 500)
  }
})

// Patient portal middleware
const patientAuth = async (c: any, next: any) => {
  const token = c.req.header('Authorization')?.replace('Bearer ', '')

  if (!token) {
    return c.json({ success: false, error: 'Authentication required' }, 401)
  }

  try {
    const payload = JSON.parse(atob(token))
    
    if (payload.type !== 'patient') {
      return c.json({ success: false, error: 'Invalid token type' }, 403)
    }

    if (payload.exp < Date.now()) {
      return c.json({ success: false, error: 'Token expired' }, 401)
    }

    c.set('patientId', payload.id)
    await next()
  } catch (e) {
    return c.json({ success: false, error: 'Invalid token' }, 401)
  }
}

// GET /api/portal/profile - Get patient profile
portal.get('/profile', patientAuth, async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const patientId = c.get('patientId')

  try {
    const result = await pool.query(
      `SELECT id, first_name, last_name, date_of_birth, gender, email, phone,
              address_line1, city, state, zip_code,
              height_cm, weight_kg, blood_type,
              primary_insurance_provider, referring_physician,
              primary_diagnosis_description, patient_status,
              created_at, updated_at
       FROM patients WHERE id = $1`,
      [patientId]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Patient not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0]
    })

  } catch (error: any) {
    console.error('[PORTAL] Profile error:', error.message)
    return c.json({ success: false, error: 'Failed to load profile' }, 500)
  }
})

// GET /api/portal/exercises - Get patient's prescribed exercises
portal.get('/exercises', patientAuth, async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const patientId = c.get('patientId')

  try {
    const result = await pool.query(
      `SELECT pe.*, 
              e.exercise_name, e.exercise_category, e.description,
              e.instructions, e.demo_video_url, e.target_muscles,
              e.sets_default, e.reps_default, e.hold_seconds_default,
              u.first_name as therapist_first_name,
              u.last_name as therapist_last_name
       FROM prescribed_exercises pe
       JOIN exercises e ON pe.exercise_id = e.id
       LEFT JOIN users u ON pe.prescribed_by = u.id
       WHERE pe.patient_id = $1 AND pe.prescription_status = 'active'
       ORDER BY pe.prescribed_at DESC`,
      [patientId]
    )

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[PORTAL] Exercises error:', error.message)
    return c.json({ success: false, error: 'Failed to load exercises' }, 500)
  }
})

// POST /api/portal/exercises/:id/complete - Mark exercise as completed
portal.post('/exercises/:id/complete', patientAuth, async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const patientId = c.get('patientId')
  const prescriptionId = parseInt(c.req.param('id'))

  if (isNaN(prescriptionId)) {
    return c.json({ success: false, error: 'Invalid prescription ID' }, 400)
  }

  try {
    const data = await c.req.json()

    // Create exercise session
    const result = await pool.query(
      `INSERT INTO exercise_sessions (
        patient_id, prescribed_exercise_id, session_date, completed,
        sets_completed, reps_completed, duration_seconds,
        form_quality_score, pain_level_during, difficulty_rating,
        patient_notes, recording_url, created_at
      ) VALUES ($1, $2, $3, true, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
      RETURNING *`,
      [
        patientId,
        prescriptionId,
        data.session_date || new Date().toISOString(),
        data.sets_completed || 0,
        data.reps_completed || 0,
        data.duration_seconds || 0,
        data.form_quality_score || null,
        data.pain_level_during || null,
        data.difficulty_rating || null,
        data.patient_notes || null,
        data.recording_url || null
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Exercise completed and logged'
    })

  } catch (error: any) {
    console.error('[PORTAL] Complete exercise error:', error.message)
    return c.json({ success: false, error: 'Failed to record completion' }, 500)
  }
})

// GET /api/portal/progress - Get patient progress summary
portal.get('/progress', patientAuth, async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const patientId = c.get('patientId')

  try {
    // Get exercise completion stats
    const statsResult = await pool.query(
      `SELECT 
        COUNT(DISTINCT pe.id) as total_prescribed,
        COUNT(DISTINCT CASE WHEN pe.prescription_status = 'active' THEN pe.id END) as active_count,
        COUNT(DISTINCT es.id) as sessions_completed,
        AVG(es.form_quality_score) as avg_form_score,
        AVG(es.pain_level_during) as avg_pain_level,
        AVG(pe.compliance_percentage) as overall_compliance
       FROM prescribed_exercises pe
       LEFT JOIN exercise_sessions es ON pe.id = es.prescribed_exercise_id AND es.completed = true
       WHERE pe.patient_id = $1`,
      [patientId]
    )

    // Get recent sessions
    const sessionsResult = await pool.query(
      `SELECT es.*, e.exercise_name
       FROM exercise_sessions es
       JOIN prescribed_exercises pe ON es.prescribed_exercise_id = pe.id
       JOIN exercises e ON pe.exercise_id = e.id
       WHERE es.patient_id = $1
       ORDER BY es.session_date DESC
       LIMIT 10`,
      [patientId]
    )

    // Get assessment history
    const assessmentsResult = await pool.query(
      `SELECT id, assessment_type, session_date, overall_score, 
              mobility_score, stability_score, assessment_status
       FROM assessments
       WHERE patient_id = $1
       ORDER BY session_date DESC
       LIMIT 5`,
      [patientId]
    )

    return c.json({
      success: true,
      data: {
        stats: statsResult.rows[0],
        recent_sessions: sessionsResult.rows,
        recent_assessments: assessmentsResult.rows
      }
    })

  } catch (error: any) {
    console.error('[PORTAL] Progress error:', error.message)
    return c.json({ success: false, error: 'Failed to load progress' }, 500)
  }
})

// POST /api/portal/messages - Send message to therapist
portal.post('/messages', patientAuth, async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const patientId = c.get('patientId')

  try {
    const { subject, content } = await c.req.json()

    if (!content) {
      return c.json({ success: false, error: 'Message content required' }, 400)
    }

    // Get primary therapist (most recent prescriber)
    const therapistResult = await pool.query(
      `SELECT prescribed_by FROM prescribed_exercises 
       WHERE patient_id = $1 AND prescribed_by IS NOT NULL
       ORDER BY prescribed_at DESC LIMIT 1`,
      [patientId]
    )

    const therapistId = therapistResult.rows[0]?.prescribed_by || null

    const result = await pool.query(
      `INSERT INTO messages (
        patient_id, sender_id, sender_type, subject, content, created_at
      ) VALUES ($1, $2, 'patient', $3, $4, NOW())
      RETURNING *`,
      [patientId, therapistId, subject || null, content]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Message sent'
    })

  } catch (error: any) {
    console.error('[PORTAL] Message error:', error.message)
    return c.json({ success: false, error: 'Failed to send message' }, 500)
  }
})

// GET /api/portal/messages - Get messages
portal.get('/messages', patientAuth, async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const patientId = c.get('patientId')

  try {
    const result = await pool.query(
      `SELECT m.*, 
              u.first_name as sender_first_name,
              u.last_name as sender_last_name
       FROM messages m
       LEFT JOIN users u ON m.sender_id = u.id
       WHERE m.patient_id = $1
       ORDER BY m.created_at DESC`,
      [patientId]
    )

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[PORTAL] Messages error:', error.message)
    return c.json({ success: false, error: 'Failed to load messages' }, 500)
  }
})

export default portal
