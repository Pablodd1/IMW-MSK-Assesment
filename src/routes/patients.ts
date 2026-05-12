// Patient Routes - Full CRUD with Database Integration and XSS Protection

import { Hono } from 'hono'
import type { Bindings, Variables, Patient, MedicalHistory } from '../types.js'
import { authMiddleware, requireRole } from '../middleware/auth.js'
import { validate, patientCreateSchema } from '../middleware/validation.js'
import { getPool } from '../database.js'
import { phiAudit } from '../middleware/hipaa.js'
import { sanitizeString, sanitizeEmail, sanitizePhone } from '../utils/security.js'

const patients = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Apply auth middleware to all routes
patients.use('*', authMiddleware)

// GET /api/patients - List all patients (with pagination)
patients.get('/', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    // Get query parameters for pagination and filtering
    const url = new URL(c.req.url)
    const page = parseInt(url.searchParams.get('page') || '1')
    const limit = parseInt(url.searchParams.get('limit') || '20')
    const status = url.searchParams.get('status')
    const search = url.searchParams.get('search')
    const offset = (page - 1) * limit

    // Build query
    let whereClause = 'WHERE 1=1'
    const params: any[] = []
    
    if (status) {
      params.push(status)
      whereClause += ` AND patient_status = $${params.length}`
    }
    
    if (search) {
      params.push(`%${search}%`, `%${search}%`, `%${search}%`)
      whereClause += ` AND (first_name ILIKE $${params.length - 2} OR last_name ILIKE $${params.length - 1} OR email ILIKE $${params.length})`
    }

    // Get total count
    const countResult = await pool.query(
      `SELECT COUNT(*) as total FROM patients ${whereClause}`,
      params
    )
    const total = parseInt(countResult.rows[0].total)

    // Get patients
    params.push(limit, offset)
    const result = await pool.query(
      `SELECT id, first_name, last_name, date_of_birth, gender, email, phone,
              patient_status, primary_diagnosis_description, created_at, updated_at
       FROM patients ${whereClause}
       ORDER BY last_name, first_name
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
    console.error('[PATIENTS] List error:', error.message)
    return c.json({ success: false, error: 'Failed to load patients' }, 500)
  }
})

// GET /api/patients/:id - Get single patient
patients.get('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid patient ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT p.*, 
              mh.surgery_type, mh.surgery_date, mh.surgery_description,
              mh.conditions, mh.medications, mh.allergies,
              mh.current_pain_level, mh.pain_location, mh.pain_description,
              mh.previous_pt_therapy, mh.activity_level, mh.treatment_goals,
              mh.occupation, mh.sports_activities
       FROM patients p
       LEFT JOIN medical_history mh ON p.id = mh.patient_id
       WHERE p.id = $1`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Patient not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0]
    })

  } catch (error: any) {
    console.error('[PATIENTS] Get error:', error.message)
    return c.json({ success: false, error: 'Failed to load patient' }, 500)
  }
})

// POST /api/patients - Create new patient
patients.post('/', validate(patientCreateSchema), async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const data = c.get('validatedData')
    const clinicianId = c.get('clinicianId')

    // Check for duplicate email if provided
    if (data.email) {
      const existing = await pool.query(
        'SELECT id FROM patients WHERE email = $1',
        [data.email.toLowerCase()]
      )
      if (existing.rows.length > 0) {
        return c.json({ 
          success: false, 
          error: 'Patient with this email already exists',
          code: 'DUPLICATE_EMAIL'
        }, 409)
      }
    }

    const result = await pool.query(
      `INSERT INTO patients (
        first_name, last_name, date_of_birth, gender, email, phone,
        emergency_contact_name, emergency_contact_phone,
        address_line1, address_line2, city, state, zip_code, country,
        height_cm, weight_kg, blood_type,
        primary_insurance_provider, primary_insurance_policy_number, primary_insurance_group_number,
        referring_physician, primary_diagnosis_description, notes,
        patient_status, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, NOW(), NOW())
      RETURNING *`,
      [
        data.first_name,
        data.last_name,
        data.date_of_birth,
        data.gender,
        data.email?.toLowerCase() || null,
        data.phone || null,
        data.emergency_contact_name || null,
        data.emergency_contact_phone || null,
        data.address_line1 || null,
        data.address_line2 || null,
        data.city || null,
        data.state || null,
        data.zip_code || null,
        data.country || 'US',
        data.height_cm || null,
        data.weight_kg || null,
        data.blood_type || null,
        data.insurance_provider || null,
        data.insurance_policy_number || null,
        data.insurance_group_number || null,
        data.referring_physician || null,
        data.primary_diagnosis || null,
        data.notes || null,
        'active'
      ]
    )

    const patient = result.rows[0]

    // Create empty medical history record
    await pool.query(
      `INSERT INTO medical_history (patient_id, created_at, updated_at) VALUES ($1, NOW(), NOW())`,
      [patient.id]
    )

    return c.json({
      success: true,
      data: patient,
      message: 'Patient created successfully'
    })

  } catch (error: any) {
    console.error('[PATIENTS] Create error:', error.message)
    return c.json({ success: false, error: 'Failed to create patient' }, 500)
  }
})

// PUT /api/patients/:id - Update patient
patients.put('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid patient ID' }, 400)
  }

  try {
    const rawUpdates = await c.req.json()
    
    // Text fields that need sanitization
    const textFields = [
      'first_name', 'last_name', 'gender', 'email', 'phone',
      'emergency_contact_name', 'emergency_contact_phone',
      'address_line1', 'address_line2', 'city', 'state', 'zip_code', 'country',
      'primary_insurance_provider', 'primary_insurance_policy_number', 'primary_insurance_group_number',
      'referring_physician', 'primary_diagnosis_description', 'notes', 'patient_status', 'blood_type'
    ]
    
    // Numeric fields (no sanitization)
    const numericFields = ['height_cm', 'weight_kg']

    // Sanitize inputs
    const updates: Record<string, any> = {}
    for (const field of textFields) {
      if (rawUpdates[field] !== undefined) {
        if (field === 'email') {
          updates[field] = sanitizeEmail(rawUpdates[field])
        } else if (field === 'phone' || field === 'emergency_contact_phone') {
          updates[field] = sanitizePhone(rawUpdates[field])
        } else {
          updates[field] = sanitizeString(rawUpdates[field])
        }
      }
    }
    
    for (const field of numericFields) {
      if (rawUpdates[field] !== undefined) {
        updates[field] = rawUpdates[field]
      }
    }
    
    // Handle date_of_birth separately
    if (rawUpdates.date_of_birth !== undefined) {
      updates.date_of_birth = rawUpdates.date_of_birth
    }

    const setFields: string[] = []
    const values: any[] = []

    for (const field of Object.keys(updates)) {
      if (updates[field] !== undefined) {
        setFields.push(`${field} = $${values.length + 1}`)
        values.push(updates[field])
      }
    }

    if (setFields.length === 0) {
      return c.json({ success: false, error: 'No valid fields to update' }, 400)
    }

    setFields.push('updated_at = NOW()')
    values.push(id)

    const result = await pool.query(
      `UPDATE patients SET ${setFields.join(', ')} WHERE id = $${values.length} RETURNING *`,
      values
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Patient not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Patient updated successfully'
    })

  } catch (error: any) {
    console.error('[PATIENTS] Update error:', error.message)
    return c.json({ success: false, error: 'Failed to update patient' }, 500)
  }
})

// DELETE /api/patients/:id - Soft delete patient (set to inactive)
patients.delete('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid patient ID' }, 400)
  }

  try {
    const result = await pool.query(
      `UPDATE patients SET patient_status = 'inactive', updated_at = NOW() WHERE id = $1 RETURNING id`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Patient not found' }, 404)
    }

    return c.json({
      success: true,
      message: 'Patient deactivated successfully'
    })

  } catch (error: any) {
    console.error('[PATIENTS] Delete error:', error.message)
    return c.json({ success: false, error: 'Failed to deactivate patient' }, 500)
  }
})

// GET /api/patients/:id/assessments - Get patient assessments
patients.get('/:id/assessments', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid patient ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT a.*, 
              u.first_name as clinician_first_name, 
              u.last_name as clinician_last_name
       FROM assessments a
       LEFT JOIN users u ON a.clinician_id = u.id
       WHERE a.patient_id = $1
       ORDER BY a.session_date DESC`,
      [id]
    )

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[PATIENTS] Assessments error:', error.message)
    return c.json({ success: false, error: 'Failed to load assessments' }, 500)
  }
})

// GET /api/patients/:id/prescriptions - Get patient prescriptions
patients.get('/:id/prescriptions', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid patient ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT pe.*, 
              e.exercise_name, e.exercise_category, e.description, e.demo_video_url,
              u.first_name as prescribed_by_first_name,
              u.last_name as prescribed_by_last_name
       FROM prescribed_exercises pe
       JOIN exercises e ON pe.exercise_id = e.id
       LEFT JOIN users u ON pe.prescribed_by = u.id
       WHERE pe.patient_id = $1
       ORDER BY pe.prescribed_at DESC`,
      [id]
    )

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[PATIENTS] Prescriptions error:', error.message)
    return c.json({ success: false, error: 'Failed to load prescriptions' }, 500)
  }
})

// POST /api/patients/:id/medical-history - Update medical history
patients.post('/:id/medical-history', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid patient ID' }, 400)
  }

  try {
    const rawData = await c.req.json()

    // SANITIZE medical history text fields
    const data = {
      surgery_type: sanitizeString(rawData.surgery_type),
      surgery_date: rawData.surgery_date,
      surgery_description: sanitizeString(rawData.surgery_description),
      conditions: Array.isArray(rawData.conditions) 
        ? rawData.conditions.map((c: string) => sanitizeString(c)).filter(Boolean)
        : [],
      medications: Array.isArray(rawData.medications)
        ? rawData.medications.map((m: string) => sanitizeString(m)).filter(Boolean)
        : [],
      allergies: Array.isArray(rawData.allergies)
        ? rawData.allergies.map((a: string) => sanitizeString(a)).filter(Boolean)
        : [],
      current_pain_level: rawData.current_pain_level,
      pain_location: Array.isArray(rawData.pain_location)
        ? rawData.pain_location.map((p: string) => sanitizeString(p)).filter(Boolean)
        : [],
      pain_description: sanitizeString(rawData.pain_description),
      previous_pt_therapy: rawData.previous_pt_therapy,
      activity_level: sanitizeString(rawData.activity_level),
      treatment_goals: sanitizeString(rawData.treatment_goals),
      occupation: sanitizeString(rawData.occupation),
      sports_activities: Array.isArray(rawData.sports_activities)
        ? rawData.sports_activities.map((s: string) => sanitizeString(s)).filter(Boolean)
        : []
    }

    // Check if medical history exists
    const existing = await pool.query(
      'SELECT id FROM medical_history WHERE patient_id = $1',
      [id]
    )

    let result
    if (existing.rows.length > 0) {
      // Update existing
      result = await pool.query(
        `UPDATE medical_history SET
          surgery_type = $2,
          surgery_date = $3,
          surgery_description = $4,
          conditions = $5,
          medications = $6,
          allergies = $7,
          current_pain_level = $8,
          pain_location = $9,
          pain_description = $10,
          previous_pt_therapy = $11,
          activity_level = $12,
          treatment_goals = $13,
          occupation = $14,
          sports_activities = $15,
          updated_at = NOW()
         WHERE patient_id = $1
         RETURNING *`,
        [
          id,
          data.surgery_type,
          data.surgery_date,
          data.surgery_description,
          JSON.stringify(data.conditions || []),
          JSON.stringify(data.medications || []),
          JSON.stringify(data.allergies || []),
          data.current_pain_level,
          JSON.stringify(data.pain_location || []),
          data.pain_description,
          data.previous_pt_therapy,
          data.activity_level,
          data.treatment_goals,
          data.occupation,
          JSON.stringify(data.sports_activities || [])
        ]
      )
    } else {
      // Create new
      result = await pool.query(
        `INSERT INTO medical_history (
          patient_id, surgery_type, surgery_date, surgery_description,
          conditions, medications, allergies, current_pain_level,
          pain_location, pain_description, previous_pt_therapy,
          activity_level, treatment_goals, occupation, sports_activities,
          created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, NOW(), NOW())
        RETURNING *`,
        [
          id,
          data.surgery_type,
          data.surgery_date,
          data.surgery_description,
          JSON.stringify(data.conditions || []),
          JSON.stringify(data.medications || []),
          JSON.stringify(data.allergies || []),
          data.current_pain_level,
          JSON.stringify(data.pain_location || []),
          data.pain_description,
          data.previous_pt_therapy,
          data.activity_level,
          data.treatment_goals,
          data.occupation,
          JSON.stringify(data.sports_activities || [])
        ]
      )
    }

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Medical history updated successfully'
    })

  } catch (error: any) {
    console.error('[PATIENTS] Medical history error:', error.message)
    return c.json({ success: false, error: 'Failed to update medical history' }, 500)
  }
})

export default patients
