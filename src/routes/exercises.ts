// Exercise Routes - Full Database Integration with 100+ Exercises

import { Hono } from 'hono'
import type { Bindings, Variables } from '../types.js'
import { authMiddleware } from '../middleware/auth.js'
import { validate, prescriptionSchema } from '../middleware/validation.js'
import { getPool } from '../database.js'

const exercises = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Cache configuration
const CACHE_TTL = 3600 // 1 hour in seconds

// Apply auth middleware to protected routes
exercises.use('/exercises', authMiddleware)
exercises.use('/exercises/*', authMiddleware)
exercises.use('/prescriptions', authMiddleware)
exercises.use('/prescriptions/*', authMiddleware)

// GET /api/exercises - List all exercises with search and filter
exercises.get('/exercises', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const url = new URL(c.req.url)
    const category = url.searchParams.get('category')
    const bodyRegion = url.searchParams.get('body_region')
    const difficulty = url.searchParams.get('difficulty')
    const search = url.searchParams.get('search')
    const page = parseInt(url.searchParams.get('page') || '1')
    const limit = parseInt(url.searchParams.get('limit') || '50')
    const offset = (page - 1) * limit

    // Build query
    let whereClause = 'WHERE is_active = true'
    const params: any[] = []

    if (category) {
      params.push(category)
      whereClause += ` AND exercise_category = $${params.length}`
    }

    if (bodyRegion) {
      params.push(bodyRegion)
      whereClause += ` AND body_region = $${params.length}`
    }

    if (difficulty) {
      params.push(difficulty)
      whereClause += ` AND difficulty_level = $${params.length}`
    }

    if (search) {
      params.push(`%${search}%`, `%${search}%`)
      whereClause += ` AND (exercise_name ILIKE $${params.length - 1} OR description ILIKE $${params.length})`
    }

    // Get total count
    const countResult = await pool.query(
      `SELECT COUNT(*) as total FROM exercises ${whereClause}`,
      params
    )
    const total = parseInt(countResult.rows[0].total)

    // Get exercises
    params.push(limit, offset)
    const result = await pool.query(
      `SELECT id, exercise_name, exercise_category, body_region, target_muscles,
              target_joints, difficulty_level, description, demo_video_url, 
              demo_image_url, equipment_needed, estimated_duration_seconds,
              sets_default, reps_default, hold_seconds_default, tags, cpt_codes
       FROM exercises ${whereClause}
       ORDER BY exercise_name
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
    console.error('[EXERCISES] List error:', error.message)
    return c.json({ success: false, error: 'Failed to load exercises' }, 500)
  }
})

// GET /api/exercises/categories - Get all exercise categories
exercises.get('/exercises/categories', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const result = await pool.query(
      `SELECT exercise_category as category, body_region, COUNT(*) as count
       FROM exercises 
       WHERE is_active = true
       GROUP BY exercise_category, body_region
       ORDER BY exercise_category, body_region`
    )

    // Group by category
    const categories: Record<string, any> = {}
    result.rows.forEach(row => {
      if (!categories[row.category]) {
        categories[row.category] = {
          category: row.category,
          regions: [],
          total: 0
        }
      }
      categories[row.category].regions.push({
        region: row.body_region,
        count: parseInt(row.count)
      })
      categories[row.category].total += parseInt(row.count)
    })

    return c.json({
      success: true,
      data: Object.values(categories)
    })

  } catch (error: any) {
    console.error('[EXERCISES] Categories error:', error.message)
    return c.json({ success: false, error: 'Failed to load categories' }, 500)
  }
})

// GET /api/exercises/:id - Get single exercise
exercises.get('/exercises/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid exercise ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT * FROM exercises WHERE id = $1 AND is_active = true`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Exercise not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0]
    })

  } catch (error: any) {
    console.error('[EXERCISES] Get error:', error.message)
    return c.json({ success: false, error: 'Failed to load exercise' }, 500)
  }
})

// POST /api/exercises - Create new exercise (admin/clinician only)
exercises.post('/exercises', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const clinician = c.get('clinician')
  if (!clinician || !['admin', 'clinician'].includes(clinician.role)) {
    return c.json({ success: false, error: 'Insufficient permissions' }, 403)
  }

  try {
    const data = await c.req.json()

    const result = await pool.query(
      `INSERT INTO exercises (
        exercise_name, exercise_category, body_region, target_muscles, target_joints,
        target_movements, difficulty_level, description, instructions, setup_instructions,
        cues, contraindications, demo_video_url, demo_image_url, thumbnail_url,
        equipment_needed, estimated_duration_seconds, sets_default, reps_default,
        hold_seconds_default, tags, cpt_codes, created_by, is_active, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, true, NOW(), NOW())
      RETURNING *`,
      [
        data.exercise_name,
        data.exercise_category,
        data.body_region,
        JSON.stringify(data.target_muscles || []),
        JSON.stringify(data.target_joints || []),
        JSON.stringify(data.target_movements || []),
        data.difficulty_level,
        data.description,
        data.instructions,
        data.setup_instructions,
        data.cues,
        data.contraindications,
        data.demo_video_url,
        data.demo_image_url,
        data.thumbnail_url,
        JSON.stringify(data.equipment_needed || []),
        data.estimated_duration_seconds,
        data.sets_default || 3,
        data.reps_default || 10,
        data.hold_seconds_default,
        JSON.stringify(data.tags || []),
        JSON.stringify(data.cpt_codes || []),
        clinician.id
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Exercise created successfully'
    })

  } catch (error: any) {
    console.error('[EXERCISES] Create error:', error.message)
    return c.json({ success: false, error: 'Failed to create exercise' }, 500)
  }
})

// PUT /api/exercises/:id - Update exercise
exercises.put('/exercises/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const clinician = c.get('clinician')
  if (!clinician || !['admin', 'clinician'].includes(clinician.role)) {
    return c.json({ success: false, error: 'Insufficient permissions' }, 403)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid exercise ID' }, 400)
  }

  try {
    const updates = await c.req.json()

    const allowedFields = [
      'exercise_name', 'exercise_category', 'body_region', 'target_muscles',
      'target_joints', 'difficulty_level', 'description', 'instructions',
      'contraindications', 'demo_video_url', 'equipment_needed',
      'estimated_duration_seconds', 'sets_default', 'reps_default', 'is_active'
    ]

    const setFields: string[] = []
    const values: any[] = []

    for (const field of allowedFields) {
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
      `UPDATE exercises SET ${setFields.join(', ')} WHERE id = $${values.length} RETURNING *`,
      values
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Exercise not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Exercise updated successfully'
    })

  } catch (error: any) {
    console.error('[EXERCISES] Update error:', error.message)
    return c.json({ success: false, error: 'Failed to update exercise' }, 500)
  }
})

// POST /api/prescriptions - Create exercise prescription
exercises.post('/prescriptions', validate(prescriptionSchema), async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const data = c.get('validatedData')
    const clinician = c.get('clinician')

    // Verify patient exists
    const patient = await pool.query(
      'SELECT id FROM patients WHERE id = $1',
      [data.patient_id]
    )

    if (patient.rows.length === 0) {
      return c.json({ success: false, error: 'Patient not found' }, 404)
    }

    // Verify exercise exists
    const exercise = await pool.query(
      'SELECT id FROM exercises WHERE id = $1',
      [data.exercise_id]
    )

    if (exercise.rows.length === 0) {
      return c.json({ success: false, error: 'Exercise not found' }, 404)
    }

    const result = await pool.query(
      `INSERT INTO prescribed_exercises (
        patient_id, assessment_id, exercise_id, prescribed_by,
        sets, repetitions, hold_duration_seconds, rest_between_sets_seconds,
        times_per_week, total_weeks, start_date, end_date,
        prescription_status, clinical_reason, target_deficiency, prescribed_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'active', $13, $14, NOW())
      RETURNING *`,
      [
        data.patient_id,
        data.assessment_id || null,
        data.exercise_id,
        clinician?.id,
        data.sets,
        data.repetitions,
        data.hold_duration_seconds || null,
        data.rest_between_sets_seconds || 60,
        data.times_per_week,
        data.total_weeks || 4,
        data.start_date || new Date().toISOString().split('T')[0],
        data.end_date || null,
        data.clinical_reason || null,
        data.target_deficiency || null
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Exercise prescribed successfully'
    })

  } catch (error: any) {
    console.error('[EXERCISES] Prescription error:', error.message)
    return c.json({ success: false, error: 'Failed to create prescription' }, 500)
  }
})

// GET /api/prescriptions/patient/:patientId - Get patient prescriptions
exercises.get('/prescriptions/patient/:patientId', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const patientId = parseInt(c.req.param('patientId'))
  if (isNaN(patientId)) {
    return c.json({ success: false, error: 'Invalid patient ID' }, 400)
  }

  try {
    const status = c.req.query('status') || 'active'

    const result = await pool.query(
      `SELECT pe.*, 
              e.exercise_name, e.exercise_category, e.description, 
              e.demo_video_url, e.instructions, e.target_muscles,
              u.first_name as prescribed_by_first_name,
              u.last_name as prescribed_by_last_name
       FROM prescribed_exercises pe
       JOIN exercises e ON pe.exercise_id = e.id
       LEFT JOIN users u ON pe.prescribed_by = u.id
       WHERE pe.patient_id = $1 AND pe.prescription_status = $2
       ORDER BY pe.prescribed_at DESC`,
      [patientId, status]
    )

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[EXERCISES] Get prescriptions error:', error.message)
    return c.json({ success: false, error: 'Failed to load prescriptions' }, 500)
  }
})

// PUT /api/prescriptions/:id - Update prescription
exercises.put('/prescriptions/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid prescription ID' }, 400)
  }

  try {
    const updates = await c.req.json()

    const allowedFields = [
      'sets', 'repetitions', 'times_per_week', 'prescription_status',
      'compliance_percentage', 'patient_notes', 'clinician_response'
    ]

    const setFields: string[] = []
    const values: any[] = []

    for (const field of allowedFields) {
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
      `UPDATE prescribed_exercises SET ${setFields.join(', ')} WHERE id = $${values.length} RETURNING *`,
      values
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Prescription not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Prescription updated successfully'
    })

  } catch (error: any) {
    console.error('[EXERCISES] Update prescription error:', error.message)
    return c.json({ success: false, error: 'Failed to update prescription' }, 500)
  }
})

// POST /api/prescriptions/:id/complete - Mark prescription as completed
exercises.post('/prescriptions/:id/complete', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid prescription ID' }, 400)
  }

  try {
    const result = await pool.query(
      `UPDATE prescribed_exercises SET prescription_status = 'completed', updated_at = NOW() WHERE id = $1 RETURNING *`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Prescription not found' }, 404)
    }

    return c.json({
      success: true,
      message: 'Prescription marked as completed'
    })

  } catch (error: any) {
    console.error('[EXERCISES] Complete prescription error:', error.message)
    return c.json({ success: false, error: 'Failed to complete prescription' }, 500)
  }
})

// POST /api/exercise-sessions - Record exercise session completion
exercises.post('/exercise-sessions', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const data = await c.req.json()
    const clinician = c.get('clinician')

    const result = await pool.query(
      `INSERT INTO exercise_sessions (
        patient_id, prescribed_exercise_id, session_date, completed,
        sets_completed, reps_completed, duration_seconds,
        form_quality_score, pain_level_during, difficulty_rating,
        patient_notes, recording_url, billable, created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW())
      RETURNING *`,
      [
        data.patient_id,
        data.prescribed_exercise_id || null,
        data.session_date || new Date().toISOString(),
        data.completed !== false,
        data.sets_completed || 0,
        data.reps_completed || 0,
        data.duration_seconds || 0,
        data.form_quality_score || null,
        data.pain_level_during || null,
        data.difficulty_rating || null,
        data.patient_notes || null,
        data.recording_url || null,
        data.billable || false
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Exercise session recorded'
    })

  } catch (error: any) {
    console.error('[EXERCISES] Session error:', error.message)
    return c.json({ success: false, error: 'Failed to record session' }, 500)
  }
})

export default exercises
