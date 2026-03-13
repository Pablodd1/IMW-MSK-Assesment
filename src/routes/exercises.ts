import { Hono } from 'hono'
import type { Bindings, Exercise, PrescribedExercise } from '../types'
import { authMiddleware } from '../middleware/auth'
import { validate } from '../middleware/validation'
import { prescriptionSchema } from '../middleware/validation'

const exercises = new Hono<{ Bindings: Bindings }>()

const CACHE_KEY_EXERCISES = 'app:exercises'
const CACHE_KEY_BILLING = 'app:billing'
const CACHE_TTL = 3600 // 1 hour in seconds

async function getCachedData<T>(c: any, key: string): Promise<T | null> {
  const cached = await c.env.KV?.get(key)
  return cached ? JSON.parse(cached) : null
}

async function setCachedData(c: any, key: string, data: any) {
  await c.env.KV?.put(key, JSON.stringify(data), { expirationTtl: CACHE_TTL })
}

exercises.get('/exercises', async (c) => {
  try {
    const cached = await getCachedData<any[]>(c, CACHE_KEY_EXERCISES)
    if (cached) return c.json({ success: true, data: cached })

    const { results } = (await c.env.DB.prepare(`
      SELECT * FROM exercises ORDER BY exercise_name
    `).all()) as any

    await setCachedData(c, CACHE_KEY_EXERCISES, results)
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

exercises.get('/exercises/:id', async (c) => {
  try {
    const id = c.req.param('id')
    const exercise = await c.env.DB.prepare(`
      SELECT * FROM exercises WHERE id = ?
    `).bind(id).first()

    if (!exercise) {
      return c.json({ success: false, error: 'Exercise not found' }, 404)
    }

    return c.json({ success: true, data: exercise })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

exercises.post('/prescriptions', authMiddleware, validate(prescriptionSchema), async (c) => {
  try {
    const prescription = c.get('validatedData')
    const clinician = c.get('clinician')

    const result = await c.env.DB.prepare(`
      INSERT INTO prescribed_exercises (
        patient_id, exercise_id, assessment_id, sets, repetitions,
        hold_duration_seconds, rest_between_sets_seconds, times_per_week,
        total_weeks, clinical_reason, target_deficiency, prescribed_by, prescription_status
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
    `).bind(
      prescription.patient_id, prescription.exercise_id, prescription.assessment_id,
      prescription.sets, prescription.repetitions,
      prescription.hold_duration_seconds, prescription.rest_between_sets_seconds,
      prescription.times_per_week, prescription.total_weeks,
      prescription.clinical_reason, prescription.target_deficiency,
      clinician?.id
    ).run()

    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

exercises.get('/prescriptions/patient/:patientId', authMiddleware, async (c) => {
  try {
    const patientId = c.req.param('patientId')
    const { results } = (await c.env.DB.prepare(`
      SELECT pe.*, e.exercise_name, e.exercise_category, e.description
      FROM prescribed_exercises pe
      JOIN exercises e ON pe.exercise_id = e.id
      WHERE pe.patient_id = ? AND pe.prescription_status = 'active'
      ORDER BY pe.prescribed_at DESC
    `).bind(patientId).all()) as any
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

exercises.put('/prescriptions/:id', authMiddleware, async (c) => {
  try {
    const id = c.req.param('id')
    const updates = await c.req.json()

    const fields: string[] = []
    const values: any[] = []

    if (updates.sets !== undefined) { fields.push('sets = ?'); values.push(updates.sets) }
    if (updates.repetitions !== undefined) { fields.push('repetitions = ?'); values.push(updates.repetitions) }
    if (updates.times_per_week !== undefined) { fields.push('times_per_week = ?'); values.push(updates.times_per_week) }
    if (updates.prescription_status !== undefined) { fields.push('prescription_status = ?'); values.push(updates.prescription_status) }

    if (fields.length > 0) {
      values.push(id)
      await c.env.DB.prepare(`
        UPDATE prescribed_exercises SET ${fields.join(', ')} WHERE id = ?
      `).bind(...values).run()
    }

    return c.json({ success: true })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

export default exercises
