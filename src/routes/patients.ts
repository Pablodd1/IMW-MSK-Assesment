import { Hono } from 'hono'
import type { Bindings } from '../types'
import { validate } from '../middleware/validation'
import { patientCreateSchema } from '../middleware/validation'
import { authMiddleware } from '../middleware/auth'

const patients = new Hono<{ Bindings: Bindings }>()

patients.use('/*', authMiddleware)

patients.post('/', validate(patientCreateSchema), async (c) => {
  try {
    const patient = c.get('validatedData')

    const result = await c.env.DB.prepare(`
      INSERT INTO patients (
        first_name, last_name, date_of_birth, gender, email, phone,
        emergency_contact_name, emergency_contact_phone,
        address_line1, city, state, zip_code,
        height_cm, weight_kg, insurance_provider
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      patient.first_name, patient.last_name, patient.date_of_birth,
      patient.gender, patient.email, patient.phone,
      patient.emergency_contact_name, patient.emergency_contact_phone,
      patient.address_line1, patient.city, patient.state, patient.zip_code,
      patient.height_cm, patient.weight_kg, patient.insurance_provider
    ).run()

    return c.json({
      success: true,
      data: { id: result.meta.last_row_id, ...patient }
    })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

patients.get('/', async (c) => {
  try {
    const { results } = (await c.env.DB.prepare(`
      SELECT * FROM patients ORDER BY created_at DESC
    `).all()) as any
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

patients.get('/:id', async (c) => {
  try {
    const id = c.req.param('id')
    const patient = await c.env.DB.prepare(`
      SELECT * FROM patients WHERE id = ?
    `).bind(id).first()

    if (!patient) {
      return c.json({ success: false, error: 'Patient not found' }, 404)
    }

    return c.json({ success: true, data: patient })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

patients.put('/:id', async (c) => {
  try {
    const id = c.req.param('id')
    const patient = await c.req.json()

    await c.env.DB.prepare(`
      UPDATE patients SET
        first_name = ?, last_name = ?, date_of_birth = ?, gender = ?,
        email = ?, phone = ?, emergency_contact_name = ?, emergency_contact_phone = ?,
        address_line1 = ?, city = ?, state = ?, zip_code = ?,
        height_cm = ?, weight_kg = ?, insurance_provider = ?,
        updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `).bind(
      patient.first_name, patient.last_name, patient.date_of_birth,
      patient.gender, patient.email, patient.phone,
      patient.emergency_contact_name, patient.emergency_contact_phone,
      patient.address_line1, patient.city, patient.state, patient.zip_code,
      patient.height_cm, patient.weight_kg, patient.insurance_provider,
      id
    ).run()

    return c.json({ success: true })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

patients.delete('/:id', async (c) => {
  try {
    const id = c.req.param('id')

    await c.env.DB.prepare(`
      UPDATE patients SET patient_status = 'inactive' WHERE id = ?
    `).bind(id).run()

    return c.json({ success: true })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

patients.post('/:id/medical-history', async (c) => {
  try {
    const patientId = c.req.param('id')
    const history = await c.req.json()

    const result = await c.env.DB.prepare(`
      INSERT INTO medical_history (
        patient_id, surgery_type, surgery_date, conditions, medications, allergies,
        current_pain_level, pain_location, activity_level, treatment_goals
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      patientId, history.surgery_type, history.surgery_date,
      JSON.stringify(history.conditions), JSON.stringify(history.medications),
      JSON.stringify(history.allergies), history.current_pain_level,
      JSON.stringify(history.pain_location), history.activity_level, history.treatment_goals
    ).run()

    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

patients.get('/:id/assessments', async (c) => {
  try {
    const patientId = c.req.param('id')
    const { results } = (await c.env.DB.prepare(`
      SELECT * FROM assessments WHERE patient_id = ? ORDER BY assessment_date DESC
    `).bind(patientId).all()) as any
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

export default patients
