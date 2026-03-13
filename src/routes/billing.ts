import { Hono } from 'hono'
import type { Bindings } from '../types'
import { authMiddleware } from '../middleware/auth'

const billing = new Hono<{ Bindings: Bindings }>()

const CACHE_KEY_BILLING = 'app:billing'
const CACHE_TTL = 3600 // 1 hour in seconds

async function getCachedData<T>(c: any, key: string): Promise<T | null> {
  const cached = await c.env.KV?.get(key)
  return cached ? JSON.parse(cached) : null
}

async function setCachedData(c: any, key: string, data: any) {
  await c.env.KV?.put(key, JSON.stringify(data), { expirationTtl: CACHE_TTL })
}

billing.get('/codes', async (c) => {
  try {
    const cached = await getCachedData<any[]>(c, CACHE_KEY_BILLING)
    if (cached) return c.json({ success: true, data: cached })

    const { results } = (await c.env.DB.prepare(`
      SELECT * FROM billing_codes ORDER BY cpt_code
    `).all()) as any

    await setCachedData(c, CACHE_KEY_BILLING, results)
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

billing.post('/events', authMiddleware, async (c) => {
  try {
    const clinician = c.get('clinician')
    const event = await c.req.json()

    const result = await c.env.DB.prepare(`
      INSERT INTO billable_events (
        patient_id, assessment_id, exercise_session_id, cpt_code_id,
        service_date, duration_minutes, clinical_note, medical_necessity,
        provider_id, billing_status
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
    `).bind(
      event.patient_id, event.assessment_id, event.exercise_session_id,
      event.cpt_code_id, event.service_date, event.duration_minutes,
      event.clinical_note, event.medical_necessity, clinician?.id
    ).run()

    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

billing.get('/events/patient/:patientId', authMiddleware, async (c) => {
  try {
    const patientId = c.req.param('patientId')
    const { results } = (await c.env.DB.prepare(`
      SELECT be.*, bc.cpt_code, bc.code_description
      FROM billable_events be
      JOIN billing_codes bc ON be.cpt_code_id = bc.id
      WHERE be.patient_id = ?
      ORDER BY be.service_date DESC
    `).bind(patientId).all()) as any
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

billing.get('/events', authMiddleware, async (c) => {
  try {
    const { results } = (await c.env.DB.prepare(`
      SELECT be.*, bc.cpt_code, bc.code_description, p.first_name, p.last_name
      FROM billable_events be
      JOIN billing_codes bc ON be.cpt_code_id = bc.id
      JOIN patients p ON be.patient_id = p.id
      ORDER BY be.service_date DESC
    `).all()) as any
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

export default billing
