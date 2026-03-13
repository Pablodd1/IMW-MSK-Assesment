import { Hono } from 'hono'
import type { Bindings, Assessment } from '../types'
import { authMiddleware } from '../middleware/auth'
import { validate } from '../middleware/validation'
import { assessmentCreateSchema } from '../middleware/validation'

const assessments = new Hono<{ Bindings: Bindings }>()

assessments.use('/*', authMiddleware)

assessments.post('/', validate(assessmentCreateSchema), async (c) => {
  try {
    const assessment = c.get('validatedData')
    const clinician = c.get('clinician')

    const result = await c.env.DB.prepare(`
      INSERT INTO assessments (
        patient_id, clinician_id, assessment_type, status
      ) VALUES (?, ?, ?, ?)
    `).bind(
      assessment.patient_id, clinician?.id || 1,
      assessment.assessment_type, 'in_progress'
    ).run()

    return c.json({ success: true, data: { id: result.meta.last_row_id, ...assessment } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

assessments.get('/', async (c) => {
  try {
    const { results } = (await c.env.DB.prepare(`
      SELECT a.*, p.first_name, p.last_name 
      FROM assessments a
      JOIN patients p ON a.patient_id = p.id
      ORDER BY a.assessment_date DESC
    `).all()) as any
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

assessments.get('/:id', async (c) => {
  try {
    const id = c.req.param('id')
    const assessment = await c.env.DB.prepare(`
      SELECT a.*, p.first_name, p.last_name, p.date_of_birth
      FROM assessments a
      JOIN patients p ON a.patient_id = p.id
      WHERE a.id = ?
    `).bind(id).first()

    if (!assessment) {
      return c.json({ success: false, error: 'Assessment not found' }, 404)
    }

    return c.json({ success: true, data: assessment })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

assessments.put('/:id', async (c) => {
  try {
    const id = c.req.param('id')
    const assessment = await c.req.json<Partial<Assessment>>()

    const updates: string[] = []
    const values: any[] = []

    if (assessment.assessment_status) {
      updates.push('status = ?')
      values.push(assessment.assessment_status)
    }
    if (assessment.overall_score !== undefined) {
      updates.push('overall_score = ?')
      values.push(assessment.overall_score)
    }
    if (assessment.subjective_findings !== undefined) {
      updates.push('subjective_findings = ?')
      values.push(assessment.subjective_findings)
    }
    if (assessment.objective_findings !== undefined) {
      updates.push('objective_findings = ?')
      values.push(assessment.objective_findings)
    }
    if (assessment.assessment_summary !== undefined) {
      updates.push('assessment_summary = ?')
      values.push(assessment.assessment_summary)
    }
    if (assessment.plan !== undefined) {
      updates.push('plan = ?')
      values.push(assessment.plan)
    }

    if (updates.length > 0) {
      updates.push('updated_at = CURRENT_TIMESTAMP')
      values.push(id)

      await c.env.DB.prepare(`
        UPDATE assessments SET ${updates.join(', ')} WHERE id = ?
      `).bind(...values).run()
    }

    return c.json({ success: true })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

assessments.post('/:id/complete', async (c) => {
  try {
    const id = c.req.param('id')
    const { overall_score, mobility_score, stability_score, movement_pattern_score } = await c.req.json()

    await c.env.DB.prepare(`
      UPDATE assessments SET
        status = 'completed',
        overall_score = ?,
        mobility_score = ?,
        stability_score = ?,
        movement_pattern_score = ?,
        updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `).bind(overall_score, mobility_score, stability_score, movement_pattern_score, id).run()

    return c.json({ success: true })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

export default assessments
