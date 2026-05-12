// Assessment Routes - Full Database Integration with XSS Protection

import { Hono } from 'hono'
import type { Bindings, Variables } from '../types.js'
import { authMiddleware } from '../middleware/auth.js'
import { getPool } from '../database.js'
import { sanitizeString, sanitizeJsonContent } from '../utils/security.js'

const assessments = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Apply auth middleware
assessments.use('*', authMiddleware)

// GET /api/assessments - List assessments with pagination
assessments.get('/', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const url = new URL(c.req.url)
    const page = parseInt(url.searchParams.get('page') || '1')
    const limit = parseInt(url.searchParams.get('limit') || '20')
    const patientId = url.searchParams.get('patient_id')
    const status = url.searchParams.get('status')
    const offset = (page - 1) * limit

    let whereClause = 'WHERE 1=1'
    const params: any[] = []

    if (patientId) {
      params.push(parseInt(patientId))
      whereClause += ` AND a.patient_id = $${params.length}`
    }

    if (status) {
      params.push(status)
      whereClause += ` AND a.assessment_status = $${params.length}`
    }

    // Get total count
    const countResult = await pool.query(
      `SELECT COUNT(*) as total FROM assessments a ${whereClause}`,
      params
    )
    const total = parseInt(countResult.rows[0].total)

    // Get assessments
    params.push(limit, offset)
    const result = await pool.query(
      `SELECT a.*, 
              p.first_name as patient_first_name, 
              p.last_name as patient_last_name,
              u.first_name as clinician_first_name,
              u.last_name as clinician_last_name
       FROM assessments a
       JOIN patients p ON a.patient_id = p.id
       LEFT JOIN users u ON a.clinician_id = u.id
       ${whereClause}
       ORDER BY a.session_date DESC
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
    console.error('[ASSESSMENTS] List error:', error.message)
    return c.json({ success: false, error: 'Failed to load assessments' }, 500)
  }
})

// GET /api/assessments/:id - Get single assessment
assessments.get('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid assessment ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT a.*, 
              p.first_name as patient_first_name, 
              p.last_name as patient_last_name,
              u.first_name as clinician_first_name,
              u.last_name as clinician_last_name
       FROM assessments a
       JOIN patients p ON a.patient_id = p.id
       LEFT JOIN users u ON a.clinician_id = u.id
       WHERE a.id = $1`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Assessment not found' }, 404)
    }

    // Get associated tests
    const testsResult = await pool.query(
      `SELECT * FROM movement_tests WHERE assessment_id = $1 ORDER BY test_order`,
      [id]
    )

    return c.json({
      success: true,
      data: {
        ...result.rows[0],
        tests: testsResult.rows
      }
    })

  } catch (error: any) {
    console.error('[ASSESSMENTS] Get error:', error.message)
    return c.json({ success: false, error: 'Failed to load assessment' }, 500)
  }
})

// POST /api/assessments - Create new assessment
assessments.post('/', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const rawData = await c.req.json()
    const clinician = c.get('clinician')

    // Validate required fields
    if (!rawData.patient_id) {
      return c.json({ success: false, error: 'patient_id is required' }, 400)
    }

    // Verify patient exists
    const patient = await pool.query(
      'SELECT id FROM patients WHERE id = $1',
      [rawData.patient_id]
    )

    if (patient.rows.length === 0) {
      return c.json({ success: false, error: 'Patient not found' }, 404)
    }

    // SANITIZE user inputs to prevent XSS
    const data = {
      patient_id: rawData.patient_id,
      clinician_id: clinician?.id || rawData.clinician_id,
      assessment_type: sanitizeString(rawData.assessment_type) || 'initial',
      subjective_findings: sanitizeString(rawData.subjective_findings),
      objective_findings: sanitizeString(rawData.objective_findings),
      assessment_summary: sanitizeString(rawData.assessment_summary),
      plan: sanitizeString(rawData.plan),
      femto_mega_connected: rawData.femto_mega_connected || false,
      video_recorded: rawData.video_recorded || false
    }

    const result = await pool.query(
      `INSERT INTO assessments (
        patient_id, clinician_id, assessment_type, assessment_status,
        session_date, subjective_findings, objective_findings,
        assessment_summary, plan, femto_mega_connected, video_recorded,
        created_at, updated_at
      ) VALUES ($1, $2, $3, $4, NOW(), $5, $6, $7, $8, $9, $10, NOW(), NOW())
      RETURNING *`,
      [
        data.patient_id,
        data.clinician_id,
        data.assessment_type,
        'in_progress',
        data.subjective_findings,
        data.objective_findings,
        data.assessment_summary,
        data.plan,
        data.femto_mega_connected,
        data.video_recorded
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Assessment created successfully'
    })

  } catch (error: any) {
    console.error('[ASSESSMENTS] Create error:', error.message)
    return c.json({ success: false, error: 'Failed to create assessment' }, 500)
  }
})

// PUT /api/assessments/:id - Update assessment
assessments.put('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid assessment ID' }, 400)
  }

  try {
    const rawUpdates = await c.req.json()

    // Fields that need sanitization (text fields)
    const textFields = [
      'assessment_type', 'assessment_status',
      'subjective_findings', 'objective_findings', 
      'assessment_summary', 'plan', 'video_url', 'billing_status'
    ]
    
    // Fields that are numeric/boolean (no sanitization needed)
    const numericFields = [
      'duration_minutes', 'overall_score', 'mobility_score', 
      'stability_score', 'movement_pattern_score'
    ]
    
    const booleanFields = ['femto_mega_connected', 'video_recorded']

    const setFields: string[] = []
    const values: any[] = []

    // Sanitize text fields
    for (const field of textFields) {
      if (rawUpdates[field] !== undefined) {
        setFields.push(`${field} = $${values.length + 1}`)
        values.push(sanitizeString(rawUpdates[field]))
      }
    }
    
    // Handle numeric fields
    for (const field of numericFields) {
      if (rawUpdates[field] !== undefined) {
        setFields.push(`${field} = $${values.length + 1}`)
        values.push(rawUpdates[field])
      }
    }
    
    // Handle boolean fields
    for (const field of booleanFields) {
      if (rawUpdates[field] !== undefined) {
        setFields.push(`${field} = $${values.length + 1}`)
        values.push(rawUpdates[field])
      }
    }

    if (setFields.length === 0) {
      return c.json({ success: false, error: 'No valid fields to update' }, 400)
    }

    setFields.push('updated_at = NOW()')
    values.push(id)

    const result = await pool.query(
      `UPDATE assessments SET ${setFields.join(', ')} WHERE id = $${values.length} RETURNING *`,
      values
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Assessment not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Assessment updated successfully'
    })

  } catch (error: any) {
    console.error('[ASSESSMENTS] Update error:', error.message)
    return c.json({ success: false, error: 'Failed to update assessment' }, 500)
  }
})

// DELETE /api/assessments/:id - Delete assessment
assessments.delete('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid assessment ID' }, 400)
  }

  try {
    const result = await pool.query(
      'DELETE FROM assessments WHERE id = $1 RETURNING id',
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Assessment not found' }, 404)
    }

    return c.json({
      success: true,
      message: 'Assessment deleted successfully'
    })

  } catch (error: any) {
    console.error('[ASSESSMENTS] Delete error:', error.message)
    return c.json({ success: false, error: 'Failed to delete assessment' }, 500)
  }
})

// POST /api/assessments/:id/tests - Add movement test to assessment
assessments.post('/:id/tests', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const assessmentId = parseInt(c.req.param('id'))
  if (isNaN(assessmentId)) {
    return c.json({ success: false, error: 'Invalid assessment ID' }, 400)
  }

  try {
    const rawData = await c.req.json()

    // SANITIZE test data
    const data = {
      test_name: sanitizeString(rawData.test_name),
      test_category: sanitizeString(rawData.test_category) || 'functional',
      test_order: rawData.test_order || 1,
      instructions: sanitizeString(rawData.instructions) || '',
      demo_video_url: sanitizeString(rawData.demo_video_url),
      test_status: sanitizeString(rawData.test_status) || 'pending'
    }

    const result = await pool.query(
      `INSERT INTO movement_tests (
        assessment_id, test_name, test_category, test_order,
        instructions, demo_video_url, test_status, created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
      RETURNING *`,
      [
        assessmentId,
        data.test_name,
        data.test_category,
        data.test_order,
        data.instructions,
        data.demo_video_url,
        data.test_status
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Test created successfully'
    })

  } catch (error: any) {
    console.error('[ASSESSMENTS] Create test error:', error.message)
    return c.json({ success: false, error: 'Failed to create test' }, 500)
  }
})

// PUT /api/assessments/:id/complete - Complete assessment
assessments.put('/:id/complete', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid assessment ID' }, 400)
  }

  try {
    const data = await c.req.json()

    const result = await pool.query(
      `UPDATE assessments SET 
        assessment_status = 'completed',
        overall_score = $2,
        mobility_score = $3,
        stability_score = $4,
        updated_at = NOW()
       WHERE id = $1
       RETURNING *`,
      [
        id,
        data.overall_score || null,
        data.mobility_score || null,
        data.stability_score || null
      ]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Assessment not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Assessment completed'
    })

  } catch (error: any) {
    console.error('[ASSESSMENTS] Complete error:', error.message)
    return c.json({ success: false, error: 'Failed to complete assessment' }, 500)
  }
})

// POST /api/assessments/:id/generate-note - Generate SOAP note
assessments.post('/:id/generate-note', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid assessment ID' }, 400)
  }

  try {
    // Get assessment with patient info
    const assessmentResult = await pool.query(
      `SELECT a.*, p.first_name, p.last_name, p.date_of_birth, p.gender
       FROM assessments a
       JOIN patients p ON a.patient_id = p.id
       WHERE a.id = $1`,
      [id]
    )

    if (assessmentResult.rows.length === 0) {
      return c.json({ success: false, error: 'Assessment not found' }, 404)
    }

    const assessment = assessmentResult.rows[0]

    // Get movement analysis
    const analysisResult = await pool.query(
      `SELECT ma.*, mt.test_name
       FROM movement_analysis ma
       JOIN movement_tests mt ON ma.test_id = mt.id
       WHERE mt.assessment_id = $1`,
      [id]
    )

    // Generate SOAP note
    const soapNote = {
      subjective: assessment.subjective_findings || `Patient presents for ${assessment.assessment_type} assessment.`,
      objective: assessment.objective_findings || generateObjectiveFromAnalysis(analysisResult.rows),
      assessment: assessment.assessment_summary || generateAssessmentFromScores(assessment),
      plan: assessment.plan || generatePlanFromAssessment(assessment, analysisResult.rows)
    }

    // Save generated note
    await pool.query(
      `UPDATE assessments SET 
        subjective_findings = $2,
        objective_findings = $3,
        assessment_summary = $4,
        plan = $5,
        updated_at = NOW()
       WHERE id = $1`,
      [id, soapNote.subjective, soapNote.objective, soapNote.assessment, soapNote.plan]
    )

    return c.json({
      success: true,
      data: soapNote,
      message: 'SOAP note generated successfully'
    })

  } catch (error: any) {
    console.error('[ASSESSMENTS] Generate note error:', error.message)
    return c.json({ success: false, error: 'Failed to generate note' }, 500)
  }
})

// Helper functions for SOAP note generation
function generateObjectiveFromAnalysis(analysisRows: any[]): string {
  if (analysisRows.length === 0) return 'Objective assessment data pending.'
  
  const parts = ['Movement analysis performed:']
  analysisRows.forEach(row => {
    parts.push(`- ${row.test_name}: Quality score ${row.movement_quality_score}/100`)
    if (row.compensation_detected) {
      parts.push('  Compensations detected')
    }
  })
  return parts.join('\n')
}

function generateAssessmentFromScores(assessment: any): string {
  const scores = []
  if (assessment.overall_score) scores.push(`Overall: ${assessment.overall_score}/100`)
  if (assessment.mobility_score) scores.push(`Mobility: ${assessment.mobility_score}/100`)
  if (assessment.stability_score) scores.push(`Stability: ${assessment.stability_score}/100`)
  
  if (scores.length === 0) return 'Assessment in progress.'
  
  return `Movement assessment completed. ${scores.join(', ')}.`
}

function generatePlanFromAssessment(assessment: any, analysisRows: any[]): string {
  const plans = ['Home exercise program prescribed.', 'Follow-up assessment in 2-4 weeks.']
  
  if (analysisRows.some(r => r.compensation_detected)) {
    plans.push('Focus on movement quality and compensation correction.')
  }
  
  return plans.join(' ')
}

export default assessments
