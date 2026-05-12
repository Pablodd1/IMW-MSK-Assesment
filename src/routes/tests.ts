// Movement Test Analysis Routes

import { Hono } from 'hono'
import type { Bindings, Variables } from '../types.js'
import { authMiddleware } from '../middleware/auth.js'
import { getPool } from '../database.js'
import { performBiomechanicalAnalysis } from '../utils/biomechanics.js'

const tests = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Apply auth middleware
tests.use('*', authMiddleware)

// GET /api/tests/:id - Get test details
tests.get('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid test ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT mt.*, a.patient_id, a.assessment_type
       FROM movement_tests mt
       JOIN assessments a ON mt.assessment_id = a.id
       WHERE mt.id = $1`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Test not found' }, 404)
    }

    // Get analysis if available
    const analysisResult = await pool.query(
      'SELECT * FROM movement_analysis WHERE test_id = $1',
      [id]
    )

    return c.json({
      success: true,
      data: {
        ...result.rows[0],
        analysis: analysisResult.rows[0] || null
      }
    })

  } catch (error: any) {
    console.error('[TESTS] Get error:', error.message)
    return c.json({ success: false, error: 'Failed to load test' }, 500)
  }
})

// PUT /api/tests/:id/analyze - Analyze movement data
tests.put('/:id/analyze', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid test ID' }, 400)
  }

  try {
    const data = await c.req.json()
    
    if (!data.skeleton_data) {
      return c.json({ success: false, error: 'skeleton_data is required' }, 400)
    }

    // Get test info
    const testResult = await pool.query(
      `SELECT mt.*, a.patient_id 
       FROM movement_tests mt
       JOIN assessments a ON mt.assessment_id = a.id
       WHERE mt.id = $1`,
      [id]
    )

    if (testResult.rows.length === 0) {
      return c.json({ success: false, error: 'Test not found' }, 404)
    }

    const test = testResult.rows[0]

    // Perform biomechanical analysis
    const analysis = performBiomechanicalAnalysis(data.skeleton_data)

    // Store analysis results
    const analysisResult = await pool.query(
      `INSERT INTO movement_analysis (
        test_id, joint_angles, rom_measurements, left_right_asymmetry,
        movement_quality_score, stability_score, compensation_detected,
        deficiencies, ai_recommendations, ai_analysis_text, analyzed_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
      ON CONFLICT (test_id) DO UPDATE SET
        joint_angles = EXCLUDED.joint_angles,
        movement_quality_score = EXCLUDED.movement_quality_score,
        stability_score = EXCLUDED.stability_score,
        compensation_detected = EXCLUDED.compensation_detected,
        deficiencies = EXCLUDED.deficiencies,
        ai_recommendations = EXCLUDED.ai_recommendations,
        analyzed_at = EXCLUDED.analyzed_at
      RETURNING *`,
      [
        id,
        JSON.stringify(analysis.joint_angles || []),
        JSON.stringify(analysis.rom_measurements || {}),
        JSON.stringify(analysis.left_right_asymmetry || {}),
        analysis.movement_quality_score || 0,
        analysis.stability_score || 0,
        analysis.compensation_detected || false,
        JSON.stringify(analysis.deficiencies || []),
        JSON.stringify(analysis.recommendations || []),
        analysis.analysis_text || null
      ]
    )

    // Update test status
    await pool.query(
      `UPDATE movement_tests SET 
        test_status = 'completed',
        completed_at = NOW(),
        skeleton_data = $2,
        movement_quality_score = $3
       WHERE id = $1`,
      [id, JSON.stringify(data.skeleton_data), analysis.movement_quality_score || 0]
    )

    // Update assessment with video reference if provided
    if (data.video_id) {
      await pool.query(
        `UPDATE assessments SET 
          video_recorded = true,
          video_url = $2
         WHERE id = $1`,
        [test.assessment_id, `/api/videos/${data.video_id}`]
      )
    }

    return c.json({
      success: true,
      data: {
        analysis: analysis,
        stored_analysis: analysisResult.rows[0]
      },
      message: 'Movement analysis completed'
    })

  } catch (error: any) {
    console.error('[TESTS] Analyze error:', error.message)
    return c.json({ success: false, error: 'Failed to analyze movement' }, 500)
  }
})

// GET /api/tests/:id/results - Get test results
tests.get('/:id/results', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid test ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT ma.*, mt.test_name, mt.test_category, mt.instructions
       FROM movement_analysis ma
       JOIN movement_tests mt ON ma.test_id = mt.id
       WHERE ma.test_id = $1`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Results not found' }, 404)
    }

    return c.json({
      success: true,
      data: result.rows[0]
    })

  } catch (error: any) {
    console.error('[TESTS] Results error:', error.message)
    return c.json({ success: false, error: 'Failed to load results' }, 500)
  }
})

export default tests
