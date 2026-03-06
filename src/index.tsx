import { validateEnv } from './env';
import { db } from './db';
import { Hono } from 'hono'
import type { Context } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-pages'
import { createMiddleware } from 'hono/factory'
import type {
  Bindings, Variables, Patient, Assessment, Exercise,
  PrescribedExercise, ExerciseSession, SkeletonData
} from './types'
import { performBiomechanicalAnalysis } from './utils/biomechanics'
import { queryExerciseKnowledge } from './utils/rag'
import { secureAuth, hashPassword, verifyPassword, generateToken, verifyToken } from './middleware/auth'
import { validate, patientCreateSchema, assessmentCreateSchema, prescriptionSchema, clinicianRegisterSchema } from './middleware/validation'
import { auditLog, phiAudit, securityHeaders, safeLog } from './middleware/hipaa'
import { apiRateLimit, authRateLimit, writeRateLimit, analysisRateLimit } from './middleware/rateLimit'
import { errorHandler } from './middleware/error'
import { mockD1 } from './db'

validateEnv();

const app = new Hono<{ Bindings: Bindings, Variables: Variables }>()

// Auth Middleware - Protects routes
export const authMiddleware = createMiddleware<{ Bindings: Bindings, Variables: Variables }>(async (c, next) => {
  const authHeader = c.req.header('Authorization')
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return c.json({ success: false, error: 'Unauthorized' }, 401)
  }

  const token = authHeader.split(' ')[1]
  const secret = process.env.JWT_SECRET || process.env.AUTH_SECRET
  if (!secret) {
    safeLog.error('JWT_SECRET or AUTH_SECRET not set')
    return c.json({ success: false, error: 'Server configuration error' }, 500)
  }

  try {
    const payload = await verifyToken(token, secret)
    if (!payload) {
      return c.json({ success: false, error: 'Invalid token' }, 401)
    }
    c.set('clinician', payload) // Store clinician payload in context
    await next()
  } catch (e) {
    safeLog.warn('Token verification failed', e as Error)
    return c.json({ success: false, error: 'Invalid or expired token' }, 401)
  }
})

// ============================================================================
// SECURITY MIDDLEWARE
// ============================================================================

// Security headers for all responses
app.use('*', securityHeaders)

// Global error handler
// Removed invalid require

// Global error handler
if (typeof app.onError === 'function') {
  app.onError((err: Error, c: Context<{ Bindings: Bindings, Variables: Variables }>) => {
    safeLog.error('Unhandled server error', err, {
      path: c.req.path,
      method: c.req.method
    })
    return c.json({ success: false, error: 'Internal server error' }, 500)
  })
}

// Not-found handler
if (typeof (app as any).notFound === 'function') {
  app.notFound((c: Context<{ Bindings: Bindings, Variables: Variables }>) => {
    return c.json({ success: false, error: 'Endpoint not found', code: 'NOT_FOUND' }, 404)
  })
}

// Production CORS configuration
const corsConfig = {
  origin: (origin: string) => {
    const allowedOrigins = [
      'https://physiomotion.com',
      'https://www.physiomotion.com',
      'https://app.physiomotion.com'
    ]
    // Allow no-origin requests (mobile apps, Postman, server-to-server)
    if (!origin) return allowedOrigins[0]
    if (allowedOrigins.includes(origin) || origin.startsWith('http://localhost')) {
      return origin
    }
    return allowedOrigins[0]
  },
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  credentials: true,
  maxAge: 86400
}

app.use('/api/*', cors(corsConfig))
app.use(errorHandler)

// Health checks
app.get('/api/healthz', async (c: Context<{ Bindings: Bindings, Variables: Variables }>) => {
  let dbAlive = true
  try {
    if (mockD1) {
      await mockD1.prepare('SELECT 1').first()
    }
  } catch (e) {
    dbAlive = false
  }
  return c.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    db: dbAlive ? 'connected' : 'unavailable',
    environment: process.env.NODE_ENV || 'production',
    version: '1.0.0'
  })
})

app.get('/api/ready', async (c: Context<{ Bindings: Bindings, Variables: Variables }>) => {
  // Lightweight readiness: DB + basic services
  let ready = true
  try {
    if (mockD1) {
      await mockD1.prepare('SELECT 1').first()
    }
  } catch {
    ready = false
  }
  return c.json({ ready })
})

// Token refresh endpoint (very small flow)
app.post('/api/auth/refresh', async (c: Context<{ Bindings: Bindings, Variables: Variables }>) => {
  try {
    const cookieHeader = c.req.header('Cookie') || ''
    const match = cookieHeader.match(/refresh_token=([^;]+)/)
    const refreshToken = match ? match[1] : null
    const secret = process.env.JWT_SECRET || process.env.AUTH_SECRET
    if (!refreshToken || !secret) {
      return c.json({ success: false, error: 'Unauthorized' }, 401)
    }
    const payload = await verifyToken(refreshToken, secret)
    if (!payload) {
      return c.json({ success: false, error: 'Invalid token' }, 401)
    }
    const newToken = await generateToken({ id: payload.id, email: payload.email, role: payload.role }, secret)
    // Return new token in body; optionally set as cookie
    return c.json({ success: true, token: newToken })
  } catch (e) {
    safeLog.error('Refresh token failed', e as Error)
    return c.json({ success: false, error: 'Unauthorized' }, 401)
  }
})

// Serve static files
app.use('/static/*', serveStatic())

// ============================================================================
// CACHING
// ============================================================================
let exercisesCache: any[] | null = null
let lastExerciseCacheTime = 0
const EXERCISE_CACHE_TTL = 3600 * 1000 // 1 hour

let billingCodesCache: any[] | null = null
let lastBillingCacheTime = 0
const BILLING_CACHE_TTL = 3600 * 1000 // 1 hour

// ============================================================================
// AUTHENTICATION API
// ============================================================================

// Register new clinician (rate limited)
app.post('/api/auth/register', authRateLimit, validate(clinicianRegisterSchema), async (c) => {
  try {
    const data = c.get('validatedData') as typeof clinicianRegisterSchema._type

    // Check if email already exists
    const existing = await mockD1.prepare(`
      SELECT id FROM clinicians WHERE email = ?
  `).bind(data.email).first()

    if (existing) {
      return c.json({ success: false, error: 'Email already registered' }, 400)
    }

    // Generate secure salt and hash password
    const salt = crypto.randomUUID()
    const passwordHash = await hashPassword(data.password, salt)

    const result = await mockD1.prepare(`
      INSERT INTO clinicians(
    email, password_hash, salt, first_name, last_name, title,
    license_number, license_state, npi_number, phone, clinic_name,
    role, active
  ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'clinician', 1)
    `).bind(
      data.email, passwordHash, salt, data.first_name, data.last_name, data.title,
      data.license_number, data.license_state, data.npi_number,
      data.phone, data.clinic_name
    ).run()

    safeLog.info('New clinician registered', { clinicianId: result.meta.last_row_id, email: data.email })

    return c.json({
      success: true,
      data: { id: result.meta.last_row_id }
    })
  } catch (error: any) {
    safeLog.error('Registration failed', error as Error)
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Login (rate limited)
app.post('/api/auth/login', authRateLimit, async (c) => {
  try {
    const { email, password } = await c.req.json()

    const clinician = await mockD1.prepare(`
SELECT * FROM clinicians WHERE email = ? AND active = true
  `).bind(email).first() as any

    if (!clinician) {
      safeLog.warn('Login failed - user not found', { email })
      return c.json({ success: false, error: 'Invalid email or password' }, 401)
    }

    // Verify password with stored salt
    const salt = clinician.salt || 'default-salt'
    const isValid = await verifyPassword(password, salt, clinician.password_hash)

    if (!isValid) {
      safeLog.warn('Login failed - invalid password', { clinicianId: clinician.id, email })
      return c.json({ success: false, error: 'Invalid email or password' }, 401)
    }

    // Generate JWT token
    const token = await generateToken(
      { id: clinician.id, email: clinician.email, role: clinician.role },
      process.env.JWT_SECRET || process.env.AUTH_SECRET || 'default-secret-change-me'
    )

    // Update last login and activity
    await mockD1.prepare(`
      UPDATE clinicians SET last_login = CURRENT_TIMESTAMP, last_activity = CURRENT_TIMESTAMP WHERE id = ?
  `).bind(clinician.id).run()

    safeLog.info('Login successful', { clinicianId: clinician.id, email })

    // Return user data (excluding password) and token
    const { password_hash, salt: _, ...userData } = clinician

    return c.json({
      success: true,
      data: {
        ...userData,
        token
      }
    })
  } catch (error: any) {
    safeLog.error('Login error', error as Error)
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get current clinician profile
app.get('/api/auth/profile/:id', authMiddleware, async (c) => {
  try {
    const id = c.req.param('id')
    const clinicianPayload = c.get('clinician')

    if (clinicianPayload.id !== parseInt(id)) {
      return c.json({ success: false, error: 'Unauthorized access to profile' }, 403)
    }

    const clinician = await mockD1.prepare(`
      SELECT id, email, first_name, last_name, title, license_number,
  license_state, npi_number, phone, clinic_name, role, active,
  created_at, last_login
      FROM clinicians WHERE id = ?
  `).bind(id).first()

    if (!clinician) {
      return c.json({ success: false, error: 'Clinician not found' }, 404)
    }

    return c.json({ success: true, data: clinician })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// PATIENT MANAGEMENT API (Secured with Auth, Validation, Audit)
// ============================================================================

// Create new patient (requires auth, validated, audited)
app.post('/api/patients',
  authMiddleware,
  validate(patientCreateSchema),
  writeRateLimit,
  auditLog('PATIENT_CREATE', 'patient'),
  async (c) => {
    try {
      const validatedData = c.get('validatedData') as typeof patientCreateSchema._type
      const clinician = c.get('clinician')

      const result = await mockD1.prepare(`
        INSERT INTO patients(
    first_name, last_name, date_of_birth, gender, email, phone,
    emergency_contact_name, emergency_contact_phone,
    address_line1, city, state, zip_code,
    height_cm, weight_kg, insurance_provider, created_by_clinician_id
  ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
        validatedData.first_name, validatedData.last_name, validatedData.date_of_birth,
        validatedData.gender, validatedData.email, validatedData.phone,
        validatedData.emergency_contact_name, validatedData.emergency_contact_phone,
        validatedData.address_line1, validatedData.city, validatedData.state, validatedData.zip_code,
        validatedData.height_cm, validatedData.weight_kg, validatedData.insurance_provider,
        clinician.id
      ).run()

      safeLog.info('Patient created', { patientId: result.meta.last_row_id, createdBy: clinician.id })

      return c.json({
        success: true,
        data: { id: result.meta.last_row_id, ...validatedData }
      })
    } catch (error: any) {
      safeLog.error('Patient creation failed', error as Error)
      return c.json({ success: false, error: error.message }, 500)
    }
  }
)

// Get all patients (requires auth, audited)
app.get('/api/patients',
  authMiddleware,
  apiRateLimit,
  auditLog('PATIENT_VIEW', 'patient'),
  async (c) => {
    try {
      const clinician = c.get('clinician')
      const { results } = await mockD1.prepare(`
        SELECT id, first_name, last_name, date_of_birth, gender, email, phone,
  city, state, created_at, patient_status
        FROM patients WHERE created_by_clinician_id = ? ORDER BY created_at DESC
  `).bind(clinician.id).all()

      return c.json({ success: true, data: results })
    } catch (error: any) {
      return c.json({ success: false, error: error.message }, 500)
    }
  }
)

// Get patient by ID (requires auth, PHI audit)
app.get('/api/patients/:id',
  authMiddleware,
  apiRateLimit,
  phiAudit('VIEW', 'patient'),
  async (c) => {
    try {
      const id = c.req.param('id')
      const clinician = c.get('clinician')
      const patient = await mockD1.prepare(`
SELECT * FROM patients WHERE id = ? AND created_by_clinician_id = ?
  `).bind(id, clinician.id).first()

      if (!patient) {
        return c.json({ success: false, error: 'Patient not found or unauthorized' }, 404)
      }

      return c.json({ success: true, data: patient })
    } catch (error: any) {
      return c.json({ success: false, error: error.message }, 500)
    }
  })

// ============================================================================
// MEDICAL HISTORY API
// ============================================================================

app.post('/api/patients/:id/medical-history', authMiddleware, async (c) => {
  try {
    const patientId = c.req.param('id')
    const history = await c.req.json()
    const clinician = c.get('clinician')

    // Verify patient belongs to clinician
    const patientCheck = await mockD1.prepare(`
      SELECT id FROM patients WHERE id = ? AND created_by_clinician_id = ?
  `).bind(patientId, clinician.id).first()
    if (!patientCheck) {
      return c.json({ success: false, error: 'Patient not found or unauthorized' }, 404)
    }

    const result = await mockD1.prepare(`
      INSERT INTO medical_history(
    patient_id, surgery_type, surgery_date, conditions, medications, allergies,
    current_pain_level, pain_location, activity_level, treatment_goals
  ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

// ============================================================================
// ASSESSMENT API
// ============================================================================

// Create new assessment
app.post('/api/assessments', authMiddleware, validate(assessmentCreateSchema), async (c) => {
  try {
    const assessment = c.get('validatedData') as typeof assessmentCreateSchema._type
    const clinician = c.get('clinician')

    // Verify patient belongs to clinician
    const patientCheck = await mockD1.prepare(`
      SELECT id FROM patients WHERE id = ? AND created_by_clinician_id = ?
  `).bind(assessment.patient_id, clinician.id).first()
    if (!patientCheck) {
      return c.json({ success: false, error: 'Patient not found or unauthorized' }, 404)
    }

    const result = await mockD1.prepare(`
      INSERT INTO assessments(
    patient_id, clinician_id, assessment_type, status
  ) VALUES(?, ?, ?, ?)
    `).bind(
      assessment.patient_id, clinician.id,
      assessment.assessment_type, 'in_progress'
    ).run()

    return c.json({ success: true, data: { id: result.meta.last_row_id, ...assessment } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get assessments for patient
app.get('/api/patients/:id/assessments', authMiddleware, async (c) => {
  try {
    const patientId = c.req.param('id')
    const clinician = c.get('clinician')

    // Verify patient belongs to clinician
    const patientCheck = await mockD1.prepare(`
      SELECT id FROM patients WHERE id = ? AND created_by_clinician_id = ?
  `).bind(patientId, clinician.id).first()
    if (!patientCheck) {
      return c.json({ success: false, error: 'Patient not found or unauthorized' }, 404)
    }

    const { results } = await mockD1.prepare(`
      SELECT * FROM assessments WHERE patient_id = ? AND clinician_id = ? ORDER BY assessment_date DESC
  `).bind(patientId, clinician.id).all()

    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get all assessments
app.get('/api/assessments', authMiddleware, async (c) => {
  try {
    const clinician = c.get('clinician')
    const { results } = await mockD1.prepare(`
      SELECT a.*, p.first_name, p.last_name
      FROM assessments a
      JOIN patients p ON a.patient_id = p.id
      WHERE a.clinician_id = ? AND p.created_by_clinician_id = ?
  ORDER BY a.assessment_date DESC
    `).bind(clinician.id, clinician.id).all()

    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get assessment by ID
app.get('/api/assessments/:id', authMiddleware, async (c) => {
  try {
    const id = c.req.param('id')
    const clinician = c.get('clinician')
    const assessment = await mockD1.prepare(`
      SELECT a.*, p.first_name, p.last_name, p.date_of_birth
      FROM assessments a
      JOIN patients p ON a.patient_id = p.id
      WHERE a.id = ? AND a.clinician_id = ? AND p.created_by_clinician_id = ?
  `).bind(id, clinician.id, clinician.id).first()

    if (!assessment) {
      return c.json({ success: false, error: 'Assessment not found or unauthorized' }, 404)
    }

    return c.json({ success: true, data: assessment })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// MOVEMENT TEST API
// ============================================================================

// Create movement test
app.post('/api/assessments/:id/tests', authMiddleware, async (c) => {
  try {
    const assessmentId = c.req.param('id')
    const test = await c.req.json()
    const clinician = c.get('clinician')

    // Verify assessment belongs to clinician
    const assessmentCheck = await mockD1.prepare(`
      SELECT a.id FROM assessments a JOIN patients p ON a.patient_id = p.id
      WHERE a.id = ? AND a.clinician_id = ? AND p.created_by_clinician_id = ?
  `).bind(assessmentId, clinician.id, clinician.id).first()
    if (!assessmentCheck) {
      return c.json({ success: false, error: 'Assessment not found or unauthorized' }, 404)
    }

    const result = await mockD1.prepare(`
      INSERT INTO movement_tests(
    assessment_id, test_name, test_category, test_order, instructions, status
  ) VALUES(?, ?, ?, ?, ?, ?)
    `).bind(
      assessmentId, test.test_name, test.test_category,
      test.test_order, test.instructions, 'pending'
    ).run()

    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Update test with skeleton data and perform biomechanical analysis
app.post('/api/tests/:id/analyze', authMiddleware, analysisRateLimit, async (c) => {
  try {
    const testId = c.req.param('id')
    const { skeleton_data } = await c.req.json<{ skeleton_data: SkeletonData }>()
    const clinician = c.get('clinician')

    // Verify test belongs to clinician
    const testCheck = await mockD1.prepare(`
      SELECT mt.id FROM movement_tests mt JOIN assessments a ON mt.assessment_id = a.id
      JOIN patients p ON a.patient_id = p.id
      WHERE mt.id = ? AND a.clinician_id = ? AND p.created_by_clinician_id = ?
  `).bind(testId, clinician.id, clinician.id).first()
    if (!testCheck) {
      return c.json({ success: false, error: 'Test not found or unauthorized' }, 404)
    }

    // Perform biomechanical analysis
    const analysis = performBiomechanicalAnalysis(skeleton_data)

    // Update movement test with skeleton data and analysis results
    await mockD1.prepare(`
      UPDATE movement_tests
      SET skeleton_data = ?,
  movement_quality_score = ?,
  deficiencies = ?,
  compensations_detected = ?,
  status = 'completed',
  completed_at = CURRENT_TIMESTAMP
      WHERE id = ?
  `).bind(
      JSON.stringify(skeleton_data),
      analysis.movement_quality_score,
      JSON.stringify(analysis.deficiencies),
      JSON.stringify(analysis.detected_compensations),
      testId
    ).run()

    return c.json({
      success: true,
      data: {
        movement_quality_score: analysis.movement_quality_score,
        deficiencies: analysis.deficiencies,
        compensations: analysis.detected_compensations,
        recommendations: analysis.recommendations
      }
    })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get test results
app.get('/api/tests/:id/results', authMiddleware, async (c) => {
  try {
    const testId = c.req.param('id')
    const clinician = c.get('clinician')

    // Verify test belongs to clinician
    const test = await mockD1.prepare(`
      SELECT mt.* FROM movement_tests mt JOIN assessments a ON mt.assessment_id = a.id
      JOIN patients p ON a.patient_id = p.id
      WHERE mt.id = ? AND a.clinician_id = ? AND p.created_by_clinician_id = ?
  `).bind(testId, clinician.id, clinician.id).first()

    if (!test) {
      return c.json({ success: false, error: 'Test not found or unauthorized' }, 404)
    }

    return c.json({ success: true, data: test })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get movement tests for assessment
app.get('/api/assessments/:id/tests', authMiddleware, async (c) => {
  try {
    const assessmentId = c.req.param('id')
    const clinician = c.get('clinician')

    // Verify assessment belongs to clinician
    const assessmentCheck = await mockD1.prepare(`
      SELECT a.id FROM assessments a JOIN patients p ON a.patient_id = p.id
      WHERE a.id = ? AND a.clinician_id = ? AND p.created_by_clinician_id = ?
  `).bind(assessmentId, clinician.id, clinician.id).first()
    if (!assessmentCheck) {
      return c.json({ success: false, error: 'Assessment not found or unauthorized' }, 404)
    }

    const { results } = await mockD1.prepare(`
      SELECT * FROM movement_tests WHERE assessment_id = ? ORDER BY test_order
  `).bind(assessmentId).all()

    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// EXERCISE LIBRARY API
// ============================================================================

// Get all exercises (with caching)
app.get('/api/exercises', authMiddleware, async (c) => {
  try {
    const category = c.req.query('category')

    // Use cache if available and no category filter
    const now = Date.now()
    if (!category && exercisesCache && (now - lastExerciseCacheTime < EXERCISE_CACHE_TTL)) {
      return c.json({ success: true, data: exercisesCache })
    }

    let query = 'SELECT * FROM exercises'
    const params: any[] = []

    if (category) {
      query += ' WHERE category = ?'
      params.push(category)
    }

    query += ' ORDER BY name'

    const { results } = await mockD1.prepare(query).bind(...params).all()

    // Update cache if no filter
    if (!category) {
      exercisesCache = results
      lastExerciseCacheTime = now
    }

    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// PRESCRIPTION API
// ============================================================================

// Prescribe exercises
app.post('/api/prescriptions', authMiddleware, validate(prescriptionSchema), async (c) => {
  try {
    const prescription = c.get('validatedData') as typeof prescriptionSchema._type
    const clinician = c.get('clinician')

    // Verify patient and assessment belong to clinician
    const patientAssessmentCheck = await mockD1.prepare(`
      SELECT a.id FROM assessments a JOIN patients p ON a.patient_id = p.id
      WHERE a.id = ? AND a.patient_id = ? AND a.clinician_id = ? AND p.created_by_clinician_id = ?
  `).bind(prescription.assessment_id, prescription.patient_id, clinician.id, clinician.id).first()
    if (!patientAssessmentCheck) {
      return c.json({ success: false, error: 'Patient or Assessment not found or unauthorized' }, 404)
    }

    const result = await mockD1.prepare(`
      INSERT INTO prescribed_exercises(
    patient_id, assessment_id, exercise_id, sets, repetitions,
    times_per_week, clinical_reason, target_deficiency, prescribed_by
  ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      prescription.patient_id, prescription.assessment_id,
      prescription.exercise_id, prescription.sets, prescription.repetitions,
      prescription.times_per_week, prescription.clinical_reason,
      prescription.target_deficiency, clinician.id
    ).run()

    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get patient's prescribed exercises
app.get('/api/patients/:id/prescriptions', authMiddleware, async (c) => {
  try {
    const patientId = c.req.param('id')
    const clinician = c.get('clinician')

    // Verify patient belongs to clinician
    const patientCheck = await mockD1.prepare(`
      SELECT id FROM patients WHERE id = ? AND created_by_clinician_id = ?
  `).bind(patientId, clinician.id).first()
    if (!patientCheck) {
      return c.json({ success: false, error: 'Patient not found or unauthorized' }, 404)
    }

    const { results } = await mockD1.prepare(`
      SELECT
pe.*,
  e.name as exercise_name,
  e.description,
  e.instructions,
  e.demo_video_url,
  e.difficulty
      FROM prescribed_exercises pe
      JOIN exercises e ON pe.exercise_id = e.id
      WHERE pe.patient_id = ? AND pe.status = 'active'
      ORDER BY pe.created_at DESC
  `).bind(patientId).all()

    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// EXERCISE SESSION API (Remote Patient Monitoring)
// ============================================================================

// Record export const sessionActivity = createMiddleware<{ Bindings: Bindings, Variables: Variables }>(async (c, next) => {
app.post('/api/exercise-sessions', authMiddleware, async (c) => {
  try {
    const session = await c.req.json<ExerciseSession>()
    const clinician = c.get('clinician')

    // Verify prescribed exercise belongs to clinician's patient
    const prescribedExerciseCheck = await mockD1.prepare(`
      SELECT pe.id FROM prescribed_exercises pe JOIN patients p ON pe.patient_id = p.id
      WHERE pe.id = ? AND pe.patient_id = ? AND p.created_by_clinician_id = ?
  `).bind(session.prescribed_exercise_id, session.patient_id, clinician.id).first()
    if (!prescribedExerciseCheck) {
      return c.json({ success: false, error: 'Prescribed exercise not found or unauthorized' }, 404)
    }

    const result = await mockD1.prepare(`
      INSERT INTO exercise_sessions(
    patient_id, prescribed_exercise_id, sets_completed, reps_completed,
    duration_seconds, form_quality_score, pose_accuracy_data,
    pain_level_during, difficulty_rating, completed
  ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      session.patient_id, session.prescribed_exercise_id,
      session.sets_completed, session.reps_completed, session.duration_seconds,
      session.form_quality_score, JSON.stringify(session.pose_accuracy_data), // Ensure JSON stringify
      session.pain_level_during, session.difficulty_rating, session.completed
    ).run()

    // Update compliance tracking
    if (c.executionCtx) {
      c.executionCtx.waitUntil(updateCompliancePercentage(mockD1, session.prescribed_exercise_id))
    } else {
      // Fallback for environments without executionCtx
      updateCompliancePercentage(mockD1, session.prescribed_exercise_id).catch(console.error)
    }

    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get patient exercise history
app.get('/api/patients/:id/sessions', authMiddleware, async (c) => {
  try {
    const patientId = c.req.param('id')
    const clinician = c.get('clinician')

    // Verify patient belongs to clinician
    const patientCheck = await mockD1.prepare(`
      SELECT id FROM patients WHERE id = ? AND created_by_clinician_id = ?
  `).bind(patientId, clinician.id).first()
    if (!patientCheck) {
      return c.json({ success: false, error: 'Patient not found or unauthorized' }, 404)
    }

    const { results } = await mockD1.prepare(`
      SELECT
es.*,
  e.name as exercise_name,
  pe.sets as prescribed_sets,
  pe.repetitions as prescribed_reps
      FROM exercise_sessions es
      JOIN prescribed_exercises pe ON es.prescribed_exercise_id = pe.id
      JOIN exercises e ON pe.exercise_id = e.id
      WHERE es.patient_id = ?
  ORDER BY es.session_date DESC
      LIMIT 50
  `).bind(patientId).all()

    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// BILLING API
// ============================================================================

// Get CPT codes (with caching)
app.get('/api/billing/codes', authMiddleware, async (c) => {
  try {
    const now = Date.now()
    if (billingCodesCache && (now - lastBillingCacheTime < BILLING_CACHE_TTL)) {
      return c.json({ success: true, data: billingCodesCache })
    }

    const { results } = await mockD1.prepare(`
SELECT * FROM billing_codes ORDER BY cpt_code
  `).all()

    billingCodesCache = results
    lastBillingCacheTime = now

    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Create billable event
app.post('/api/billing/events', authMiddleware, async (c) => {
  try {
    const event = await c.req.json()
    const clinician = c.get('clinician')

    // Basic check: ensure patient belongs to clinician if patient_id is provided
    if (event.patient_id) {
      const patientCheck = await mockD1.prepare(`
        SELECT id FROM patients WHERE id = ? AND created_by_clinician_id = ?
  `).bind(event.patient_id, clinician.id).first()
      if (!patientCheck) {
        return c.json({ success: false, error: 'Patient not found or unauthorized' }, 404)
      }
    }

    const result = await mockD1.prepare(`
      INSERT INTO billable_events(
    patient_id, assessment_id, exercise_session_id,
    cpt_code_id, service_date, duration_minutes,
    clinical_note, provider_id
  ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      event.patient_id, event.assessment_id, event.exercise_session_id,
      event.cpt_code_id, event.service_date, event.duration_minutes,
      event.clinical_note, clinician.id
    ).run()

    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// RAG & ANALYTICS API
// ============================================================================

app.post('/api/rag/query', authMiddleware, async (c) => {
  try {
    const { query } = await c.req.json<{ query: string }>()
    const result = await queryExerciseKnowledge(mockD1, query)
    return c.json({ success: true, data: result })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// MEDICAL NOTE GENERATION
// ============================================================================

app.post('/api/assessments/:id/generate-note', authMiddleware, async (c) => {
  try {
    const assessmentId = c.req.param('id')
    const clinician = c.get('clinician')

    // Get assessment data
    const assessment = await mockD1.prepare(`
      SELECT a.* FROM assessments a JOIN patients p ON a.patient_id = p.id
      WHERE a.id = ? AND a.clinician_id = ? AND p.created_by_clinician_id = ?
  `).bind(assessmentId, clinician.id, clinician.id).first() as any

    if (!assessment) {
      return c.json({ success: false, error: 'Assessment not found or unauthorized' }, 404)
    }

    // Get all movement tests and analyses
    const { results: tests } = await mockD1.prepare(`
      SELECT mt.*, ma.*
  FROM movement_tests mt
      LEFT JOIN movement_analysis ma ON mt.id = ma.test_id
      WHERE mt.assessment_id = ?
  `).bind(assessmentId).all()

    // Generate comprehensive medical note
    const medicalNote = generateMedicalNote(assessment, tests)

    // Try to enhance with AI insights if deficiencies exist
    let aiInsights = ""
    if (tests.length > 0 && tests[0].deficiencies && typeof tests[0].deficiencies === 'string') {
      try {
        const defs = JSON.parse(tests[0].deficiencies)
        if (defs && Array.isArray(defs) && defs.length > 0) {
          const ragResult = await queryExerciseKnowledge(mockD1, defs[0].area)
          aiInsights = ragResult.answer
        }
      } catch (e) { }
    }

    if (aiInsights) {
      medicalNote.plan += `\n\nAI CLINICAL INSIGHTS: \n${aiInsights} `
    }

    // Update assessment with generated notes
    await mockD1.prepare(`
      UPDATE assessments
      SET subjective_findings = ?,
  objective_findings = ?,
  assessment_summary = ?,
  plan = ?,
  status = 'completed'
      WHERE id = ?
  `).bind(
      medicalNote.subjective,
      medicalNote.objective,
      medicalNote.assessment,
      medicalNote.plan,
      assessmentId
    ).run()

    return c.json({ success: true, data: medicalNote })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Simple password hashing (for demo - use proper bcrypt in production)
// Removed duplicate password helper functions

async function updateCompliancePercentage(db: any, prescribedExerciseId: number) {
  // Get total sessions completed vs expected
  const result = await db.prepare(`
    SELECT COUNT(*) as completed_count
    FROM exercise_sessions
    WHERE prescribed_exercise_id = ? AND completed = true
  `).bind(prescribedExerciseId).first() as any

  const prescription = await db.prepare(`
    SELECT times_per_week, prescribed_at FROM prescribed_exercises WHERE id = ?
  `).bind(prescribedExerciseId).first() as any

  if (result && prescription) {
    const prescribedDate = new Date(prescription.prescribed_at)
    const now = new Date()

    // Don't calculate compliance for future dates
    if (prescribedDate > now) {
      return
    }

    // Calculate weeks since prescribed (minimum 1 week to avoid division by zero)
    const weeksSincePrescribed = Math.max(1, Math.floor(
      (now.getTime() - prescribedDate.getTime()) / (7 * 24 * 60 * 60 * 1000)
    ))

    const expectedSessions = prescription.times_per_week * weeksSincePrescribed

    // Calculate compliance percentage, capped at 100%
    const compliance = Math.min(100, Math.round((result.completed_count / expectedSessions) * 100))

    await db.prepare(`
      UPDATE prescribed_exercises
      SET compliance_percentage = ?,
  last_performed_at = CURRENT_TIMESTAMP
      WHERE id = ?
  `).bind(compliance, prescribedExerciseId).run()
  }
}

function generateMedicalNote(assessment: any, tests: any[]) {
  // Parse deficiencies from all tests
  const allDeficiencies: any[] = []
  const allRecommendations: string[] = []
  let avgQualityScore = 0

  for (const test of tests) {
    if (test.deficiencies && typeof test.deficiencies === 'string') {
      try {
        const deficiencies = JSON.parse(test.deficiencies)
        if (Array.isArray(deficiencies)) {
          allDeficiencies.push(...deficiencies)
        }
      } catch (e) { }
    }
    if (test.ai_recommendations && typeof test.ai_recommendations === 'string') {
      try {
        const recs = JSON.parse(test.ai_recommendations)
        if (Array.isArray(recs)) {
          allRecommendations.push(...recs)
        }
      } catch (e) { }
    }
    if (test.movement_quality_score) {
      avgQualityScore += test.movement_quality_score
    }
  }

  avgQualityScore = tests.length > 0 ? Math.round(avgQualityScore / tests.length) : 0

  // Generate SOAP note
  const subjective = `Patient presents for ${assessment.assessment_type} assessment.Willing and able to participate in functional movement screening.`

  const objective = `
FUNCTIONAL MOVEMENT ASSESSMENT RESULTS:

Overall Movement Quality Score: ${avgQualityScore}/100

Tests Completed: ${tests.length}
${tests.map(t => `- ${t.test_name}: ${t.test_status}`).join('\n')}

DEFICIENCIES IDENTIFIED:
${allDeficiencies.map((d, i) => `${i + 1}. ${d.area} (${d.severity}): ${d.description}`).join('\n\n')}

Movement Quality: ${avgQualityScore >= 80 ? 'Good' : avgQualityScore >= 60 ? 'Fair' : 'Poor'}
Compensatory patterns observed and documented in biomechanical analysis.
  `.trim()

  const assessmentSummary = `
Patient demonstrates ${avgQualityScore >= 80 ? 'good' : avgQualityScore >= 60 ? 'fair' : 'poor'} movement quality with ${allDeficiencies.length} significant deficiencies identified. 

PRIMARY FINDINGS:
${allDeficiencies.slice(0, 3).map((d, i) => `${i + 1}. ${d.area} - ${d.severity} severity`).join('\n')}

FUNCTIONAL IMPACT:
${avgQualityScore < 60 ? 'Significant functional limitations present. Patient would benefit from comprehensive therapeutic exercise program.' :
      avgQualityScore < 80 ? 'Moderate functional limitations. Targeted interventions recommended.' :
        'Minor limitations identified. Preventive exercise program appropriate.'
    }
`.trim()

  const plan = `
TREATMENT PLAN:

1. THERAPEUTIC EXERCISES: Prescribed evidence - based exercise program targeting identified deficiencies
   ${allDeficiencies.slice(0, 3).map(d => `   - Address ${d.area}`).join('\n')}

2. FREQUENCY: ${avgQualityScore < 60 ? '2-3x per week supervised therapy + daily HEP' : '1-2x per week supervised + daily HEP'}

3. DURATION: ${avgQualityScore < 60 ? '8-12 weeks' : '4-8 weeks'}

4. REMOTE MONITORING: Patient enrolled in remote therapeutic monitoring program(RTM)
  - Daily exercise compliance tracking via mobile app
    - Real - time form analysis and feedback
      - Weekly progress reports

5. RE - ASSESSMENT: Schedule follow - up functional movement assessment in 4 weeks

6. PATIENT EDUCATION:
${allRecommendations.slice(0, 3).map(r => `   - ${r}`).join('\n')}

CPT CODES: 97163, 97110, 97112, 98975, 98977
  `.trim()

  return {
    subjective,
    objective,
    assessment: assessmentSummary,
    plan
  }
}

// ============================================================================
// FRONTEND ROUTES
// ============================================================================

// Assessment page - serve static HTML
app.get('/assessment', (c) => {
  return c.redirect('/static/assessment')
})

// Intake page - new patient registration
app.get('/intake', (c) => {
  return c.redirect('/static/intake')
})

// Patients list page
app.get('/patients', (c) => {
  return c.redirect('/static/patients')
})

// Login redirect
app.get('/login', (c) => {
  return c.redirect('/static/login.html')
})

app.get('/register', (c) => {
  return c.redirect('/static/register.html')
})

// Main dashboard
app.get('/', (c) => {
  return c.html(`
  < !DOCTYPE html >
    <html lang="en">
      <head>
        <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PhysioMotion - Medical Movement Assessment Platform</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
            </head>
            <body class="bg-gray-50">
              <script>
            // Check authentication on page load
                const session = localStorage.getItem('clinician_session') || sessionStorage.getItem('clinician_session');
                if (!session) {
                  window.location.href = '/static/login.html';
            }
              </script>
              <div id="app">
                <!-- Navigation -->
                <nav class="bg-white shadow-lg border-b-2 border-cyan-500">
                  <div class="max-w-7xl mx-auto px-4">
                    <div class="flex justify-between h-16">
                      <div class="flex items-center">
                        <i class="fas fa-heartbeat text-cyan-600 text-2xl mr-3"></i>
                        <span class="text-xl font-bold text-slate-800">PhysioMotion</span>
                      </div>
                      <div class="flex items-center space-x-4">
                        <a href="/" class="text-gray-700 hover:text-cyan-600 transition-colors"><i class="fas fa-home mr-2"></i>Home</a>
                        <a href="/patients" class="text-gray-700 hover:text-cyan-600 transition-colors"><i class="fas fa-users mr-2"></i>Patients</a>
                        <a href="/intake" class="text-gray-700 hover:text-cyan-600 transition-colors"><i class="fas fa-user-plus mr-2"></i>New Patient</a>
                        <a href="/assessment" class="text-gray-700 hover:text-cyan-600 transition-colors"><i class="fas fa-video mr-2"></i>Assessment</a>
                        <div class="flex items-center space-x-3 ml-4 pl-4 border-l border-gray-300">
                          <span id="clinicianName" class="text-gray-700 font-medium"></span>
                          <button onclick="logout()" class="text-red-600 hover:text-red-700 transition-colors">
                            <i class="fas fa-sign-out-alt mr-1"></i>Logout
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </nav>

                <!-- Hero Section -->
                <div class="bg-gradient-to-r from-cyan-600 to-purple-600 text-white py-16">
                  <div class="max-w-7xl mx-auto px-4 text-center">
                    <h1 class="text-4xl font-bold mb-4">Medical Movement Assessment Platform</h1>
                    <p class="text-xl mb-8">AI-Powered Biomechanical Analysis for Physical Therapy & Chiropractic Care</p>
                    <div class="flex justify-center space-x-4">
                      <button onclick="window.location.href='/intake'" class="bg-white text-cyan-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-all">
                        <i class="fas fa-user-plus mr-2"></i>New Patient Intake
                      </button>
                      <button onclick="window.location.href='/assessment'" class="bg-purple-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-purple-400 transition-all">
                        <i class="fas fa-video mr-2"></i>Start Assessment
                      </button>
                    </div>
                  </div>
                </div>

                <!-- Features Grid -->
                <div class="max-w-7xl mx-auto px-4 py-12">
                  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Feature 1 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                      <div class="text-cyan-600 text-3xl mb-4"><i class="fas fa-camera"></i></div>
                      <h3 class="text-xl font-bold mb-2">Real-Time Motion Capture</h3>
                      <p class="text-gray-600">Advanced Orbbec Femto Mega integration with Azure Kinect Body Tracking SDK for professional clinical assessments</p>
                    </div>

                    <!-- Feature 2 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                      <div class="text-purple-600 text-3xl mb-4"><i class="fas fa-brain"></i></div>
                      <h3 class="text-xl font-bold mb-2">AI Biomechanical Analysis</h3>
                      <p class="text-gray-600">Automated joint angle calculations, ROM measurements, and compensation pattern detection</p>
                    </div>

                    <!-- Feature 3 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                      <div class="text-violet-600 text-3xl mb-4"><i class="fas fa-mobile-alt"></i></div>
                      <h3 class="text-xl font-bold mb-2">Home Exercise Monitoring</h3>
                      <p class="text-gray-600">MediaPipe Pose integration for remote patient monitoring via mobile camera with real-time feedback</p>
                    </div>

                    <!-- Feature 4 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                      <div class="text-rose-600 text-3xl mb-4"><i class="fas fa-file-medical"></i></div>
                      <h3 class="text-xl font-bold mb-2">Automated Medical Notes</h3>
                      <p class="text-gray-600">AI-generated SOAP notes with comprehensive deficiency documentation and treatment plans</p>
                    </div>

                    <!-- Feature 5 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                      <div class="text-teal-600 text-3xl mb-4"><i class="fas fa-dumbbell"></i></div>
                      <h3 class="text-xl font-bold mb-2">Exercise Prescription</h3>
                      <p class="text-gray-600">Evidence-based exercise library with automated prescription based on identified deficiencies</p>
                    </div>

                    <!-- Feature 6 -->
                    <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
                      <div class="text-indigo-600 text-3xl mb-4"><i class="fas fa-dollar-sign"></i></div>
                      <h3 class="text-xl font-bold mb-2">Medical Billing</h3>
                      <p class="text-gray-600">Integrated CPT coding with RPM/RTM billing support for remote patient monitoring</p>
                    </div>
                  </div>
                </div>

                <!-- Stats Section -->
                <div class="bg-gradient-to-r from-slate-50 to-slate-100 py-12">
                  <div class="max-w-7xl mx-auto px-4">
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 text-center">
                      <div>
                        <div class="text-4xl font-bold text-cyan-600">32</div>
                        <div class="text-gray-600">Joint Points Tracked</div>
                      </div>
                      <div>
                        <div class="text-4xl font-bold text-purple-600">15+</div>
                        <div class="text-gray-600">Evidence-Based Exercises</div>
                      </div>
                      <div>
                        <div class="text-4xl font-bold text-violet-600">Real-Time</div>
                        <div class="text-gray-600">Analysis & Feedback</div>
                      </div>
                      <div>
                        <div class="text-4xl font-bold text-teal-600">92%</div>
                        <div class="text-gray-600">AI Confidence Score</div>
                      </div>
                    </div>
                  </div>
                </div>

                <!-- Footer -->
                <footer class="bg-gradient-to-r from-slate-800 to-slate-900 text-white py-8 border-t-4 border-cyan-500">
                  <div class="max-w-7xl mx-auto px-4 text-center">
                    <p>&copy; 2025 PhysioMotion. Medical-Grade Movement Assessment Platform.</p>
                    <p class="text-sm text-slate-400 mt-2">Powered by Orbbec Femto Mega, Azure Kinect Body Tracking SDK, and MediaPipe Pose</p>
                  </div>
                </footer>
              </div>

              <script src="https://cdn.jsdelivr.net/npm/axios@1.6.0/dist/axios.min.js"></script>
              <script>
            // Display logged-in clinician info
                function displayClinicianInfo() {
                const session = localStorage.getItem('clinician_session') || sessionStorage.getItem('clinician_session');
                if (session) {
                    try {
                        const clinician = JSON.parse(session);
                const nameElement = document.getElementById('clinicianName');
                if (nameElement) {
                            const demoLabel = clinician.is_demo ? ' <span class="text-xs bg-violet-500 text-white px-2 py-1 rounded">DEMO</span>' : '';
                nameElement.innerHTML = '<i class="fas fa-user-md mr-1"></i>' + clinician.first_name + ' ' + clinician.last_name + (clinician.title ? ', ' + clinician.title : '') + demoLabel;
                        }
                    } catch (e) {
                  console.error('Error parsing session:', e);
                    }
                }
            }

                // Logout function
                function logout() {
                if (confirm('Are you sure you want to logout?')) {
                  localStorage.removeItem('clinician_session');
                sessionStorage.removeItem('clinician_session');
                window.location.href = '/static/login.html';
                }
            }

                // Initialize on page load
                displayClinicianInfo();
              </script>
            </body>
          </html>
          `)
})

export default app
validateEnv();
