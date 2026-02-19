import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'
import type { Bindings, Patient, Assessment, Exercise, PrescribedExercise, ExerciseSession, SkeletonData } from './types'
import { performBiomechanicalAnalysis } from './utils/biomechanics'
import { queryExerciseKnowledge } from './utils/rag'

const app = new Hono<{ Bindings: Bindings }>()

// Enable CORS for API routes
app.use('/api/*', cors())

// Serve static files
app.use('/static/*', serveStatic({ root: './public', manifest: {} }))

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

// Register new clinician
app.post('/api/auth/register', async (c) => {
  try {
    const data = await c.req.json()
    
    // Check if email already exists
    const existing = await c.env.DB.prepare(`
      SELECT id FROM clinicians WHERE email = ?
    `).bind(data.email).first()
    
    if (existing) {
      return c.json({ success: false, error: 'Email already registered' }, 400)
    }
    
    // Hash password (simple hash for demo - use bcrypt in production)
    const passwordHash = await hashPassword(data.password)
    
    const result = await c.env.DB.prepare(`
      INSERT INTO clinicians (
        email, password_hash, first_name, last_name, title,
        license_number, license_state, npi_number, phone, clinic_name
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      data.email, passwordHash, data.first_name, data.last_name, data.title,
      data.license_number, data.license_state, data.npi_number,
      data.phone, data.clinic_name
    ).run()
    
    return c.json({
      success: true,
      data: { id: result.meta.last_row_id }
    })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Login
app.post('/api/auth/login', async (c) => {
  try {
    const { email, password } = await c.req.json()
    
    const clinician = await c.env.DB.prepare(`
      SELECT * FROM clinicians WHERE email = ? AND active = 1
    `).bind(email).first() as any
    
    if (!clinician) {
      return c.json({ success: false, error: 'Invalid email or password' }, 401)
    }
    
    // Verify password
    const isValid = await verifyPassword(password, clinician.password_hash)
    
    if (!isValid) {
      return c.json({ success: false, error: 'Invalid email or password' }, 401)
    }
    
    // Automatic migration to PBKDF2 if using legacy hash
    if (!clinician.password_hash.includes(':')) {
      const newHash = await hashPassword(password);
      await c.env.DB.prepare(`
        UPDATE clinicians SET password_hash = ? WHERE id = ?
      `).bind(newHash, clinician.id).run();
    }

    // Update last login
    await c.env.DB.prepare(`
      UPDATE clinicians SET last_login = CURRENT_TIMESTAMP WHERE id = ?
    `).bind(clinician.id).run()
    
    // Return user data (excluding password)
    const { password_hash, ...userData } = clinician
    
    return c.json({
      success: true,
      data: userData
    })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get current clinician profile
app.get('/api/auth/profile/:id', async (c) => {
  try {
    const id = c.req.param('id')
    
    const clinician = await c.env.DB.prepare(`
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
// PATIENT MANAGEMENT API
// ============================================================================

// Create new patient
app.post('/api/patients', async (c) => {
  try {
    const patient = await c.req.json<Patient>()
    
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

// Get all patients
app.get('/api/patients', async (c) => {
  try {
    const { results } = await c.env.DB.prepare(`
      SELECT * FROM patients ORDER BY created_at DESC
    `).all() as { results: any[] }
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get patient by ID
app.get('/api/patients/:id', async (c) => {
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

// ============================================================================
// MEDICAL HISTORY API
// ============================================================================

app.post('/api/patients/:id/medical-history', async (c) => {
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

// ============================================================================
// ASSESSMENT API
// ============================================================================

// Create new assessment
app.post('/api/assessments', async (c) => {
  try {
    const assessment = await c.req.json<Assessment>()
    
    const result = await c.env.DB.prepare(`
      INSERT INTO assessments (
        patient_id, clinician_id, assessment_type, status
      ) VALUES (?, ?, ?, ?)
    `).bind(
      assessment.patient_id, assessment.clinician_id || 1,
      assessment.assessment_type, 'in_progress'
    ).run()
    
    return c.json({ success: true, data: { id: result.meta.last_row_id, ...assessment } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get assessments for patient
app.get('/api/patients/:id/assessments', async (c) => {
  try {
    const patientId = c.req.param('id')
    const { results } = await c.env.DB.prepare(`
      SELECT * FROM assessments WHERE patient_id = ? ORDER BY assessment_date DESC
    `).bind(patientId).all() as { results: any[] }
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get all assessments
app.get('/api/assessments', async (c) => {
  try {
    const { results } = await c.env.DB.prepare(`
      SELECT a.*, p.first_name, p.last_name 
      FROM assessments a
      JOIN patients p ON a.patient_id = p.id
      ORDER BY a.assessment_date DESC
    `).all() as { results: any[] }
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get assessment by ID
app.get('/api/assessments/:id', async (c) => {
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

// ============================================================================
// MOVEMENT TEST API
// ============================================================================

// Create movement test
app.post('/api/assessments/:id/tests', async (c) => {
  try {
    const assessmentId = c.req.param('id')
    const test = await c.req.json()
    
    const result = await c.env.DB.prepare(`
      INSERT INTO movement_tests (
        assessment_id, test_name, test_category, test_order, instructions, status
      ) VALUES (?, ?, ?, ?, ?, ?)
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
app.post('/api/tests/:id/analyze', async (c) => {
  try {
    const testId = c.req.param('id')
    const { skeleton_data } = await c.req.json<{ skeleton_data: SkeletonData }>()
    
    // Perform biomechanical analysis
    const analysis = performBiomechanicalAnalysis(skeleton_data)
    
    // Update movement test with skeleton data and analysis results
    await c.env.DB.prepare(`
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
app.get('/api/tests/:id/results', async (c) => {
  try {
    const testId = c.req.param('id')
    
    const test = await c.env.DB.prepare(`
      SELECT * FROM movement_tests WHERE id = ?
    `).bind(testId).first()
    
    if (!test) {
      return c.json({ success: false, error: 'Test not found' }, 404)
    }
    
    return c.json({ success: true, data: test })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get movement tests for assessment
app.get('/api/assessments/:id/tests', async (c) => {
  try {
    const assessmentId = c.req.param('id')
    const { results } = await c.env.DB.prepare(`
      SELECT * FROM movement_tests WHERE assessment_id = ? ORDER BY test_order
    `).bind(assessmentId).all() as { results: any[] }
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// EXERCISE LIBRARY API
// ============================================================================

// Get all exercises (with caching)
app.get('/api/exercises', async (c) => {
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
    
    const { results } = await c.env.DB.prepare(query).bind(...params).all() as { results: any[] }
    
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
app.post('/api/prescriptions', async (c) => {
  try {
    const prescription = await c.req.json<PrescribedExercise>()
    
    const result = await c.env.DB.prepare(`
      INSERT INTO prescribed_exercises (
        patient_id, assessment_id, exercise_id, sets, repetitions,
        times_per_week, clinical_reason, target_deficiency, prescribed_by
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      prescription.patient_id, prescription.assessment_id,
      prescription.exercise_id, prescription.sets, prescription.repetitions,
      prescription.times_per_week, prescription.clinical_reason,
      prescription.target_deficiency, prescription.prescribed_by || 1
    ).run()
    
    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get patient's prescribed exercises
app.get('/api/patients/:id/prescriptions', async (c) => {
  try {
    const patientId = c.req.param('id')
    
    const { results } = await c.env.DB.prepare(`
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
    `).bind(patientId).all() as { results: any[] }
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// EXERCISE SESSION API (Remote Patient Monitoring)
// ============================================================================

// Record exercise session
app.post('/api/exercise-sessions', async (c) => {
  try {
    const session = await c.req.json<ExerciseSession>()
    
    const result = await c.env.DB.prepare(`
      INSERT INTO exercise_sessions (
        patient_id, prescribed_exercise_id, sets_completed, reps_completed,
        duration_seconds, form_quality_score, pose_accuracy_data,
        pain_level_during, difficulty_rating, completed
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      session.patient_id, session.prescribed_exercise_id,
      session.sets_completed, session.reps_completed, session.duration_seconds,
      session.form_quality_score, session.pose_accuracy_data,
      session.pain_level_during, session.difficulty_rating, session.completed
    ).run()
    
    // Update compliance tracking
    if (c.executionCtx) {
      c.executionCtx.waitUntil(updateCompliancePercentage(c.env.DB, session.prescribed_exercise_id))
    } else {
      // Fallback for environments without executionCtx
      updateCompliancePercentage(c.env.DB, session.prescribed_exercise_id).catch(console.error)
    }
    
    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Get patient exercise history
app.get('/api/patients/:id/sessions', async (c) => {
  try {
    const patientId = c.req.param('id')
    
    const { results } = await c.env.DB.prepare(`
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
    `).bind(patientId).all() as { results: any[] }
    
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// BILLING API
// ============================================================================

// Get CPT codes (with caching)
app.get('/api/billing/codes', async (c) => {
  try {
    const now = Date.now()
    if (billingCodesCache && (now - lastBillingCacheTime < BILLING_CACHE_TTL)) {
        return c.json({ success: true, data: billingCodesCache })
    }

    const { results } = await c.env.DB.prepare(`
      SELECT * FROM billing_codes ORDER BY cpt_code
    `).all() as { results: any[] }
    
    billingCodesCache = results
    lastBillingCacheTime = now

    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Create billable event
app.post('/api/billing/events', async (c) => {
  try {
    const event = await c.req.json()
    
    const result = await c.env.DB.prepare(`
      INSERT INTO billable_events (
        patient_id, assessment_id, exercise_session_id,
        cpt_code_id, service_date, duration_minutes,
        clinical_note, provider_id
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      event.patient_id, event.assessment_id, event.exercise_session_id,
      event.cpt_code_id, event.service_date, event.duration_minutes,
      event.clinical_note, event.provider_id || 1
    ).run()
    
    return c.json({ success: true, data: { id: result.meta.last_row_id } })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// RAG & ANALYTICS API
// ============================================================================

app.post('/api/rag/query', async (c) => {
  try {
    const { query } = await c.req.json<{ query: string }>()
    const result = await queryExerciseKnowledge(c.env.DB, query)
    return c.json({ success: true, data: result })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// ============================================================================
// MEDICAL NOTE GENERATION
// ============================================================================

app.post('/api/assessments/:id/generate-note', async (c) => {
  try {
    const assessmentId = c.req.param('id')
    
    // Get assessment data
    const assessment = await c.env.DB.prepare(`
      SELECT * FROM assessments WHERE id = ?
    `).bind(assessmentId).first() as any
    
    // Get all movement tests and analyses
    const { results: tests } = await c.env.DB.prepare(`
      SELECT mt.*, ma.*
      FROM movement_tests mt
      LEFT JOIN movement_analysis ma ON mt.id = ma.test_id
      WHERE mt.assessment_id = ?
    `).bind(assessmentId).all() as { results: any[] }
    
    // Generate comprehensive medical note
    const medicalNote = generateMedicalNote(assessment, tests)
    
    // Try to enhance with AI insights if deficiencies exist
    let aiInsights = ""
    if (tests.length > 0 && tests[0].deficiencies && typeof tests[0].deficiencies === 'string') {
        try {
            const defs = JSON.parse(tests[0].deficiencies)
            if (defs && Array.isArray(defs) && defs.length > 0) {
                const ragResult = await queryExerciseKnowledge(c.env.DB, defs[0].area)
                aiInsights = ragResult.answer
            }
        } catch (e) {}
    }

    if (aiInsights) {
        medicalNote.plan += `\n\nAI CLINICAL INSIGHTS:\n${aiInsights}`
    }

    // Update assessment with generated notes
    await c.env.DB.prepare(`
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

// Password hashing using PBKDF2 (HIPAA-compliant)
async function hashPassword(password: string): Promise<string> {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const saltHex = Array.from(salt).map(b => b.toString(16).padStart(2, '0')).join('');

  const baseKey = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey']
  );

  const derivedKey = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt: salt,
      iterations: 100000,
      hash: 'SHA-256'
    },
    baseKey,
    256
  );

  const hashHex = Array.from(new Uint8Array(derivedKey))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');

  return `${saltHex}:${hashHex}`;
}

async function verifyPassword(password: string, storedHash: string): Promise<boolean> {
  // Legacy SHA-256 check with migration support
  if (!storedHash.includes(':')) {
    const encoder = new TextEncoder()
    const data = encoder.encode(password + 'physiomotion-salt-2025')
    const hashBuffer = await crypto.subtle.digest('SHA-256', data)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
    return hashHex === storedHash;
  }

  const [saltHex, hashHex] = storedHash.split(':');
  const match = saltHex.match(/.{1,2}/g);
  if (!match) return false;

  const salt = new Uint8Array(match.map(byte => parseInt(byte, 16)));

  const baseKey = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey']
  );

  const derivedKey = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt: salt,
      iterations: 100000,
      hash: 'SHA-256'
    },
    baseKey,
    256
  );

  const computedHashHex = Array.from(new Uint8Array(derivedKey))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');

  return computedHashHex === hashHex;
}

async function updateCompliancePercentage(db: any, prescribedExerciseId: number) {
  // Get total sessions completed vs expected
  const result = await db.prepare(`
    SELECT COUNT(*) as completed_count
    FROM exercise_sessions
    WHERE prescribed_exercise_id = ? AND completed = 1
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
      } catch (e) {}
    }
    if (test.ai_recommendations && typeof test.ai_recommendations === 'string') {
      try {
        const recs = JSON.parse(test.ai_recommendations)
        if (Array.isArray(recs)) {
          allRecommendations.push(...recs)
        }
      } catch (e) {}
    }
    if (test.movement_quality_score) {
      avgQualityScore += test.movement_quality_score
    }
  }
  
  avgQualityScore = tests.length > 0 ? Math.round(avgQualityScore / tests.length) : 0
  
  // Generate SOAP note
  const subjective = `Patient presents for ${assessment.assessment_type} assessment. Willing and able to participate in functional movement screening.`
  
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
  'Minor limitations identified. Preventive exercise program appropriate.'}
  `.trim()
  
  const plan = `
TREATMENT PLAN:

1. THERAPEUTIC EXERCISES: Prescribed evidence-based exercise program targeting identified deficiencies
   ${allDeficiencies.slice(0, 3).map(d => `   - Address ${d.area}`).join('\n')}

2. FREQUENCY: ${avgQualityScore < 60 ? '2-3x per week supervised therapy + daily HEP' : '1-2x per week supervised + daily HEP'}

3. DURATION: ${avgQualityScore < 60 ? '8-12 weeks' : '4-8 weeks'}

4. REMOTE MONITORING: Patient enrolled in remote therapeutic monitoring program (RTM)
   - Daily exercise compliance tracking via mobile app
   - Real-time form analysis and feedback
   - Weekly progress reports

5. RE-ASSESSMENT: Schedule follow-up functional movement assessment in 4 weeks

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
    <!DOCTYPE html>
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
