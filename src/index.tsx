import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger as honoLogger } from 'hono/logger';
import mskRouter from './routes/msk-analysis.js';
import aiRoutes from './routes/ai-analysis.js';
import { authMiddleware } from './middleware/auth.js';
import { analyzeMovement, quickAssessment, getAIInfo } from './utils/ai-analysis.js';
import { searchKnowledge, getRehabProtocol, REHAB_PROTOCOLS, ROM_NORMATIVE, CLINICAL_CALCULATORS, MEDICAL_CONDITIONS, CPT_CODES } from './utils/rag-knowledge.js';
import {
  listPatients, getPatient, createPatient, updatePatient,
  listAssessments, getAssessment, createAssessment,
  addTest, createPrescription, recordSession,
  listExercises, getExercise, getDashboardStats,
  getClinicianByEmail, createClinician,
  listGaitSessions, listMuscleAssessments, listExercisePrescriptions, listProgressMetrics,
} from './lib/database.js';
import { hashPassword, verifyPassword, generateToken } from './middleware/auth.js'
import { ProviderPortal } from './pages/ProviderPortal.js'
import { GaitAnalyzer } from './components/GaitAnalyzer.js'
import { MuscleAssessment } from './components/MuscleAssessment.js'
import { ClinicalTests } from './components/ClinicalTests.js'
import { ExercisePrescriber } from './components/ExercisePrescriber.js'
import { ProgressTracker } from './components/ProgressTracker.js'
import { ReportGenerator } from './components/ReportGenerator.js'
import { analyzeGait, assessMuscles, calculateFuglMeyer, runClinicalTests, prescribeExercises } from './utils/clinical.js'

const app = new Hono();

// Middleware
app.use('*', honoLogger());
app.use('*', cors({
  origin: process.env.NODE_ENV === 'production'
    ? ['https://imw-msk-assesment.vercel.app', 'https://imwmsk.com', 'https://www.imwmsk.com']
    : '*',
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization']
}));

// ============================================================================
// SPECIALIST AGENT ROUTES
// ============================================================================
app.route('/msk-analysis', mskRouter);

// ============================================================================
// AI ANALYSIS ROUTES
// ============================================================================
app.route('/ai', aiRoutes);

// Enhanced AI movement analysis (with RAG)
app.post('/ai/analyze-movement', authMiddleware, async (c) => {
  const body = await c.req.json();
  const { patientId, movementTest, landmarks, bodyRegion, frameCount, depthData } = body;

  if (!patientId || !movementTest || !landmarks || !bodyRegion) {
    return c.json({ success: false, error: 'patientId, movementTest, landmarks, and bodyRegion are required' }, 400);
  }

  try {
    const result = await analyzeMovement({ patientId, movementTest, landmarks, bodyRegion, frameCount, depthData });
    return c.json({ success: true, data: result, ai: getAIInfo() });
  } catch (err: any) {
    return c.json({ success: false, error: err.message }, 500);
  }
});

// Quick clinical assessment (no camera needed)
app.post('/ai/quick-assess', authMiddleware, async (c) => {
  const body = await c.req.json();
  const { patientId, bodyRegion, chiefComplaint, painScale, functionalLimitations } = body;

  if (!patientId || !bodyRegion || !chiefComplaint) {
    return c.json({ success: false, error: 'patientId, bodyRegion, and chiefComplaint are required' }, 400);
  }

  try {
    const result = await quickAssessment({ patientId, bodyRegion, chiefComplaint, painScale: painScale || 0, functionalLimitations: functionalLimitations || '' });
    return c.json({ success: true, data: result, ai: getAIInfo() });
  } catch (err: any) {
    return c.json({ success: false, error: err.message }, 500);
  }
});

// AI provider info
app.get('/ai/info', (c) => {
  return c.json({ success: true, data: getAIInfo() });
});

// ============================================================================
// RAG KNOWLEDGE ROUTES
// ============================================================================
app.get('/knowledge/search', (c) => {
  const q = c.req.query('q') || '';
  const results = searchKnowledge(q);
  return c.json({ success: true, data: results, count: results.length });
});

app.get('/knowledge/conditions', (c) => {
  const region = c.req.query('region');
  const conditions = region
    ? MEDICAL_CONDITIONS.filter(c => c.category === region)
    : MEDICAL_CONDITIONS;
  return c.json({ success: true, data: conditions.map(c => ({ name: c.name, icd10: c.icd10, category: c.category, specialTests: c.specialTests })) });
});

app.get('/knowledge/rehab/:condition', (c) => {
  const condition = c.req.param('condition');
  const protocol = getRehabProtocol(condition);
  if (!protocol) return c.json({ success: false, error: 'No protocol found for condition' }, 404);
  return c.json({ success: true, data: protocol });
});

app.get('/knowledge/rom', (c) => {
  return c.json({ success: true, data: ROM_NORMATIVE });
});

app.get('/knowledge/calculators', (c) => {
  return c.json({ success: true, data: Object.entries(CLINICAL_CALCULATORS).map(([key, calc]) => ({ key, ...calc })) });
});

app.get('/knowledge/cpt', (c) => {
  return c.json({ success: true, data: CPT_CODES });
});

// ============================================================================
// CLINICAL PLATFORM PHASES 4-9
// ============================================================================
app.post('/clinical/gait/analyze', authMiddleware, async (c) => {
  const body = await c.req.json();
  const metrics = analyzeGait(body.keypoints || body.landmarks || [], body.frameRate || 30, body.history || []);
  return c.json({ success: true, data: metrics });
});

app.post('/clinical/muscle/assess', authMiddleware, async (c) => {
  const body = await c.req.json();
  const grades = assessMuscles(body.keypoints || body.landmarks || [], body.previousKeypoints || []);
  return c.json({ success: true, data: { grades, fma: calculateFuglMeyer(grades) } });
});

app.post('/clinical/tests/run', authMiddleware, async (c) => {
  const body = await c.req.json();
  const results = runClinicalTests(body.keypoints || body.landmarks || []);
  return c.json({ success: true, data: results });
});

app.post('/clinical/exercises/prescribe', authMiddleware, async (c) => {
  const body = await c.req.json();
  const exercises = prescribeExercises(body.findings || [], body.filters || {});
  return c.json({ success: true, data: exercises });
});

app.get('/gait', (c) => c.html(<GaitAnalyzer />));
app.get('/muscle', (c) => c.html(<MuscleAssessment />));
app.get('/clinical-tests', (c) => c.html(<ClinicalTests />));
app.get('/progress', (c) => c.html(<ProgressTracker />));
app.get('/reports', (c) => c.html(<ReportGenerator />));

// ============================================================================
// ENHANCED ASSESSMENT
// ============================================================================
app.post('/assessments/enhanced', authMiddleware, async (c) => {
  const body = await c.req.json();
  const { patient_id, assessment_type, clinician_id, chief_complaint, pain_scale, movement_test, landmarks, body_region, run_ai } = body;

  const patient = await getPatient(patient_id);
  if (!patient) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }

  const newAssessment = await createAssessment({
    patient_id,
    assessment_type: assessment_type || 'initial',
    clinician_id,
    chief_complaint: chief_complaint || '',
    pain_scale: pain_scale || 0,
    movement_test: movement_test || null,
    goals: body.goals || [],
    clinical_notes: '',
    ai_analysis: null,
  });

  // Run AI analysis if landmarks provided
  if (run_ai && landmarks && body_region) {
    try {
      const aiResult = await analyzeMovement({
        patientId: patient_id,
        movementTest: movement_test || 'general_assessment',
        landmarks,
        bodyRegion: body_region,
      });
      // Update assessment with AI results
      const updated = { ...newAssessment, ai_analysis: aiResult, clinical_notes: aiResult.soapNote.assessment };
      // Add recommended tests
      updated.tests = (aiResult.recommendedTests || []).map((t: any) => ({
        id: Date.now() + Math.random(),
        test_name: t.name,
        rationale: t.rationale,
        status: 'pending',
      }));
      return c.json({ success: true, data: updated }, 201);
    } catch (err: any) {
      console.error('[Enhanced Assessment] AI analysis failed:', err.message);
      return c.json({ success: true, data: { ...newAssessment, ai_analysis: { error: err.message } } }, 201);
    }
  }

  return c.json({ success: true, data: newAssessment }, 201);
});

// ============================================================================
// HEALTH CHECK
// ============================================================================
app.get('/health', (c) => {
  return c.json({
    success: true,
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '3.0.0',
    commit: 'supabase-integration',
    ai: getAIInfo(),
    features: {
      patients: true,
      assessments: true,
      exercises: true,
      camera: true,
      reporting: true,
      ai_analysis: true,
      rag_knowledge: {
        conditions: MEDICAL_CONDITIONS.length,
        calculators: Object.keys(CLINICAL_CALCULATORS).length,
        rom_norms: Object.keys(ROM_NORMATIVE).length,
        cpt_codes: Object.keys(CPT_CODES).length,
        rehab_protocols: Object.keys(REHAB_PROTOCOLS).length,
      },
      telegram_notifications: !!process.env.TELEGRAM_BOT_TOKEN,
    }
  });
});

// ============================================================================
// PATIENT ROUTES
// ============================================================================

app.get('/patients', authMiddleware, async (c) => {
  const patients = await listPatients();
  return c.json({ success: true, data: patients });
});

app.get('/patients/:id', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const patient = await getPatient(id);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  return c.json({ success: true, data: patient });
});

app.post('/patients', authMiddleware, async (c) => {
  const body = await c.req.json();
  const newPatient = await createPatient(body);
  return c.json({ success: true, data: newPatient }, 201);
});

app.put('/patients/:id', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const body = await c.req.json();
  const patient = await updatePatient(id, body);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  return c.json({ success: true, data: patient });
});

app.get('/patients/:id/medical-history', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const patient = await getPatient(id);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  return c.json({ success: true, data: patient.medical_history || {} });
});

app.post('/patients/:id/medical-history', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const body = await c.req.json();
  const patient = await getPatient(id);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  const updated = await updatePatient(id, {
    medical_history: { ...patient.medical_history, ...body }
  });
  return c.json({ success: true, data: updated?.medical_history });
});

app.get('/patients/:id/assessments', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const patient = await getPatient(id);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  // For now, filter from all assessments
  const all = await listAssessments();
  const filtered = all.filter((a: any) => String(a.patient_id) === id);
  return c.json({ success: true, data: filtered });
});

app.get('/patients/:id/prescriptions', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const patient = await getPatient(id);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  const prescriptions = await listExercisePrescriptions(id);
  return c.json({ success: true, data: prescriptions });
});

app.get('/patients/:id/sessions', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const patient = await getPatient(id);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  return c.json({ success: true, data: [] });
});

// ============================================================================
// ASSESSMENT ROUTES
// ============================================================================

app.post('/assessments', authMiddleware, async (c) => {
  const body = await c.req.json();
  const { patient_id, assessment_type, clinician_id } = body;
  const patient = await getPatient(patient_id);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  const newAssessment = await createAssessment({
    patient_id, assessment_type, clinician_id,
    pain_scale: body.pain_scale || 0,
    chief_complaint: body.chief_complaint || '',
    goals: body.goals || [],
  });
  return c.json({ success: true, data: newAssessment }, 201);
});

app.get('/assessments', authMiddleware, async (c) => {
  const assessments = await listAssessments();
  return c.json({ success: true, data: assessments });
});

app.get('/assessments/:id', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const assessment = await getAssessment(id);
  if (!assessment) return c.json({ success: false, error: 'Assessment not found' }, 404);
  return c.json({ success: true, data: assessment });
});

app.post('/assessments/:id/tests', authMiddleware, async (c) => {
  const assessmentId = c.req.param('id');
  const body = await c.req.json();
  const test = await addTest(assessmentId, body);
  return c.json({ success: true, data: test }, 201);
});

app.post('/assessments/:id/generate-note', authMiddleware, async (c) => {
  const assessmentId = c.req.param('id');
  const assessment = await getAssessment(assessmentId);
  if (!assessment) return c.json({ success: false, error: 'Assessment not found' }, 404);
  const patient = await getPatient(assessment.patient_id);
  const note = generateClinicalNote(patient || {}, assessment);
  return c.json({ success: true, data: { note } });
});

function generateClinicalNote(patient: any, assessment: any) {
  const tests = assessment.tests || [];
  const prescriptions = assessment.prescriptions || [];
  let note = `CLINICAL NOTE\n==============\n\n`;
  note += `Patient: ${patient?.first_name || ''} ${patient?.last_name || ''}\n`;
  note += `DOB: ${patient?.date_of_birth || 'N/A'}\n`;
  note += `Assessment Date: ${new Date(assessment.assessment_date).toLocaleDateString()}\n`;
  note += `Clinician: ${assessment.clinician_id || 'Dr. PT Clinician'}\n\n`;
  note += `CHIEF COMPLAINT:\n${assessment.chief_complaint || 'No chief complaint recorded'}\n\n`;
  note += `SUBJECTIVE:\nPain Level: ${assessment.pain_scale || 0}/10\n\n`;
  note += `OBJECTIVE:\n`;
  tests.forEach((test: any) => {
    note += `- ${test.test_name}:\n`;
    if (test.results) {
      Object.entries(test.results).forEach(([key, value]: [string, any]) => {
        note += `  ${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}\n`;
      });
    }
  });
  note += `\nASSESSMENT:\n${assessment.clinical_notes || 'Functional movement deficits identified.'}\n\n`;
  note += `PLAN:\n1. Physical therapy evaluation and treatment\n2. Home exercise program compliance essential\n`;
  if (assessment.goals?.length) note += `3. Goals: ${assessment.goals.join('; ')}\n`;
  note += `4. Re-evaluate in 2-4 weeks\n\n`;
  note += `PRESCRIBED EXERCISES:\n`;
  prescriptions.forEach((p: any, i: number) => {
    note += `${i + 1}. ${p.exercise_name}: ${p.sets} sets x ${p.reps} reps, ${p.frequency}\n`;
    if (p.notes) note += `   Notes: ${p.notes}\n`;
  });
  return note;
}

// ============================================================================
// EXERCISE ROUTES
// ============================================================================

app.get('/exercises', async (c) => {
  const accept = c.req.header('Accept') || '';
  if (accept.includes('text/html')) return c.html(<ExercisePrescriber />);
  const category = c.req.query('category');
  const exercises = await listExercises(category || undefined);
  return c.json({ success: true, data: exercises });
});

app.get('/exercises/:id', async (c) => {
  const id = c.req.param('id');
  const exercise = await getExercise(id);
  if (!exercise) return c.json({ success: false, error: 'Exercise not found' }, 404);
  return c.json({ success: true, data: exercise });
});

// ============================================================================
// PRESCRIPTION ROUTES
// ============================================================================

app.post('/prescriptions', authMiddleware, async (c) => {
  const body = await c.req.json();
  const prescription = await createPrescription(body);
  return c.json({ success: true, data: prescription }, 201);
});

// ============================================================================
// SESSION ROUTES
// ============================================================================

app.post('/exercise-sessions', authMiddleware, async (c) => {
  const body = await c.req.json();
  const session = await recordSession(body);
  return c.json({ success: true, data: session }, 201);
});

// ============================================================================
// ANALYTICS & REPORTING
// ============================================================================

app.get('/dashboard/stats', authMiddleware, async (c) => {
  const stats = await getDashboardStats();
  return c.json({ success: true, data: stats });
});

app.get('/patients/:id/progress', authMiddleware, async (c) => {
  const id = c.req.param('id');
  const patient = await getPatient(id);
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404);
  const [progressMetrics, gaitSessions, muscleAssessments, prescriptions] = await Promise.all([
    listProgressMetrics(id),
    listGaitSessions(id),
    listMuscleAssessments(id),
    listExercisePrescriptions(id),
  ]);
  const first = progressMetrics[0]?.current_value || 0;
  const latest = progressMetrics[progressMetrics.length - 1]?.current_value || first;
  const progress = {
    pain_trend: patient.progress_metrics?.pain_trend || [],
    functional_trend: patient.progress_metrics?.functional_score_trend || [],
    metrics: progressMetrics,
    gait_sessions: gaitSessions,
    muscle_assessments: muscleAssessments,
    exercise_prescriptions: prescriptions,
    sessions_completed: gaitSessions.length,
    total_exercise_time: 0,
    adherence_rate: prescriptions.length ? 80 : 0,
    latest_pain: patient.progress_metrics?.pain_trend?.at?.(-1) || 0,
    improvement_percentage: first ? Math.round(((latest - first) / Math.abs(first)) * 100) : 0,
  };
  return c.json({ success: true, data: progress });
});

// ============================================================================
// AUTH ROUTES
// ============================================================================

app.post('/auth/login', async (c) => {
  const body = await c.req.json();
  const { email, password } = body;

  // Development demo mode
  if (process.env.NODE_ENV !== 'production' && (!email || !password || password === 'demo')) {
    return c.json({
      success: true,
      data: {
        user: { id: 'demo', name: 'Dr. Demo Clinician', email: email || 'demo@physiomotion.com', role: 'clinician', license: 'PT12345' },
        token: 'demo-token-12345',
        demo_mode: true
      }
    });
  }

  if (!email || !password) {
    return c.json({ success: false, error: 'Email and password required', code: 'MISSING_CREDENTIALS' }, 400);
  }

  try {
    const clinician = await getClinicianByEmail(email);
    if (!clinician) {
      return c.json({ success: false, error: 'Invalid credentials', code: 'INVALID_CREDENTIALS' }, 401);
    }

    const valid = await verifyPassword(password, clinician.password_hash);
    if (!valid) {
      return c.json({ success: false, error: 'Invalid credentials', code: 'INVALID_CREDENTIALS' }, 401);
    }

    const secret = process.env.JWT_SECRET || process.env.AUTH_SECRET;
    if (!secret) {
      return c.json({ success: false, error: 'Server configuration error', code: 'CONFIG_ERROR' }, 500);
    }

    const token = await generateToken(
      { id: clinician.id, email: clinician.email, role: clinician.role, clinic_id: clinician.clinic_id },
      secret
    );

    return c.json({
      success: true,
      data: {
        user: { id: clinician.id, name: clinician.name, email: clinician.email, role: clinician.role, license: clinician.license },
        token,
        demo_mode: false
      }
    });
  } catch (err: any) {
    console.error('[AUTH] Login error:', err);
    return c.json({ success: false, error: 'Authentication failed' }, 500);
  }
});

app.post('/auth/register', async (c) => {
  const body = await c.req.json();
  const { email, password, name, role, license } = body;

  if (!email || !password || !name) {
    return c.json({ success: false, error: 'Email, password, and name are required' }, 400);
  }

  try {
    const existing = await getClinicianByEmail(email);
    if (existing) {
      return c.json({ success: false, error: 'Email already registered', code: 'EMAIL_EXISTS' }, 409);
    }

    const passwordHash = await hashPassword(password);
    const clinician = await createClinician({
      email,
      password_hash: passwordHash,
      name,
      role: role || 'clinician',
      license: license || null,
    });

    return c.json({ success: true, data: { id: clinician.id, email: clinician.email, name: clinician.name } }, 201);
  } catch (err: any) {
    console.error('[AUTH] Register error:', err);
    return c.json({ success: false, error: 'Registration failed' }, 500);
  }
});

app.get('/auth/me', authMiddleware, (c) => {
  const clinician = c.get('clinician');
  return c.json({
    success: true,
    data: {
      user: {
        id: (clinician as any).id,
        name: (clinician as any).name || 'Clinician',
        email: clinician.email,
        role: clinician.role,
      },
      demo_mode: process.env.NODE_ENV !== 'production' && c.req.header('Authorization')?.replace('Bearer ', '') === 'demo-token-12345'
    }
  });
});

// ============================================================================
// TELEGRAM NOTIFICATIONS
// ============================================================================

app.post('/notify/telegram', authMiddleware, async (c) => {
  const body = await c.req.json() as any;
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const defaultChatId = process.env.TELEGRAM_CHAT_ID || '7838956683';

  if (!token) {
    return c.json({ success: false, error: 'TELEGRAM_BOT_TOKEN not configured' }, 500);
  }

  const chatId = body.chatId || defaultChatId;
  const prefix = body.type === 'alert' ? '🚨 ALERT'
    : body.type === 'assessment' ? '📋 Assessment'
    : body.type === 'daily_digest' ? '🌅 Daily Digest'
    : '📌';

  const fullMessage = `${prefix}${body.patientName ? ` — ${body.patientName}` : ''}\n${body.message}`;

  try {
    const tgUrl = `https://api.telegram.org/bot${token}/sendMessage`;
    const res = await fetch(tgUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text: fullMessage,
        parse_mode: 'HTML',
      }),
    });
    const data = await res.json() as any;
    if (!data.ok) throw new Error(data.description || 'Telegram API error');
    return c.json({ success: true, data: { messageId: data.result?.message_id, chat: chatId } });
  } catch (err: any) {
    return c.json({ success: false, error: err.message }, 500);
  }
});

// ============================================================================
// HEALTH CHECK
// ============================================================================

app.get('/health', (c) => {
  return c.json({ success: true, status: 'healthy', version: '2.0.0-demo', timestamp: new Date().toISOString() })
});

app.get('/api/health', (c) => {
  return c.json({ success: true, status: 'healthy', version: '2.0.0-demo', timestamp: new Date().toISOString() })
});

// ============================================================================
// PROVIDER PORTAL
// ============================================================================

app.get('/provider', (c) => {
  return c.html(<ProviderPortal />)
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

app.onError((err, c) => {
  console.error('Error:', err);
  return c.json({ success: false, error: 'Internal server error', message: err.message }, 500);
});

app.notFound((c) => {
  return c.json({ success: false, error: 'Not found', path: c.req.path }, 404);
});

export default app;
