import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger as honoLogger } from 'hono/logger';
import { MOCK_PATIENTS, EXERCISE_LIBRARY } from './mockData.js';
import mskRouter from './routes/msk-analysis.js';
import aiRoutes from './routes/ai-analysis.js';

const app = new Hono();

// Middleware
app.use('*', honoLogger());
app.use('*', cors({
  origin: '*',
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

// ============================================================================
// HEALTH CHECK
// ============================================================================
app.get('/health', (c) => {
  return c.json({
    success: true,
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '2.0.0-demo',
    commit: 'c8dd9fe',
    features: {
      patients: true,
      assessments: true,
      exercises: true,
      camera: true,
      reporting: true
    }
  });
});

// ============================================================================
// PATIENT ROUTES
// ============================================================================

// Get all patients
app.get('/patients', (c) => {
  const patients = MOCK_PATIENTS.map(p => ({
    id: p.id,
    first_name: p.first_name,
    last_name: p.last_name,
    date_of_birth: p.date_of_birth,
    gender: p.gender,
    email: p.email,
    phone: p.phone,
    created_at: p.created_at,
    medical_history: p.medical_history,
    assessments: p.assessments,
    exercise_sessions: p.exercise_sessions,
    progress_metrics: p.progress_metrics,
    pain_trend: p.progress_metrics?.pain_trend || [],
    latest_assessment: p.assessments?.[p.assessments.length - 1]?.assessment_date || null
  }));
  
  return c.json({ success: true, data: patients });
});

// Get patient by ID
app.get('/patients/:id', (c) => {
  const id = parseInt(c.req.param('id'));
  const patient = MOCK_PATIENTS.find(p => p.id === id);
  
  if (!patient) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  return c.json({ success: true, data: patient });
});

// Create new patient
app.post('/patients', async (c) => {
  const body = await c.req.json();
  
  const newPatient = {
    id: MOCK_PATIENTS.length + 1,
    ...body,
    created_at: new Date().toISOString(),
    assessments: [],
    exercise_sessions: [],
    progress_metrics: {
      pain_trend: [],
      functional_score_trend: []
    }
  };
  
  MOCK_PATIENTS.push(newPatient as any);
  
  return c.json({ success: true, data: newPatient }, 201);
});

// Update patient
app.put('/patients/:id', async (c) => {
  const id = parseInt(c.req.param('id'));
  const body = await c.req.json();
  
  const patientIndex = MOCK_PATIENTS.findIndex(p => p.id === id);
  if (patientIndex === -1) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  MOCK_PATIENTS[patientIndex] = { ...MOCK_PATIENTS[patientIndex], ...body };
  
  return c.json({ success: true, data: MOCK_PATIENTS[patientIndex] });
});

// Get patient medical history
app.get('/patients/:id/medical-history', (c) => {
  const id = parseInt(c.req.param('id'));
  const patient = MOCK_PATIENTS.find(p => p.id === id);
  
  if (!patient) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  return c.json({ 
    success: true, 
    data: patient.medical_history || {}
  });
});

// Add medical history
app.post('/patients/:id/medical-history', async (c) => {
  const id = parseInt(c.req.param('id'));
  const body = await c.req.json();
  
  const patientIndex = MOCK_PATIENTS.findIndex(p => p.id === id);
  if (patientIndex === -1) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  MOCK_PATIENTS[patientIndex].medical_history = {
    ...MOCK_PATIENTS[patientIndex].medical_history,
    ...body
  };
  
  return c.json({ success: true, data: MOCK_PATIENTS[patientIndex].medical_history });
});

// Get patient assessments
app.get('/patients/:id/assessments', (c) => {
  const id = parseInt(c.req.param('id'));
  const patient = MOCK_PATIENTS.find(p => p.id === id);
  
  if (!patient) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  return c.json({ 
    success: true, 
    data: patient.assessments || []
  });
});

// Get patient prescriptions
app.get('/patients/:id/prescriptions', (c) => {
  const id = parseInt(c.req.param('id'));
  const patient = MOCK_PATIENTS.find(p => p.id === id);
  
  if (!patient) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  const prescriptions = patient.assessments?.flatMap((a: any) => 
    a.prescriptions?.map((p: any) => ({
      ...p,
      assessment_id: a.id,
      assessment_date: a.assessment_date
    })) || []
  ) || [];
  
  return c.json({ success: true, data: prescriptions });
});

// Get patient sessions
app.get('/patients/:id/sessions', (c) => {
  const id = parseInt(c.req.param('id'));
  const patient = MOCK_PATIENTS.find(p => p.id === id);
  
  if (!patient) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  return c.json({ 
    success: true, 
    data: patient.exercise_sessions || []
  });
});

// ============================================================================
// ASSESSMENT ROUTES
// ============================================================================

// Create assessment
app.post('/assessments', async (c) => {
  const body = await c.req.json();
  const { patient_id, assessment_type, clinician_id } = body;
  
  const patientIndex = MOCK_PATIENTS.findIndex(p => p.id === patient_id);
  if (patientIndex === -1) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  const newAssessment = {
    id: Date.now(),
    patient_id,
    assessment_type,
    clinician_id,
    assessment_date: new Date().toISOString(),
    tests: [],
    prescriptions: [],
    pain_scale: body.pain_scale || 0,
    chief_complaint: body.chief_complaint || '',
    goals: body.goals || [],
    clinical_notes: ''
  };
  
  if (!MOCK_PATIENTS[patientIndex].assessments) {
    MOCK_PATIENTS[patientIndex].assessments = [];
  }
  MOCK_PATIENTS[patientIndex].assessments.push(newAssessment as any);
  
  return c.json({ success: true, data: newAssessment }, 201);
});

// Get all assessments
app.get('/assessments', (c) => {
  const allAssessments = MOCK_PATIENTS.flatMap(p => 
    (p.assessments || []).map(a => ({
      ...a,
      patient_id: p.id,
      patient_name: `${p.first_name} ${p.last_name}`
    }))
  );
  
  return c.json({ success: true, data: allAssessments });
});

// Get assessment by ID
app.get('/assessments/:id', (c) => {
  const id = parseInt(c.req.param('id'));
  
  for (const patient of MOCK_PATIENTS) {
    const assessment = patient.assessments?.find((a: any) => a.id === id);
    if (assessment) {
      return c.json({ 
        success: true, 
        data: { ...assessment, patient }
      });
    }
  }
  
  return c.json({ success: false, error: 'Assessment not found' }, 404);
});

// Add test to assessment
app.post('/assessments/:id/tests', async (c) => {
  const assessmentId = parseInt(c.req.param('id'));
  const body = await c.req.json();
  
  for (const patient of MOCK_PATIENTS) {
    const assessment = patient.assessments?.find((a: any) => a.id === assessmentId);
    if (assessment) {
      const newTest = {
        id: Date.now(),
        ...body,
        created_at: new Date().toISOString()
      };
      
      if (!assessment.tests) assessment.tests = [];
      assessment.tests.push(newTest);
      
      return c.json({ success: true, data: newTest }, 201);
    }
  }
  
  return c.json({ success: false, error: 'Assessment not found' }, 404);
});

// Generate clinical note
app.post('/assessments/:id/generate-note', async (c) => {
  const assessmentId = parseInt(c.req.param('id'));
  
  for (const patient of MOCK_PATIENTS) {
    const assessment = patient.assessments?.find((a: any) => a.id === assessmentId);
    if (assessment) {
      // Generate automated note based on assessment data
      const note = generateClinicalNote(patient, assessment);
      assessment.clinical_notes = note;
      
      return c.json({ success: true, data: { note } });
    }
  }
  
  return c.json({ success: false, error: 'Assessment not found' }, 404);
});

function generateClinicalNote(patient: any, assessment: any) {
  const tests = assessment.tests || [];
  const prescriptions = assessment.prescriptions || [];
  
  let note = `CLINICAL NOTE\n`;
  note += `==============\n\n`;
  note += `Patient: ${patient.first_name} ${patient.last_name}\n`;
  note += `DOB: ${patient.date_of_birth}\n`;
  note += `Assessment Date: ${new Date(assessment.assessment_date).toLocaleDateString()}\n`;
  note += `Clinician: ${assessment.clinician || 'Dr. PT Clinician'}\n\n`;
  
  note += `CHIEF COMPLAINT:\n`;
  note += `${assessment.chief_complaint || 'No chief complaint recorded'}\n\n`;
  
  note += `SUBJECTIVE:\n`;
  note += `Pain Level: ${assessment.pain_scale}/10\n`;
  note += `Patient reports ${assessment.functional_status || 'functional limitations as noted above'}.\n\n`;
  
  note += `OBJECTIVE:\n`;
  tests.forEach((test: any) => {
    note += `- ${test.test_name}:\n`;
    if (test.results) {
      Object.entries(test.results).forEach(([key, value]: [string, any]) => {
        if (typeof value === 'object' && value !== null) {
          note += `  ${key}: ${value.value || value.result || JSON.stringify(value)}\n`;
        } else {
          note += `  ${key}: ${value}\n`;
        }
      });
    }
  });
  note += `\n`;
  
  note += `ASSESSMENT:\n`;
  note += `${assessment.clinical_notes || 'Functional movement deficits identified. See objective findings above.'}\n\n`;
  
  note += `PLAN:\n`;
  note += `1. Physical therapy ${prescriptions.length > 0 ? 'with prescribed exercises' : 'evaluation and treatment'}\n`;
  note += `2. Home exercise program compliance essential\n`;
  if (assessment.goals && assessment.goals.length > 0) {
    note += `3. Goals: ${assessment.goals.join('; ')}\n`;
  }
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

// Get all exercises
app.get('/exercises', (c) => {
  const category = c.req.query('category');
  
  let exercises: any[] = EXERCISE_LIBRARY;
  if (category) {
    exercises = exercises.filter(e => e.category === category);
  }
  
  return c.json({ success: true, data: exercises });
});

// Get exercise by ID
app.get('/exercises/:id', (c) => {
  const id = c.req.param('id');
  const exercise = EXERCISE_LIBRARY.find(e => e.id === id);
  
  if (!exercise) {
    return c.json({ success: false, error: 'Exercise not found' }, 404);
  }
  
  return c.json({ success: true, data: exercise });
});

// ============================================================================
// PRESCRIPTION ROUTES
// ============================================================================

// Create prescription
app.post('/prescriptions', async (c) => {
  const body = await c.req.json();
  const { patient_id, exercise_id, assessment_id, sets, reps, frequency, notes } = body;
  
  const exercise = EXERCISE_LIBRARY.find(e => e.id === exercise_id);
  if (!exercise) {
    return c.json({ success: false, error: 'Exercise not found' }, 404);
  }
  
  const prescription = {
    id: Date.now(),
    exercise_id,
    exercise_name: exercise.name,
    sets,
    reps,
    frequency,
    notes,
    created_at: new Date().toISOString()
  };
  
  // Add to patient's assessment
  if (assessment_id) {
    for (const patient of MOCK_PATIENTS) {
      const assessment = patient.assessments?.find((a: any) => a.id === assessment_id);
      if (assessment) {
        if (!assessment.prescriptions) assessment.prescriptions = [];
        assessment.prescriptions.push(prescription);
        break;
      }
    }
  }
  
  return c.json({ success: true, data: prescription }, 201);
});

// ============================================================================
// SESSION ROUTES
// ============================================================================

// Record exercise session
app.post('/exercise-sessions', async (c) => {
  const body = await c.req.json();
  const { patient_id, exercises_completed, duration_minutes, pain_level, notes } = body;
  
  const patientIndex = MOCK_PATIENTS.findIndex(p => p.id === patient_id);
  if (patientIndex === -1) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  const session = {
    id: Date.now(),
    session_date: new Date().toISOString(),
    exercises_completed,
    duration_minutes,
    pain_level,
    notes,
    adherence: pain_level <= 3 ? 'excellent' : pain_level <= 5 ? 'good' : 'fair'
  };
  
  if (!MOCK_PATIENTS[patientIndex].exercise_sessions) {
    MOCK_PATIENTS[patientIndex].exercise_sessions = [];
  }
  MOCK_PATIENTS[patientIndex].exercise_sessions.push(session as any);
  
  // Update progress metrics
  if (!MOCK_PATIENTS[patientIndex].progress_metrics) {
    MOCK_PATIENTS[patientIndex].progress_metrics = { pain_trend: [] };
  }
  if (!MOCK_PATIENTS[patientIndex].progress_metrics.pain_trend) {
    MOCK_PATIENTS[patientIndex].progress_metrics.pain_trend = [];
  }
  MOCK_PATIENTS[patientIndex].progress_metrics.pain_trend.push(pain_level);
  
  return c.json({ success: true, data: session }, 201);
});

// ============================================================================
// ANALYTICS & REPORTING
// ============================================================================

// Get dashboard stats
app.get('/dashboard/stats', (c) => {
  console.log('[PHYSIOMOTION] Dashboard stats endpoint hit');
  const stats = {
    total_patients: MOCK_PATIENTS.length,
    active_patients: MOCK_PATIENTS.filter(p => {
      const latest = p.assessments?.[p.assessments.length - 1];
      if (!latest) return false;
      const daysSince = (Date.now() - new Date(latest.assessment_date).getTime()) / (1000 * 60 * 60 * 24);
      return daysSince < 30;
    }).length,
    total_assessments: MOCK_PATIENTS.reduce((acc, p) => acc + (p.assessments?.length || 0), 0),
    total_sessions: MOCK_PATIENTS.reduce((acc, p) => acc + (p.exercise_sessions?.length || 0), 0),
    recent_activity: MOCK_PATIENTS.flatMap(p => 
      (p.exercise_sessions || []).map(s => ({
        patient_name: `${p.first_name} ${p.last_name}`,
        date: s.session_date,
        duration: s.duration_minutes,
        exercises: s.exercises_completed?.length || 0
      }))
    ).sort((a: any, b: any) => new Date(b.date).getTime() - new Date(a.date).getTime()).slice(0, 5)
  };
  
  return c.json({ success: true, data: stats });
});

// Get patient progress
app.get('/patients/:id/progress', (c) => {
  const id = parseInt(c.req.param('id'));
  const patient = MOCK_PATIENTS.find(p => p.id === id);
  
  if (!patient) {
    return c.json({ success: false, error: 'Patient not found' }, 404);
  }
  
  const progress = {
    pain_trend: patient.progress_metrics?.pain_trend || [],
    functional_trend: patient.progress_metrics?.functional_score_trend || [],
    sessions_completed: patient.exercise_sessions?.length || 0,
    total_exercise_time: patient.exercise_sessions?.reduce((acc: number, s: { duration_minutes?: number }) => acc + (s.duration_minutes || 0), 0) || 0,
    adherence_rate: patient.exercise_sessions?.length > 0 ? 85 : 0, // Mock calculation
    latest_pain: patient.assessments?.[patient.assessments.length - 1]?.pain_scale || 0
  };
  
  return c.json({ success: true, data: progress });
});

// ============================================================================
// AUTH ROUTES (Demo Mode)
// ============================================================================

app.post('/auth/login', async (c) => {
  const body = await c.req.json();
  
  // Demo mode - accept any credentials
  return c.json({
    success: true,
    data: {
      user: {
        id: 1,
        name: 'Dr. Demo Clinician',
        email: body.email || 'demo@physiomotion.com',
        role: 'clinician',
        license: 'PT12345'
      },
      token: 'demo-token-12345',
      demo_mode: true
    }
  });
});

app.get('/auth/me', (c) => {
  return c.json({
    success: true,
    data: {
      user: {
        id: 1,
        name: 'Dr. Demo Clinician',
        email: 'demo@physiomotion.com',
        role: 'clinician',
        license: 'PT12345'
      },
      demo_mode: true
    }
  });
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

app.onError((err, c) => {
  console.error('Error:', err);
  return c.json({
    success: false,
    error: 'Internal server error',
    message: err.message
  }, 500);
});

app.notFound((c) => {
  return c.json({
    success: false,
    error: 'Not found',
    path: c.req.path
  }, 404);
});

export default app;
