import { handle } from 'hono/vercel'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { logger as honoLogger } from 'hono/logger'

const app = new Hono()

// Middleware
app.use('*', honoLogger())
app.use('*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization']
}))

// ─── Mock Data ───
const OLLAMA_BASE = process.env.OLLAMA_HOST || 'http://localhost:11434'

const ROM: Record<string, Record<string, [number, number]>> = {
  shoulder: { flexion: [150,180], extension: [45,60], abduction: [150,180], ext_rotation: [80,90], int_rotation: [60,70] },
  elbow: { flexion: [130,150], extension: [0,10] },
  hip: { flexion: [110,130], extension: [10,30], abduction: [30,50] },
  knee: { flexion: [120,150], extension: [0,10] },
  ankle: { dorsiflexion: [10,20], plantarflexion: [30,50] },
  cervical: { flexion: [40,60], extension: [40,75], rotation: [60,80], lateral_flexion: [30,45] },
  lumbar: { flexion: [40,60], extension: [15,30], lateral_flexion: [15,25] },
}

// ─── Health ───
app.get('/health', (c) => {
  return c.json({
    success: true, status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '2.0.0-demo',
    features: { patients: true, assessments: true, exercises: true, camera: true, reporting: true }
  })
})

// ─── Info ───
app.get('/info', (c) => {
  return c.json({
    success: true,
    model: {
      name: 'phi4-mini:3.8b / deepseek-r1:7b',
      description: 'Boxer3D multi-agent MSK assessment',
      specialists: [
        { id: 'upper-extremity', name: 'Upper Extremity & Spine Specialist', persona: 'orthopedic surgeon' },
        { id: 'lower-extremity', name: 'Lower Extremity & Gait Specialist', persona: 'sports medicine physician' },
        { id: 'spine-core', name: 'Spine, Core & Posture Specialist', persona: 'chiropractic and rehabilitation physician' },
        { id: 'auditor', name: 'Clinical Auditor & Scribe', persona: 'senior physiatrist' }
      ]
    }
  })
})

// ─── AI Analysis ───
async function ollamaChat(model: string, prompt: string): Promise<string> {
  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, prompt, stream: false, keep_alive: '5m', temperature: 0.2 }),
  })
  if (!res.ok) throw new Error(`Ollama error: ${res.status}`)
  const data = await res.json() as any
  return data.response || ''
}

app.post('/ai/analyze-swarm', async (c) => {
  try {
    const body = await c.req.json() as any
    const { keypoints, abnormalities, body_regions } = body
    if (!keypoints || keypoints.length === 0) {
      return c.json({ success: false, error: 'Keypoint data required' }, 400)
    }
    const regions = body_regions || ['upper', 'lower', 'spine']
    return c.json({
      success: true,
      swarm: {
        specialists: [],
        auditor: {
          subjective: 'Swarm analysis endpoint ready',
          objective: 'Connect Ollama to enable AI analysis',
          assessment: 'Endpoint active, model needs local connection',
          plan: 'Deploy Ollama server alongside this app'
        },
        model: 'phi4-mini:3.8b (connect Ollama for live analysis)'
      },
      timestamp: new Date().toISOString()
    })
  } catch (error: any) {
    console.error('[SWARM] Error:', error)
    return c.json({ success: false, error: 'Swarm analysis failed' }, 500)
  }
})

app.post('/ai/analyze', async (c) => {
  try {
    const body = await c.req.json() as any
    const assessmentData = body.assessmentData || body
    if (!assessmentData) {
      return c.json({ success: false, error: 'Assessment data is required' }, 400)
    }
    return c.json({
      success: true,
      analysis: {
        summary: 'AI analysis endpoint ready. Connect Ollama for live inference.',
        findings: ['Endpoint active'],
        recommendations: ['Deploy Ollama server alongside this app'],
        confidence: 0.5,
        model: 'phi4-mini:3.8b (RAG enhanced)'
      }
    })
  } catch (error: any) {
    console.error('[MSK-AI] Error:', error)
    return c.json({ success: false, error: 'Analysis failed' }, 500)
  }
})

// ─── Patient Routes ───
const MOCK_PATIENTS: any[] = [
  {
    id: 1, first_name: 'John', last_name: 'Smith', date_of_birth: '1979-05-15', gender: 'male',
    email: 'john.smith@email.com', phone: '(305) 555-0123',
    created_at: '2025-01-15T10:30:00Z',
    medical_history: {
      conditions: [{ condition: 'Chronic Lower Back Pain', diagnosed_date: '2019-03-10', status: 'active' }],
      medications: [{ name: 'Ibuprofen', dosage: '400mg', frequency: 'As needed', reason: 'Pain' }],
      allergies: ['Penicillin'],
      surgeries: [{ procedure: 'Lumbar Microdiscectomy L4-L5', date: '2020-11-15', outcome: 'Improved' }]
    },
    assessments: [
      {
        id: 101, assessment_type: 'Initial Evaluation', assessment_date: '2025-01-15T10:30:00Z',
        clinician: 'Dr. Sarah Johnson, PT',
        chief_complaint: 'Lower back pain radiating to right leg, pain level 7/10',
        pain_scale: 7,
        functional_status: 'Limited household ambulation',
        goals: ['Reduce pain to 3/10', 'Return to work', 'Resume walking 30 min daily'],
        clinical_notes: 'Chronic LBP with radicular symptoms. Core weakness identified.',
        tests: [
          { test_name: 'Lumbar ROM', results: { flexion: { value: 45, normal: 60, unit: 'degrees', status: 'limited' } } }
        ],
        prescriptions: [
          { exercise_id: 'core_01', exercise_name: 'Dead Bug', sets: 3, reps: 10, frequency: 'Daily' }
        ]
      }
    ],
    exercise_sessions: [],
    progress_metrics: { pain_trend: [7, 6, 5, 4, 4] }
  },
  {
    id: 2, first_name: 'Maria', last_name: 'Garcia', date_of_birth: '1963-08-22', gender: 'female',
    email: 'maria.garcia@email.com', phone: '(305) 555-0234',
    created_at: '2025-02-20T11:00:00Z',
    medical_history: {
      conditions: [
        { condition: 'Total Knee Replacement (Right)', diagnosed_date: '2024-12-10', status: 'post_op' },
        { condition: 'Osteoarthritis (Left Knee)', diagnosed_date: '2020-01-15', status: 'active' }
      ],
      medications: [{ name: 'Metformin', dosage: '500mg', frequency: 'Twice daily', reason: 'Diabetes' }],
      allergies: ['Codeine'],
      surgeries: [{ procedure: 'Total Knee Arthroplasty (Right)', date: '2024-12-10', outcome: 'Recovering' }]
    },
    assessments: [
      {
        id: 201, assessment_type: 'Post-Op Evaluation', assessment_date: '2025-02-20T11:00:00Z',
        clinician: 'Dr. Michael Chen, PT',
        chief_complaint: 'Post TKA rehabilitation, 10 weeks post-op',
        pain_scale: 4,
        functional_status: 'Ambulating with cane',
        goals: ['Achieve 120° knee flexion', 'Ambulate without assistive device'],
        clinical_notes: '10 weeks post-op TKA. Quadriceps weakness noted (3/5 MMT).',
        tests: [
          { test_name: 'Knee ROM', results: { right_flexion: { value: 105, normal: 135, unit: 'degrees', status: 'limited' } } }
        ],
        prescriptions: [
          { exercise_id: 'knee_01', exercise_name: 'Heel Slides', sets: 3, reps: 15, frequency: '3x daily' },
          { exercise_id: 'quad_01', exercise_name: 'Quad Sets', sets: 3, reps: 15, frequency: '3x daily' }
        ]
      }
    ],
    exercise_sessions: [],
    progress_metrics: { pain_trend: [5, 5, 4, 4, 4] }
  },
  {
    id: 3, first_name: 'David', last_name: 'Chen', date_of_birth: '1987-03-10', gender: 'male',
    email: 'david.chen@email.com', phone: '(305) 555-0345',
    created_at: '2025-03-05T09:30:00Z',
    medical_history: {
      conditions: [
        { condition: 'Shoulder Impingement Syndrome', diagnosed_date: '2025-01-20', status: 'active' },
        { condition: 'Rotator Cuff Tendinopathy', diagnosed_date: '2025-01-20', status: 'active' }
      ],
      medications: [{ name: 'Meloxicam', dosage: '15mg', frequency: 'Daily', reason: 'Anti-inflammatory' }],
      allergies: [],
      surgeries: [],
      occupation: 'Software Engineer',
      activities: ['Weight lifting', 'Swimming']
    },
    assessments: [
      {
        id: 301, assessment_type: 'Initial Evaluation', assessment_date: '2025-03-05T09:30:00Z',
        clinician: 'Dr. Lisa Wong, PT',
        chief_complaint: 'Right shoulder pain, worse with overhead activity',
        pain_scale: 5,
        functional_status: 'Unable to lift overhead, sleep disrupted',
        goals: ['Return to full overhead ROM', 'Resume swimming'],
        clinical_notes: 'Impingement presentation. Postural dysfunction, scapular dyskinesis.',
        tests: [
          { test_name: 'Shoulder ROM', results: { right_flexion: { value: 150, normal: 180, unit: 'degrees', status: 'limited' } } }
        ],
        prescriptions: [
          { exercise_id: 'shoulder_01', exercise_name: 'Sleeper Stretch', sets: 2, reps: 30, frequency: 'Daily' },
          { exercise_id: 'scap_01', exercise_name: 'Scapular Retraction', sets: 3, reps: 15, frequency: 'Daily' }
        ]
      }
    ],
    exercise_sessions: [],
    progress_metrics: { pain_trend: [5] }
  }
]

const EXERCISE_LIBRARY: any[] = [
  { id: 'core_01', name: 'Dead Bug', category: 'core', difficulty: 'beginner', description: 'Core stability exercise', muscle_groups: ['abs', 'hip flexors'] },
  { id: 'knee_01', name: 'Heel Slides', category: 'knee', difficulty: 'beginner', description: 'Knee flexion mobility', muscle_groups: ['knee', 'quadriceps'] },
  { id: 'quad_01', name: 'Quad Sets', category: 'knee', difficulty: 'beginner', description: 'Isometric quad strengthening', muscle_groups: ['quadriceps'] },
  { id: 'shoulder_01', name: 'Sleeper Stretch', category: 'shoulder', difficulty: 'beginner', description: 'Posterior capsule stretch', muscle_groups: ['shoulder'] },
  { id: 'scap_01', name: 'Scapular Retraction', category: 'shoulder', difficulty: 'beginner', description: 'Scapular stabilization', muscle_groups: ['scapular', 'upper back'] },
  { id: 'glute_01', name: 'Glute Bridge', category: 'hip', difficulty: 'beginner', description: 'Glute activation', muscle_groups: ['glutes', 'hamstrings'] },
  { id: 'hip_01', name: '90/90 Hip Stretch', category: 'hip', difficulty: 'intermediate', description: 'Hip mobility stretch', muscle_groups: ['hip rotators'] },
  { id: 'rotator_01', name: 'External Rotation with Band', category: 'shoulder', difficulty: 'intermediate', description: 'Rotator cuff strengthening', muscle_groups: ['rotator cuff'] },
  { id: 'posture_01', name: 'Thoracic Extension over Foam Roller', category: 'posture', difficulty: 'beginner', description: 'Thoracic spine mobility', muscle_groups: ['thoracic spine'] },
  { id: 'func_01', name: 'Sit to Stand', category: 'functional', difficulty: 'beginner', description: 'Functional lower body strength', muscle_groups: ['quadriceps', 'glutes'] }
]

// ─── CRUD routes ───
app.get('/patients', (c) => {
  return c.json({ success: true, data: MOCK_PATIENTS.map(p => ({
    ...p,
    latest_assessment: p.assessments?.[p.assessments.length - 1]?.assessment_date || null
  })) })
})

app.get('/patients/:id', (c) => {
  const id = parseInt(c.req.param('id'))
  const patient = MOCK_PATIENTS.find(p => p.id === id)
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404)
  return c.json({ success: true, data: patient })
})

app.get('/patients/:id/medical-history', (c) => {
  const id = parseInt(c.req.param('id'))
  const patient = MOCK_PATIENTS.find(p => p.id === id)
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404)
  return c.json({ success: true, data: patient.medical_history || {} })
})

app.get('/patients/:id/assessments', (c) => {
  const id = parseInt(c.req.param('id'))
  const patient = MOCK_PATIENTS.find(p => p.id === id)
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404)
  return c.json({ success: true, data: patient.assessments || [] })
})

app.get('/patients/:id/prescriptions', (c) => {
  const id = parseInt(c.req.param('id'))
  const patient = MOCK_PATIENTS.find(p => p.id === id)
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404)
  return c.json({ success: true, data: (patient.assessments || []).flatMap((a: any) => (a.prescriptions || []).map((p: any) => ({ ...p, assessment_id: a.id }))) })
})

app.get('/patients/:id/sessions', (c) => {
  const id = parseInt(c.req.param('id'))
  const patient = MOCK_PATIENTS.find(p => p.id === id)
  if (!patient) return c.json({ success: false, error: 'Patient not found' }, 404)
  return c.json({ success: true, data: patient.exercise_sessions || [] })
})

app.get('/exercises', (c) => {
  const category = c.req.query('category')
  let exercises = EXERCISE_LIBRARY
  if (category) exercises = exercises.filter(e => e.category === category)
  return c.json({ success: true, data: exercises })
})

app.get('/exercises/:id', (c) => {
  const exercise = EXERCISE_LIBRARY.find(e => e.id === c.req.param('id'))
  if (!exercise) return c.json({ success: false, error: 'Exercise not found' }, 404)
  return c.json({ success: true, data: exercise })
})

app.get('/dashboard/stats', (c) => {
  return c.json({
    success: true,
    data: {
      total_patients: MOCK_PATIENTS.length,
      active_patients: MOCK_PATIENTS.filter(p => {
        const latest = p.assessments?.[p.assessments.length - 1]
        if (!latest) return false
        return (Date.now() - new Date(latest.assessment_date).getTime()) / (1000 * 60 * 60 * 24) < 30
      }).length,
      total_assessments: MOCK_PATIENTS.reduce((acc, p) => acc + (p.assessments?.length || 0), 0),
      total_exercises: EXERCISE_LIBRARY.length
    }
  })
})

// Debug route
app.get('/test', (c) => c.json({ ok: true, msg: 'Vercel serverless running', ts: new Date().toISOString() }))

export default handle(app)
