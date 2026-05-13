import { Hono } from 'hono'
import { handle } from 'hono/vercel'
import { cors } from 'hono/cors'

const app = new Hono()
app.use('*', cors({ origin: '*', allowHeaders: ['Content-Type', 'Authorization'] }))

app.get('/', (c) => c.json({ success: true, status: 'healthy', timestamp: new Date().toISOString(), version: '2.0.0-demo' }))
app.get('/health', (c) => c.json({ success: true, status: 'healthy', ts: new Date().toISOString() }))
app.get('/test', (c) => c.json({ ok: true, msg: 'Vercel serverless works', ts: new Date().toISOString() }))

// Info
app.get('/info', (c) => c.json({
  success: true,
  model: {
    name: 'phi4-mini:3.8b / deepseek-r1:7b',
    description: 'Boxer3D multi-agent MSK assessment',
    specialists: [
      { id: 'upper-extremity', name: 'Upper Extremity & Spine Specialist' },
      { id: 'lower-extremity', name: 'Lower Extremity & Gait Specialist' },
      { id: 'spine-core', name: 'Spine, Core & Posture Specialist' },
      { id: 'auditor', name: 'Clinical Auditor & Scribe' }
    ]
  }
}))

// Mock patients
const MOCK_PATIENTS = [
  { id: 1, first_name: 'John', last_name: 'Smith', date_of_birth: '1979-05-15', gender: 'male', email: 'john.smith@email.com', phone: '(305) 555-0123', created_at: '2025-01-15T10:30:00Z' },
  { id: 2, first_name: 'Maria', last_name: 'Garcia', date_of_birth: '1963-08-22', gender: 'female', email: 'maria.garcia@email.com', phone: '(305) 555-0234', created_at: '2025-02-20T11:00:00Z' },
  { id: 3, first_name: 'David', last_name: 'Chen', date_of_birth: '1987-03-10', gender: 'male', email: 'david.chen@email.com', phone: '(305) 555-0345', created_at: '2025-03-05T09:30:00Z' }
]

app.get('/patients', (c) => c.json({ success: true, data: MOCK_PATIENTS }))
app.get('/patients/:id', (c) => {
  const p = MOCK_PATIENTS.find(p => p.id === parseInt(c.req.param('id')))
  return p ? c.json({ success: true, data: p }) : c.json({ success: false, error: 'Not found' }, 404)
})

app.get('/dashboard/stats', (c) => c.json({
  success: true, data: {
    total_patients: MOCK_PATIENTS.length,
    active_patients: MOCK_PATIENTS.filter(p => {
      const d = new Date(p.created_at)
      return (Date.now() - d.getTime()) / 86400000 < 30
    }).length
  }
}))

export default handle(app)
