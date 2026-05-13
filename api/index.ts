import { handle } from 'hono/vercel'
import { Hono } from 'hono'
import { cors } from 'hono/cors'

const app = new Hono()
app.use('*', cors({ origin: '*', allowHeaders: ['Content-Type', 'Authorization'] }))

app.get('/api/health', (c) => c.json({ success: true, status: 'healthy', ts: new Date().toISOString() }))
app.get('/api/test', (c) => c.json({ ok: true, msg: 'Vercel serverless works', ts: new Date().toISOString() }))
app.get('/api/info', (c) => c.json({ success: true, model: { name: 'Boxer3D MSK', specialists: ['Upper Extremity', 'Lower Extremity', 'Spine/Core', 'Auditor'] } }))
app.get('/api/patients', (c) => c.json({ success: true, data: [
  { id: 1, first_name: 'John', last_name: 'Smith' },
  { id: 2, first_name: 'Maria', last_name: 'Garcia' }
]}))

// Also handle without /api prefix for local dev
app.get('/health', (c) => c.json({ success: true, status: 'healthy', ts: new Date().toISOString() }))
app.get('/test', (c) => c.json({ ok: true, msg: 'Vercel works', ts: new Date().toISOString() }))

export default handle(app)
