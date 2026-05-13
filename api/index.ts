import { handle } from 'hono/vercel'
import { Hono } from 'hono'
import { cors } from 'hono/cors'

const app = new Hono()
app.use('*', cors({ origin: '*', allowHeaders: ['Content-Type', 'Authorization'] }))

app.get('/api/health', (c) => c.json({ success: true, status: 'healthy', ts: new Date().toISOString() }))
app.get('/api/test', (c) => c.json({ ok: true, msg: 'Vercel works', ts: new Date().toISOString() }))
app.get('/api/info', (c) => c.json({ success: true, model: { name: 'Boxer3D MSK' } }))

// Direct paths (without /api prefix)
app.get('/health', (c) => c.json({ success: true, status: 'healthy', ts: new Date().toISOString() }))
app.get('/test', (c) => c.json({ ok: true, msg: 'Vercel works', ts: new Date().toISOString() }))
app.get('/info', (c) => c.json({ success: true, model: { name: 'Boxer3D MSK' } }))

export default handle(app)
