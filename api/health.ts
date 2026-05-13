import { handle } from 'hono/vercel'
import { Hono } from 'hono'
import { cors } from 'hono/cors'

const app = new Hono()
app.use('*', cors({ origin: '*', allowHeaders: ['Content-Type', 'Authorization'] }))

app.get('/', (c) => c.json({ success: true, status: 'healthy', ts: new Date().toISOString(), version: '2.0.0-demo' }))
app.get('/test', (c) => c.json({ ok: true, msg: 'Vercel serverless works', ts: new Date().toISOString() }))

export default handle(app)
