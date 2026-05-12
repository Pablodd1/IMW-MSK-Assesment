import { handle } from 'hono/vercel'
import { Hono } from 'hono'
import app from '../src/index.js'

// Debug route at root level - should be at /test when accessed via /api/test
app.get('/test', (c) => c.json({ ok: true, msg: 'direct test route works' }))

export default handle(app)
