import { handle } from 'hono/vercel'
import app from '../src/index'

// Debug route - verify function is running
app.get('/test', (c) => c.json({ ok: true, msg: 'function is running' }))

export default handle(app)
