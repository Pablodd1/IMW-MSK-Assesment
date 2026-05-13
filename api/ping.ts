import { handle } from 'hono/vercel'
import { Hono } from 'hono'

const app = new Hono()

app.get('/health', (c) => c.json({ success: true, status: 'healthy' }))
app.get('/*', (c) => {
  return c.json({ success: true, path: c.req.path, message: 'IMW MSK Assessment is running!' })
})

export default handle(app)
