import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'
import type { Bindings } from './types'
import { performBiomechanicalAnalysis } from './utils/biomechanics'
import { queryExerciseKnowledge } from './utils/rag'
import { medicalAI } from './utils/gemini'
import authRoutes from './routes/auth'
import patientRoutes from './routes/patients'
import assessmentRoutes from './routes/assessments'
import exerciseRoutes from './routes/exercises'
import billingRoutes from './routes/billing'
import { errorHandler } from './middleware/error'

const app = new Hono<{ Bindings: Bindings }>()

// Global Error Handler
app.use('*', errorHandler)

const corsOptions = {
  origin: (origin: string) => {
    const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || []
    const isAllowed = allowedOrigins.length === 0 || allowedOrigins.some(o => o.trim() === origin || o.trim() === '*')
    return isAllowed ? origin : null
  },
  credentials: true,
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  maxAge: 86400,
}

app.use('/api/*', cors(corsOptions))

app.use('/static/*', serveStatic({ root: './public', manifest: {} }))

// Security headers
app.use('*', async (c, next) => {
  await next()
  c.res.headers.set('X-Content-Type-Options', 'nosniff')
  c.res.headers.set('X-Frame-Options', 'DENY')
  c.res.headers.set('X-XSS-Protection', '1; mode=block')
})

app.route('/api/auth', authRoutes)
app.route('/api/patients', patientRoutes)
app.route('/api/assessments', assessmentRoutes)
app.route('/api', exerciseRoutes)
app.route('/api/billing', billingRoutes)

// Real-time joint tracking API
app.post('/api/video/analyze', async (c) => {
  try {
    const { landmarks, patientId } = await c.req.json()
    const analysis = performBiomechanicalAnalysis(landmarks)
    
    // If Gemini API key exists, get AI analysis
    if (process.env.GEMINI_API_KEY && medicalAI.instance) {
      const aiInsights = await medicalAI.instance.analyzeBiomechanics(analysis, { patientId })
      return c.json({ success: true, data: { analysis, aiInsights } })
    }
    
    return c.json({ success: true, data: analysis })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

app.post('/api/analyze-movement', async (c) => {
  try {
    const { skeletonData, testType } = await c.req.json()
    const analysis = await performBiomechanicalAnalysis(skeletonData, testType)
    return c.json({ success: true, data: analysis })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

app.post('/api/rag/query', async (c) => {
  try {
    const { query, patientId } = await c.req.json()
    const results = await queryExerciseKnowledge(query, patientId)
    return c.json({ success: true, data: results })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

app.get('/api/health', (c) => {
  return c.json({ status: 'ok', timestamp: new Date().toISOString() })
})

// Camera/device detection API
app.get('/api/cameras', async (c) => {
  try {
    // This works in browser context via POST request
    return c.json({ 
      success: true, 
      data: {
        supported: ['webcam', 'femto-mega', 'azure-kinect', 'orbbec'],
        note: 'Camera access requires frontend request'
      } 
    })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

app.get('*', serveStatic({ root: './public', index: 'index.html' }))

export default app
