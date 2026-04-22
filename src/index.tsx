// Vercel Serverless API Routes
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'
import { createClient } from '@supabase/supabase-js'

// Initialize Supabase client
const supabaseUrl = process.env.VITE_SUPABASE_URL || ''
const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY || ''
const supabase = supabaseUrl && supabaseKey ? createClient(supabaseUrl, supabaseKey) : null

const app = new Hono()

// CORS
app.use('*', cors({
  origin: '*',
  credentials: true
}))

// Health check
app.get('/api/health', (c) => {
  return c.json({ status: 'ok', timestamp: new Date().toISOString() })
})

// Auth - Login
app.post('/api/auth/login', async (c) => {
  try {
    const { email, password } = await c.req.json()
    
    if (!supabase) {
      // Demo mode - return demo user
      if (email === 'demo@physiomotion.com' || email === 'demo@test.com') {
        return c.json({
          success: true,
          data: { id: 1, email: 'demo@physiomotion.com', first_name: 'Demo', last_name: 'User', role: 'clinician' }
        })
      }
      return c.json({ success: false, error: 'Demo mode - use demo@physiomotion.com / demo123' }, 401)
    }
    
    // Real Supabase auth
    const { data: { user }, error } = await supabase.auth.signInWithPassword({ email, password })
    
    if (error || !user) {
      return c.json({ success: false, error: 'Invalid credentials' }, 401)
    }
    
    // Get user metadata
    const { data: clinician } = await supabase
      .from('clinicians')
      .select('*')
      .eq('email', email)
      .single()
    
    return c.json({ success: true, data: clinician })
  } catch (error: any) {
    // Demo fallback
    if (email === 'demo@physiomotion.com' || email === 'demo@test.com') {
      return c.json({
        success: true,
        data: { id: 1, email: 'demo@physiomotion.com', first_name: 'Demo', last_name: 'User', role: 'clinician' }
      })
    }
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Auth - Register
app.post('/api/auth/register', async (c) => {
  try {
    const { email, password, first_name, last_name } = await c.req.json()
    
    if (!supabase) {
      return c.json({ success: true, data: { id: Math.floor(Math.random() * 1000) } })
    }
    
    const { data: { user }, error } = await supabase.auth.signUp({ email, password })
    
    if (error) return c.json({ success: false, error: error.message }, 400)
    
    // Create clinician record
    const { data: clinician } = await supabase
      .from('clinicians')
      .insert({ email, first_name, last_name, password_hash: 'supabase-auth' })
      .select()
      .single()
    
    return c.json({ success: true, data: clinician })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Patients - List
app.get('/api/patients', async (c) => {
  try {
    if (!supabase) {
      // Demo patients
      return c.json({
        success: true,
        data: [
          { id: 1, first_name: 'John', last_name: 'Demo', date_of_birth: '1985-03-15', gender: 'male', email: 'john@demo.com' },
          { id: 2, first_name: 'Jane', last_name: 'Demo', date_of_birth: '1990-07-22', gender: 'female', email: 'jane@demo.com' },
          { id: 3, first_name: 'Bob', last_name: 'Test', date_of_birth: '1978-11-08', gender: 'male', email: 'bob@demo.com' }
        ]
      })
    }
    
    const { data: patients, error } = await supabase
      .from('patients')
      .select('*')
      .order('created_at', { ascending: false })
    
    if (error) return c.json({ success: false, error: error.message }, 400)
    
    return c.json({ success: true, data: patients || [] })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Patients - Create
app.post('/api/patients', async (c) => {
  try {
    const patient = await c.req.json()
    
    if (!supabase) {
      return c.json({ success: true, data: { id: Math.floor(Math.random() * 1000), ...patient } })
    }
    
    const { data, error } = await supabase
      .from('patients')
      .insert(patient)
      .select()
      .single()
    
    if (error) return c.json({ success: false, error: error.message }, 400)
    
    return c.json({ success: true, data })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Patients - Get single
app.get('/api/patients/:id', async (c) => {
  const id = c.req.param('id')
  
  try {
    if (!supabase) {
      return c.json({
        success: true,
        data: { id: parseInt(id), first_name: 'Demo', last_name: 'Patient', gender: 'male' }
      })
    }
    
    const { data, error } = await supabase
      .from('patients')
      .select('*')
      .eq('id', id)
      .single()
    
    if (error) return c.json({ success: false, error: 'Not found' }, 404)
    
    return c.json({ success: true, data })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Exercises - List
app.get('/api/exercises', async (c) => {
  try {
    if (!supabase) {
      return c.json({
        success: true,
        data: [
          { id: 1, exercise_name: 'Deep Squat', exercise_category: 'mobility', difficulty_level: 'intermediate', description: 'Full squat with proper form' },
          { id: 2, exercise_name: 'Hip Flexor Stretch', exercise_category: 'flexibility', difficulty_level: 'beginner', description: 'Kneeling hip flexor stretch' },
          { id: 3, exercise_name: 'Single Leg Balance', exercise_category: 'balance', difficulty_level: 'intermediate', description: 'Balance on one leg' },
          { id: 4, exercise_name: 'Plank Hold', exercise_category: 'stability', difficulty_level: 'beginner', description: 'Core stabilization' },
          { id: 5, exercise_name: 'Lunge Pattern', exercise_category: 'functional', difficulty_level: 'intermediate', description: 'Walking lunges' },
          { id: 6, exercise_name: 'Dead Bug', exercise_category: 'stability', difficulty_level: 'intermediate', description: 'Core coordination' },
          { id: 7, exercise_name: 'Bird Dog', exercise_category: 'stability', difficulty_level: 'beginner', description: 'Quadruped stability' },
          { id: 8, exercise_name: 'Shoulder Mobility', exercise_category: 'mobility', difficulty_level: 'beginner', description: 'Arm circles and sweeps' }
        ]
      })
    }
    
    const { data: exercises, error } = await supabase
      .from('exercises')
      .select('*')
      .order('exercise_name')
    
    if (error) return c.json({ success: false, error: error.message }, 400)
    
    return c.json({ success: true, data: exercises || [] })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Assessments - List
app.get('/api/assessments', async (c) => {
  try {
    const patientId = c.req.query('patient_id')
    
    if (!supabase) {
      return c.json({ success: true, data: [] })
    }
    
    let query = supabase.from('assessments').select('*').order('created_at', { ascending: false })
    
    if (patientId) {
      query = query.eq('patient_id', patientId)
    }
    
    const { data, error } = await query
    
    if (error) return c.json({ success: false, error: error.message }, 400)
    
    return c.json({ success: true, data: data || [] })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Assessments - Create
app.post('/api/assessments', async (c) => {
  try {
    const assessment = await c.req.json()
    
    if (!supabase) {
      return c.json({ success: true, data: { id: Math.floor(Math.random() * 1000), ...assessment } })
    }
    
    const { data, error } = await supabase
      .from('assessments')
      .insert(assessment)
      .select()
      .single()
    
    if (error) return c.json({ success: false, error: error.message }, 400)
    
    return c.json({ success: true, data })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Billing Codes
app.get('/api/billing/codes', async (c) => {
  try {
    if (!supabase) {
      return c.json({
        success: true,
        data: [
          { id: 1, cpt_code: '97001', code_description: 'Physical therapy evaluation', code_category: 'evaluation' },
          { id: 2, cpt_code: '97110', code_description: 'Therapeutic exercise', code_category: 'exercise' },
          { id: 3, cpt_code: '97010', code_description: 'Therapeutic exercise', code_category: 'treatment' }
        ]
      })
    }
    
    const { data, error } = await supabase
      .from('billing_codes')
      .select('*')
      .order('cpt_code')
    
    if (error) return c.json({ success: false, error: error.message }, 400)
    
    return c.json({ success: true, data: data || [] })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Biomechanical Analysis (simplified for demo)
app.post('/api/analyze-movement', async (c) => {
  try {
    const { skeletonData } = await c.req.json()
    
    // Simplified analysis
    const score = Math.floor(Math.random() * 30) + 70 // 70-100
    
    return c.json({
      success: true,
      data: {
        joint_angles: [
          { joint_name: 'Left Shoulder', left_angle: 150, status: 'normal' },
          { joint_name: 'Right Shoulder', right_angle: 148, status: 'normal' },
          { joint_name: 'Left Hip', left_angle: 95, status: 'normal' },
          { joint_name: 'Right Hip', right_angle: 92, status: 'normal' },
          { joint_name: 'Left Knee', left_angle: 125, status: 'normal' },
          { joint_name: 'Right Knee', right_angle: 122, status: 'normal' }
        ],
        movement_quality_score: score,
        detected_compensations: score < 85 ? ['Minor asymmetry detected'] : [],
        recommendations: [
          score >= 85 ? 'Excellent performance - maintain current routine' : 'Focus on hip mobility exercises',
          'Continue prescribed exercise program',
          'Re-assess in 2-3 weeks'
        ],
        deficiencies: []
      }
    })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

// Serve static files
app.get('*', serveStatic({ root: './public', index: 'login.html' }))

export default app
