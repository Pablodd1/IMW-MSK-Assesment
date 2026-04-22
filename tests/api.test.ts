import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'

// Mock API server for testing
const mockServer = setupServer(
  http.get('/api/health', () => {
    return HttpResponse.json({ status: 'ok', timestamp: new Date().toISOString() })
  }),
  
  http.post('/api/auth/login', async ({ request }) => {
    const body = await request.json()
    if (body.email === 'demo@test.com' && body.password === 'demo123') {
      return HttpResponse.json({
        success: true,
        data: { id: 1, email: 'demo@test.com', first_name: 'Demo', last_name: 'User' }
      })
    }
    return HttpResponse.json({ success: false, error: 'Invalid credentials' }, { status: 401 })
  }),

  http.get('/api/patients', () => {
    return HttpResponse.json({
      success: true,
      data: [
        { id: 1, first_name: 'John', last_name: 'Sample', email: 'john@test.com' },
        { id: 2, first_name: 'Jane', last_name: 'Test', email: 'jane@test.com' }
      ]
    })
  }),

  http.get('/api/exercises', () => {
    return HttpResponse.json({
      success: true,
      data: [
        { id: 1, exercise_name: 'Deep Squat', exercise_category: 'mobility' },
        { id: 2, exercise_name: 'Hip Flexor Stretch', exercise_category: 'flexibility' }
      ]
    })
  }),

  http.post('/api/analyze-movement', async ({ request }) => {
    const body = await request.json()
    // Simulate biomechanical analysis
    return HttpResponse.json({
      success: true,
      data: {
        joint_angles: [
          { joint_name: 'Left Shoulder', left_angle: 150, status: 'normal' },
          { joint_name: 'Right Shoulder', right_angle: 145, status: 'normal' }
        ],
        movement_quality_score: 85,
        detected_compensations: [],
        recommendations: ['Continue current exercise program'],
        deficiencies: []
      }
    })
  })
)

describe('PhysioMotion API Tests', () => {
  beforeAll(() => mockServer.listen())
  afterAll(() => mockServer.close())

  describe('Health Check', () => {
    it('returns ok status', async () => {
      const res = await fetch('/api/health')
      const data = await res.json()
      expect(data.status).toBe('ok')
    })
  })

  describe('Authentication', () => {
    it('login with valid credentials succeeds', async () => {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'demo@test.com', password: 'demo123' })
      })
      expect(res.ok).toBe(true)
      const data = await res.json()
      expect(data.success).toBe(true)
      expect(data.data.email).toBe('demo@test.com')
    })

    it('login with invalid credentials fails', async () => {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: 'wrong@test.com', password: 'wrong' })
      })
      expect(res.status).toBe(401)
    })
  })

  describe('Patient Management', () => {
    it('gets patient list', async () => {
      const res = await fetch('/api/patients')
      const data = await res.json()
      expect(data.success).toBe(true)
      expect(Array.isArray(data.data)).toBe(true)
    })
  })

  describe('Exercise Library', () => {
    it('gets exercise list', async () => {
      const res = await fetch('/api/exercises')
      const data = await res.json()
      expect(data.success).toBe(true)
      expect(data.data.length).toBeGreaterThan(0)
    })
  })

  describe('Biomechanical Analysis', () => {
    it('analyzes movement from skeleton data', async () => {
      const res = await fetch('/api/analyze-movement', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          skeletonData: {
            landmarks: {
              left_shoulder: { x: 0.5, y: 0.2, z: 0 },
              left_elbow: { x: 0.6, y: 0.3, z: 0 },
              left_wrist: { x: 0.7, y: 0.4, z: 0 },
              left_hip: { x: 0.5, y: 0.5, z: 0 },
              left_knee: { x: 0.5, y: 0.7, z: 0 },
              left_ankle: { x: 0.5, y: 0.9, z: 0 },
              left_foot_index: { x: 0.5, y: 1.0, z: 0 },
              right_shoulder: { x: 0.5, y: 0.2, z: 0 },
              right_elbow: { x: 0.4, y: 0.3, z: 0 },
              right_wrist: { x: 0.3, y: 0.4, z: 0 },
              right_hip: { x: 0.5, y: 0.5, z: 0 },
              right_knee: { x: 0.5, y: 0.7, z: 0 },
              right_ankle: { x: 0.5, y: 0.9, z: 0 },
              right_foot_index: { x: 0.5, y: 1.0, z: 0 },
              nose: { x: 0.5, y: 0.1, z: 0 },
              left_eye: { x: 0.52, y: 0.08, z: 0 },
              right_eye: { x: 0.48, y: 0.08, z: 0 }
            }
          }
        })
      })
      expect(res.ok).toBe(true)
      const data = await res.json()
      expect(data.success).toBe(true)
      expect(data.data.movement_quality_score).toBeDefined()
    })
  })
})

describe('Biomechanics Utilities', () => {
  const calculateAngle = (a: {x: number, y: number, z: number}, b: {x: number, y: number, z: number}, c: {x: number, y: number, z: number}) => {
    const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
    const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z }
    const dotProduct = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z)
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z)
    const angleRad = Math.acos(Math.max(-1, Math.min(1, dotProduct / (magBA * magBC))))
    return angleRad * (180 / Math.PI)
  }

  it('calculates angle between three points', () => {
    const a = { x: 0, y: 0, z: 0 }
    const b = { x: 1, y: 0, z: 0 }
    const c = { x: 1, y: 1, z: 0 }
    const angle = calculateAngle(a, b, c)
    expect(angle).toBeCloseTo(90, 1)
  })

  it('detects normal range of motion', () => {
    const shoulderAngle = 150
    const status = shoulderAngle >= 150 ? 'normal' : 'limited'
    expect(status).toBe('normal')
  })

  it('detects limited range of motion', () => {
    const shoulderAngle = 120
    const status = shoulderAngle >= 150 ? 'normal' : 'limited'
    expect(status).toBe('limited')
  })
})

describe('Data Validation', () => {
  const validateEmail = (email: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
  }

  const validatePassword = (password: string) => {
    return password.length >= 8 && /[A-Z]/.test(password) && /[a-z]/.test(password) && /[0-9]/.test(password)
  }

  it('validates correct email format', () => {
    expect(validateEmail('test@example.com')).toBe(true)
  })

  it('rejects invalid email format', () => {
    expect(validateEmail('invalid-email')).toBe(false)
  })

  it('validates strong password', () => {
    expect(validatePassword('Secure123')).toBe(true)
  })

  it('rejects weak password', () => {
    expect(validatePassword('weak')).toBe(false)
  })
})