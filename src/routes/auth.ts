import { Hono } from 'hono'
import type { Bindings } from '../types'
import { hashPassword, verifyPassword, generateToken, verifyToken } from '../middleware/auth'
import { validate } from '../middleware/validation'
import { clinicianRegisterSchema, loginSchema } from '../middleware/validation'
import { auditLog } from '../middleware/hipaa'

const auth = new Hono<{ Bindings: Bindings }>()

auth.post('/register', validate(clinicianRegisterSchema), async (c) => {
  try {
    const data = c.get('validatedData')
    
    const existing = await c.env.DB.prepare(`
      SELECT id FROM clinicians WHERE email = ?
    `).bind(data.email).first()
    
    if (existing) {
      return c.json({ success: false, error: 'Email already registered' }, 400)
    }
    
    const passwordHash = await hashPassword(data.password)
    
    const result = await c.env.DB.prepare(`
      INSERT INTO clinicians (
        email, password_hash, first_name, last_name, title,
        license_number, license_state, npi_number, phone, clinic_name
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      data.email, passwordHash, data.first_name, data.last_name, data.title,
      data.license_number, data.license_state, data.npi_number,
      data.phone, data.clinic_name
    ).run()
    
    return c.json({
      success: true,
      data: { id: result.meta.last_row_id }
    })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

auth.post('/login', validate(loginSchema), async (c) => {
  try {
    const { email, password } = c.get('validatedData')
    
    const clinician = (await c.env.DB.prepare(`
      SELECT * FROM clinicians WHERE email = ? AND active = 1
    `).bind(email).first()) as any
    
    if (!clinician) {
      return c.json({ success: false, error: 'Invalid email or password' }, 401)
    }
    
    const isValid = await verifyPassword(password, clinician.password_hash)
    
    if (!isValid) {
      return c.json({ success: false, error: 'Invalid email or password' }, 401)
    }

    const secret = process.env.JWT_SECRET || process.env.AUTH_SECRET
    if (!secret) {
      return c.json({ success: false, error: 'Server configuration error' }, 500)
    }
    
    const token = await generateToken({
      id: clinician.id,
      email: clinician.email,
      role: clinician.role || 'clinician'
    }, secret)
    
    await c.env.DB.prepare(`
      UPDATE clinicians SET last_login = CURRENT_TIMESTAMP WHERE id = ?
    `).bind(clinician.id).run()
    
    const { password_hash, ...userData } = clinician
    
    return c.json({
      success: true,
      data: { ...userData, token }
    })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

auth.get('/profile/:id', async (c) => {
  try {
    const id = c.req.param('id')
    
    const clinician = await c.env.DB.prepare(`
      SELECT id, email, first_name, last_name, title, license_number,
             license_state, npi_number, phone, clinic_name, role, active,
             created_at, last_login
      FROM clinicians WHERE id = ?
    `).bind(id).first()
    
    if (!clinician) {
      return c.json({ success: false, error: 'Clinician not found' }, 404)
    }
    
    return c.json({ success: true, data: clinician })
  } catch (error: any) {
    return c.json({ success: false, error: error.message }, 500)
  }
})

export default auth
