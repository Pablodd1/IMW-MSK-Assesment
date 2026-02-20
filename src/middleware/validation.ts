// Input Validation Middleware - Medical-Grade
// Uses Zod for schema validation

import { createMiddleware } from 'hono/factory'
import { z } from 'zod'
import type { Bindings } from '../types'

// Custom error handler for Zod
function formatZodError(error: z.ZodError): { field: string; message: string }[] {
  return error.errors.map(err => ({
    field: err.path.join('.'),
    message: err.message
  }))
}

// Patient Validation Schemas
export const patientCreateSchema = z.object({
  first_name: z.string()
    .min(1, 'First name is required')
    .max(100, 'First name must be less than 100 characters')
    .regex(/^[a-zA-Z\s\-']+$/, 'First name can only contain letters, spaces, hyphens, and apostrophes'),
  
  last_name: z.string()
    .min(1, 'Last name is required')
    .max(100, 'Last name must be less than 100 characters')
    .regex(/^[a-zA-Z\s\-']+$/, 'Last name can only contain letters, spaces, hyphens, and apostrophes'),
  
  date_of_birth: z.string()
    .regex(/^\d{4}-\d{2}-\d{2}$/, 'Date must be in YYYY-MM-DD format')
    .refine(date => {
      const d = new Date(date)
      const now = new Date()
      return d < now && d > new Date('1900-01-01')
    }, 'Invalid date of birth'),
  
  gender: z.enum(['male', 'female', 'other', 'prefer_not_to_say']),
  
  email: z.string()
    .email('Invalid email format')
    .optional()
    .or(z.literal('')),
  
  phone: z.string()
    .regex(/^\+?[\d\s\-()]+$/, 'Invalid phone number format')
    .optional()
    .or(z.literal('')),
  
  emergency_contact_name: z.string()
    .max(100)
    .optional()
    .or(z.literal('')),
  
  emergency_contact_phone: z.string()
    .regex(/^\+?[\d\s\-()]+$/, 'Invalid phone number format')
    .optional()
    .or(z.literal('')),
  
  address_line1: z.string()
    .max(200)
    .optional()
    .or(z.literal('')),
  
  city: z.string()
    .max(100)
    .optional()
    .or(z.literal('')),
  
  state: z.string()
    .length(2, 'State must be 2-letter code')
    .optional()
    .or(z.literal('')),
  
  zip_code: z.string()
    .regex(/^\d{5}(-\d{4})?$/, 'Invalid ZIP code')
    .optional()
    .or(z.literal('')),
  
  height_cm: z.number()
    .min(50, 'Height seems too low')
    .max(300, 'Height seems too high')
    .optional(),
  
  weight_kg: z.number()
    .min(10, 'Weight seems too low')
    .max(500, 'Weight seems too high')
    .optional(),
  
  insurance_provider: z.string()
    .max(100)
    .optional()
    .or(z.literal(''))
})

// Assessment Validation Schemas
export const assessmentCreateSchema = z.object({
  patient_id: z.number()
    .int('Patient ID must be an integer')
    .positive('Patient ID must be positive'),
  
  assessment_type: z.enum(['initial', 'progress', 'discharge', 'athletic_performance']),
  
  clinician_id: z.number()
    .int()
    .positive()
    .optional()
})

// Movement Test Validation
export const movementTestSchema = z.object({
  test_name: z.string()
    .min(1, 'Test name is required')
    .max(100),
  
  test_category: z.enum(['mobility', 'stability', 'strength', 'balance', 'functional'])
    .optional(),
  
  test_order: z.number()
    .int()
    .positive()
    .min(1)
    .max(20),
  
  instructions: z.string()
    .min(1, 'Instructions are required')
    .max(1000),
  
  demo_video_url: z.string()
    .url('Invalid URL')
    .optional()
    .or(z.literal(''))
})

// Exercise Prescription Validation
export const prescriptionSchema = z.object({
  patient_id: z.number()
    .int()
    .positive('Patient ID must be positive'),
  
  assessment_id: z.number()
    .int()
    .positive()
    .optional(),
  
  exercise_id: z.number()
    .int()
    .positive('Exercise ID must be positive'),
  
  sets: z.number()
    .int()
    .positive('Sets must be at least 1')
    .max(20, 'Sets seems too high'),
  
  repetitions: z.number()
    .int()
    .positive('Repetitions must be at least 1')
    .max(100, 'Repetitions seems too high'),
  
  times_per_week: z.number()
    .int()
    .positive()
    .max(7, 'Cannot exceed 7 times per week'),
  
  clinical_reason: z.string()
    .max(1000)
    .optional()
    .or(z.literal('')),
  
  target_deficiency: z.string()
    .max(200)
    .optional()
    .or(z.literal(''))
})

// Exercise Session Validation
export const exerciseSessionSchema = z.object({
  patient_id: z.number()
    .int()
    .positive('Patient ID must be positive'),
  
  prescribed_exercise_id: z.number()
    .int()
    .positive('Exercise ID must be positive'),
  
  sets_completed: z.number()
    .int()
    .min(0)
    .optional(),
  
  reps_completed: z.number()
    .int()
    .min(0)
    .optional(),
  
  duration_seconds: z.number()
    .int()
    .min(0)
    .optional(),
  
  form_quality_score: z.number()
    .min(0)
    .max(100)
    .optional(),
  
  pain_level_during: z.number()
    .min(0)
    .max(10)
    .optional(),
  
  difficulty_rating: z.number()
    .min(1)
    .max(5)
    .optional(),
  
  completed: z.boolean()
    .optional()
})

// Billing Event Validation
export const billingEventSchema = z.object({
  patient_id: z.number()
    .int()
    .positive('Patient ID must be positive'),
  
  assessment_id: z.number()
    .int()
    .positive()
    .optional(),
  
  exercise_session_id: z.number()
    .int()
    .positive()
    .optional(),
  
  cpt_code_id: z.number()
    .int()
    .positive('CPT code ID must be positive'),
  
  service_date: z.string()
    .regex(/^\d{4}-\d{2}-\d{2}$/, 'Date must be YYYY-MM-DD'),
  
  duration_minutes: z.number()
    .int()
    .positive()
    .max(480, 'Duration seems too high')
    .optional(),
  
  clinical_note: z.string()
    .max(5000)
    .optional()
    .or(z.literal('')),
  
  medical_necessity: z.string()
    .max(2000)
    .optional()
    .or(z.literal(''))
})

// Clinician Registration Schema
export const clinicianRegisterSchema = z.object({
  email: z.string()
    .email('Invalid email format')
    .max(255),
  
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  
  first_name: z.string()
    .min(1)
    .max(100),
  
  last_name: z.string()
    .min(1)
    .max(100),
  
  title: z.string()
    .max(50)
    .optional()
    .or(z.literal('')),
  
  license_number: z.string()
    .max(50)
    .optional()
    .or(z.literal('')),
  
  license_state: z.string()
    .length(2)
    .optional()
    .or(z.literal('')),
  
  npi_number: z.string()
    .regex(/^\d{10}$/, 'NPI must be 10 digits')
    .optional()
    .or(z.literal('')),
  
  phone: z.string()
    .regex(/^\+?[\d\s\-()]+$/)
    .optional()
    .or(z.literal('')),
  
  clinic_name: z.string()
    .max(200)
    .optional()
    .or(z.literal(''))
})

// Medical History Schema
export const medicalHistorySchema = z.object({
  surgery_type: z.enum(['pre_surgery', 'post_surgery', 'none', 'athletic_performance'])
    .optional(),
  
  surgery_date: z.string()
    .regex(/^\d{4}-\d{2}-\d{2}$/)
    .optional(),
  
  conditions: z.array(z.string()).optional(),
  medications: z.array(z.string()).optional(),
  allergies: z.array(z.string()).optional(),
  
  current_pain_level: z.number()
    .min(0)
    .max(10)
    .optional(),
  
  pain_location: z.array(z.string()).optional(),
  
  activity_level: z.enum(['sedentary', 'light', 'moderate', 'active', 'very_active'])
    .optional(),
  
  treatment_goals: z.string()
    .max(2000)
    .optional()
    .or(z.literal(''))
})

// Validation Middleware Factory
export function validate<T extends z.ZodSchema>(schema: T) {
  return createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
    try {
      const body = await c.req.json()
      const validated = schema.parse(body)
      c.set('validatedData', validated)
      await next()
    } catch (error) {
      if (error instanceof z.ZodError) {
        return c.json({
          success: false,
          error: 'Validation failed',
          details: formatZodError(error)
        }, 400)
      }
      return c.json({
        success: false,
        error: 'Invalid request body'
      }, 400)
    }
  })
}

// Sanitization helper
export function sanitizeInput(input: string): string {
  return input
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+=/gi, '') // Remove event handlers
    .trim()
}

// ID Parameter Validation
export const idParamSchema = z.object({
  id: z.string()
    .regex(/^\d+$/, 'ID must be a number')
    .transform(Number)
    .pipe(z.number().positive())
})

export function validateIdParam() {
  return createMiddleware<{ Bindings: Bindings }>(async (c, next) => {
    try {
      const { id } = idParamSchema.parse(c.req.param())
      c.set('validatedId', id)
      await next()
    } catch (error) {
      if (error instanceof z.ZodError) {
        return c.json({
          success: false,
          error: 'Invalid ID parameter',
          details: formatZodError(error)
        }, 400)
      }
      return c.json({
        success: false,
        error: 'Invalid request'
      }, 400)
    }
  })
}
