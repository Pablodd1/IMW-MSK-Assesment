import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

// Create client only if env vars exist
export const supabase = supabaseUrl && supabaseAnonKey
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null

// Database helper for API routes (server-side)
export function createServerClient(supabaseUrl: string, supabaseKey: string) {
  return createClient(supabaseUrl, supabaseKey, {
    auth: {
      persistSession: false,
      autoRefreshToken: false
    }
  })
}

// Types
export interface Clinician {
  id: number
  email: string
  first_name: string
  last_name: string
  title?: string
  role: string
  active: number
}

export interface Patient {
  id: number
  first_name: string
  last_name: string
  date_of_birth: string
  gender: string
  email?: string
  phone?: string
  created_at: string
}

export interface Assessment {
  id: number
  patient_id: number
  clinician_id?: number
  assessment_type: string
  assessment_status: string
  overall_score?: number
  created_at: string
}

export interface Exercise {
  id: number
  exercise_name: string
  exercise_category: string
  difficulty_level: string
  description: string
  instructions: string
}

// Auth helpers
export async function signIn(email: string, password: string) {
  if (!supabase) return { error: 'Supabase not configured' }
  
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password
  })
  
  return { data, error }
}

export async function signUp(email: string, password: string, metadata?: object) {
  if (!supabase) return { error: 'Supabase not configured' }
  
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: { data: metadata }
  })
  
  return { data, error }
}

export async function signOut() {
  if (!supabase) return { error: 'Supabase not configured' }
  
  const { error } = await supabase.auth.signOut()
  return { error }
}

export function getSession() {
  if (!supabase) return null
  return supabase.auth.getSession()
}

// Database helpers
export const db = {
  // Clinicians
  async getClinicians() {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('clinicians').select('*')
  },
  
  async getClinicianByEmail(email: string) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('clinicians').select('*').eq('email', email).single()
  },
  
  // Patients
  async getPatients() {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('patients').select('*').order('created_at', { ascending: false })
  },
  
  async getPatient(id: number) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('patients').select('*').eq('id', id).single()
  },
  
  async createPatient(patient: Partial<Patient>) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('patients').insert(patient).select().single()
  },
  
  async updatePatient(id: number, patient: Partial<Patient>) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('patients').update(patient).eq('id', id).select().single()
  },
  
  async deletePatient(id: number) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('patients').delete().eq('id', id)
  },
  
  // Assessments
  async getAssessments(patientId?: number) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    let query = supabase.from('assessments').select('*').order('created_at', { ascending: false })
    if (patientId) {
      query = query.eq('patient_id', patientId)
    }
    return query
  },
  
  async createAssessment(assessment: Partial<Assessment>) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('assessments').insert(assessment).select().single()
  },
  
  async completeAssessment(id: number, data: object) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('assessments').update({ 
      ...data, 
      assessment_status: 'completed',
      updated_at: new Date().toISOString()
    }).eq('id', id).select().single()
  },
  
  // Exercises
  async getExercises() {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('exercises').select('*').order('exercise_name')
  },
  
  async getExercise(id: number) {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('exercises').select('*').eq('id', id).single()
  },
  
  // Billing
  async getBillingCodes() {
    if (!supabase) return { data: null, error: 'Supabase not configured' }
    return supabase.from('billing_codes').select('*').order('cpt_code')
  }
}