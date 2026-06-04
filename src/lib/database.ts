/**
 * Database Service Layer
 * Replaces in-memory MOCK_PATIENTS with Supabase persistence.
 * Falls back to mock data when Supabase is unavailable.
 */

import { getSupabase, isSupabaseEnabled } from './supabase.js';
import { MOCK_PATIENTS, EXERCISE_LIBRARY } from '../mockData.js';

// ─── Types ───

export interface Patient {
  id: string;
  first_name: string;
  last_name: string;
  date_of_birth?: string;
  gender?: string;
  email?: string;
  phone?: string;
  address?: string;
  emergency_contact?: string;
  created_at?: string;
  updated_at?: string;
  medical_history?: any;
  progress_metrics?: any;
}

export interface Assessment {
  id: string;
  patient_id: string;
  clinician_id?: string;
  assessment_type: string;
  assessment_date: string;
  chief_complaint?: string;
  pain_scale?: number;
  movement_test?: string;
  goals?: string[];
  clinical_notes?: string;
  ai_analysis?: any;
  status?: string;
  created_at?: string;
}

export interface Test {
  id: string;
  assessment_id: string;
  test_name: string;
  rationale?: string;
  status?: string;
  results?: any;
  created_at?: string;
}

export interface Prescription {
  id: string;
  assessment_id?: string;
  patient_id: string;
  exercise_id: string;
  exercise_name: string;
  sets?: number;
  reps?: number;
  frequency?: string;
  notes?: string;
  created_at?: string;
}

export interface ExerciseSession {
  id: string;
  patient_id: string;
  session_date: string;
  exercises_completed?: any[];
  duration_minutes?: number;
  pain_level?: number;
  notes?: string;
  adherence?: string;
}

// ─── Helpers ───

function mockToDbPatient(p: any): Patient {
  return {
    id: String(p.id),
    first_name: p.first_name || '',
    last_name: p.last_name || '',
    date_of_birth: p.date_of_birth,
    gender: p.gender,
    email: p.email,
    phone: p.phone,
    created_at: p.created_at,
    medical_history: p.medical_history || {},
    progress_metrics: p.progress_metrics || { pain_trend: [], functional_score_trend: [] },
  };
}

// ─── Patients ───

export async function listPatients(): Promise<Patient[]> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('patients').select('*').order('created_at', { ascending: false });
    if (error) throw error;
    return data || [];
  }
  return MOCK_PATIENTS.map(mockToDbPatient);
}

export async function getPatient(id: string): Promise<Patient | null> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('patients').select('*').eq('id', id).single();
    if (error && error.code !== 'PGRST116') throw error;
    return data || null;
  }
  const p = MOCK_PATIENTS.find((mp: any) => String(mp.id) === id);
  return p ? mockToDbPatient(p) : null;
}

export async function createPatient(input: Omit<Patient, 'id' | 'created_at'>): Promise<Patient> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('patients').insert({
      first_name: input.first_name,
      last_name: input.last_name,
      date_of_birth: input.date_of_birth,
      gender: input.gender,
      email: input.email,
      phone: input.phone,
      address: input.address,
      emergency_contact: input.emergency_contact,
      medical_history: input.medical_history || {},
      progress_metrics: { pain_trend: [], functional_score_trend: [] },
    }).select().single();
    if (error) throw error;
    return data;
  }
  const newPatient = {
    id: MOCK_PATIENTS.length + 1,
    ...input,
    created_at: new Date().toISOString(),
    assessments: [],
    exercise_sessions: [],
    progress_metrics: { pain_trend: [], functional_score_trend: [] },
  };
  MOCK_PATIENTS.push(newPatient as any);
  return mockToDbPatient(newPatient);
}

export async function updatePatient(id: string, input: Partial<Patient>): Promise<Patient | null> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('patients').update({
      ...input,
      updated_at: new Date().toISOString(),
    }).eq('id', id).select().single();
    if (error) throw error;
    return data;
  }
  const idx = MOCK_PATIENTS.findIndex((p: any) => String(p.id) === id);
  if (idx === -1) return null;
  MOCK_PATIENTS[idx] = { ...MOCK_PATIENTS[idx], ...input } as any;
  return mockToDbPatient(MOCK_PATIENTS[idx]);
}

// ─── Assessments ───

export async function listAssessments(): Promise<any[]> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('assessments').select('*, patients(first_name, last_name)').order('assessment_date', { ascending: false });
    if (error) throw error;
    return (data || []).map((a: any) => ({
      ...a,
      patient_name: a.patients ? `${a.patients.first_name} ${a.patients.last_name}` : '',
    }));
  }
  return MOCK_PATIENTS.flatMap((p: any) =>
    (p.assessments || []).map((a: any) => ({
      ...a,
      patient_id: String(p.id),
      patient_name: `${p.first_name} ${p.last_name}`,
    }))
  );
}

export async function getAssessment(id: string): Promise<any | null> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('assessments').select('*, patients(*)').eq('id', id).single();
    if (error && error.code !== 'PGRST116') throw error;
    return data || null;
  }
  for (const p of MOCK_PATIENTS) {
    const a = p.assessments?.find((ass: any) => String(ass.id) === id);
    if (a) return { ...a, patient: p };
  }
  return null;
}

export async function createAssessment(input: any): Promise<any> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('assessments').insert({
      patient_id: input.patient_id,
      clinician_id: input.clinician_id,
      assessment_type: input.assessment_type || 'initial',
      chief_complaint: input.chief_complaint || '',
      pain_scale: input.pain_scale || 0,
      movement_test: input.movement_test || null,
      goals: input.goals || [],
      clinical_notes: input.clinical_notes || '',
      ai_analysis: input.ai_analysis || null,
    }).select().single();
    if (error) throw error;
    return data;
  }
  const patientIdx = MOCK_PATIENTS.findIndex((p: any) => String(p.id) === input.patient_id);
  if (patientIdx === -1) throw new Error('Patient not found');
  const newAssessment = {
    id: Date.now().toString(),
    ...input,
    assessment_date: new Date().toISOString(),
    tests: [],
    prescriptions: [],
  };
  if (!MOCK_PATIENTS[patientIdx].assessments) MOCK_PATIENTS[patientIdx].assessments = [];
  MOCK_PATIENTS[patientIdx].assessments.push(newAssessment);
  return newAssessment;
}

// ─── Tests ───

export async function addTest(assessmentId: string, input: any): Promise<any> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('tests').insert({
      assessment_id: assessmentId,
      test_name: input.test_name,
      rationale: input.rationale,
      status: input.status || 'pending',
      results: input.results || null,
    }).select().single();
    if (error) throw error;
    return data;
  }
  for (const p of MOCK_PATIENTS) {
    const a = p.assessments?.find((ass: any) => String(ass.id) === assessmentId);
    if (a) {
      const newTest = { id: Date.now().toString(), ...input, created_at: new Date().toISOString() };
      if (!a.tests) a.tests = [];
      a.tests.push(newTest);
      return newTest;
    }
  }
  throw new Error('Assessment not found');
}

// ─── Prescriptions ───

export async function createPrescription(input: any): Promise<any> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('prescriptions').insert({
      assessment_id: input.assessment_id || null,
      patient_id: input.patient_id,
      exercise_id: input.exercise_id,
      exercise_name: input.exercise_name,
      sets: input.sets || 3,
      reps: input.reps || 10,
      frequency: input.frequency || 'daily',
      notes: input.notes,
    }).select().single();
    if (error) throw error;
    return data;
  }
  const prescription = {
    id: Date.now().toString(),
    exercise_id: input.exercise_id,
    exercise_name: input.exercise_name,
    sets: input.sets,
    reps: input.reps,
    frequency: input.frequency,
    notes: input.notes,
    created_at: new Date().toISOString(),
  };
  if (input.assessment_id) {
    for (const p of MOCK_PATIENTS) {
      const a = p.assessments?.find((ass: any) => String(ass.id) === input.assessment_id);
      if (a) {
        if (!a.prescriptions) a.prescriptions = [];
        a.prescriptions.push(prescription);
        break;
      }
    }
  }
  return prescription;
}

// ─── Exercise Sessions ───

export async function recordSession(input: any): Promise<any> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('exercise_sessions').insert({
      patient_id: input.patient_id,
      exercises_completed: input.exercises_completed || [],
      duration_minutes: input.duration_minutes || 0,
      pain_level: input.pain_level || 0,
      notes: input.notes,
      adherence: input.adherence || (input.pain_level <= 3 ? 'excellent' : input.pain_level <= 5 ? 'good' : 'fair'),
    }).select().single();
    if (error) throw error;
    return data;
  }
  const patientIdx = MOCK_PATIENTS.findIndex((p: any) => String(p.id) === input.patient_id);
  if (patientIdx === -1) throw new Error('Patient not found');
  const session = {
    id: Date.now().toString(),
    session_date: new Date().toISOString(),
    ...input,
    adherence: input.pain_level <= 3 ? 'excellent' : input.pain_level <= 5 ? 'good' : 'fair',
  };
  if (!MOCK_PATIENTS[patientIdx].exercise_sessions) MOCK_PATIENTS[patientIdx].exercise_sessions = [];
  MOCK_PATIENTS[patientIdx].exercise_sessions.push(session);
  if (!MOCK_PATIENTS[patientIdx].progress_metrics) MOCK_PATIENTS[patientIdx].progress_metrics = { pain_trend: [] };
  if (!MOCK_PATIENTS[patientIdx].progress_metrics.pain_trend) MOCK_PATIENTS[patientIdx].progress_metrics.pain_trend = [];
  MOCK_PATIENTS[patientIdx].progress_metrics.pain_trend.push(input.pain_level || 0);
  return session;
}

// ─── Exercises ───

export async function listExercises(category?: string): Promise<any[]> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    let q = sb.from('exercises').select('*').order('name');
    if (category) q = q.eq('category', category);
    const { data, error } = await q;
    if (error) throw error;
    if (data && data.length > 0) return data;
  }
  if (category) return EXERCISE_LIBRARY.filter((e: any) => e.category === category);
  return EXERCISE_LIBRARY;
}

export async function getExercise(id: string): Promise<any | null> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('exercises').select('*').eq('id', id).single();
    if (error && error.code !== 'PGRST116') throw error;
    if (data) return data;
  }
  return EXERCISE_LIBRARY.find((e: any) => e.id === id) || null;
}

// ─── Clinical Phase 4-9 Tables ───

export async function listGaitSessions(patientId: string): Promise<any[]> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('gait_sessions').select('*').eq('patient_id', patientId).order('created_at', { ascending: false });
    if (error) throw error;
    return data || [];
  }
  return [];
}

export async function listMuscleAssessments(patientId: string): Promise<any[]> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('muscle_assessments').select('*').eq('patient_id', patientId).order('created_at', { ascending: false });
    if (error) throw error;
    return data || [];
  }
  return [];
}

export async function listExercisePrescriptions(patientId: string): Promise<any[]> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('exercise_prescriptions').select('*').eq('patient_id', patientId).order('created_at', { ascending: false });
    if (error) throw error;
    return data || [];
  }
  return [];
}

export async function listProgressMetrics(patientId: string): Promise<any[]> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('progress_metrics').select('*').eq('patient_id', patientId).order('session_date', { ascending: true });
    if (error) throw error;
    return data || [];
  }
  const patient = MOCK_PATIENTS.find((p: any) => String(p.id) === patientId);
  const pain = patient?.progress_metrics?.pain_trend || [];
  const fms = patient?.progress_metrics?.functional_score_trend || [];
  return [
    ...pain.map((value: number, index: number) => ({ metric_type: 'pain', metric_name: 'Pain', current_value: value, session_date: new Date(Date.now() - (pain.length - index) * 86400000).toISOString(), unit: '/10' })),
    ...fms.map((value: number, index: number) => ({ metric_type: 'function', metric_name: 'FMS', current_value: value, session_date: new Date(Date.now() - (fms.length - index) * 86400000).toISOString(), unit: '/21' })),
  ];
}

// ─── Dashboard Stats ───

export async function getDashboardStats(): Promise<any> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { count: totalPatients } = await sb.from('patients').select('*', { count: 'exact', head: true });
    const { count: totalAssessments } = await sb.from('assessments').select('*', { count: 'exact', head: true });
    const { count: totalSessions } = await sb.from('exercise_sessions').select('*', { count: 'exact', head: true });
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
    const { data: activePatientsData } = await sb.from('assessments').select('patient_id').gte('assessment_date', thirtyDaysAgo);
    const activePatientIds = new Set((activePatientsData || []).map((a: any) => a.patient_id));
    return {
      total_patients: totalPatients || 0,
      active_patients: activePatientIds.size,
      total_assessments: totalAssessments || 0,
      total_sessions: totalSessions || 0,
      recent_activity: [], // Could join patients + sessions
    };
  }
  return {
    total_patients: MOCK_PATIENTS.length,
    active_patients: MOCK_PATIENTS.filter((p: any) => {
      const latest = p.assessments?.[p.assessments.length - 1];
      if (!latest) return false;
      return (Date.now() - new Date(latest.assessment_date).getTime()) / (1000 * 60 * 60 * 24) < 30;
    }).length,
    total_assessments: MOCK_PATIENTS.reduce((acc: number, p: any) => acc + (p.assessments?.length || 0), 0),
    total_sessions: MOCK_PATIENTS.reduce((acc: number, p: any) => acc + (p.exercise_sessions?.length || 0), 0),
    recent_activity: MOCK_PATIENTS.flatMap((p: any) =>
      (p.exercise_sessions || []).map((s: any) => ({
        patient_name: `${p.first_name} ${p.last_name}`,
        date: s.session_date,
        duration: s.duration_minutes,
        exercises: s.exercises_completed?.length || 0,
      }))
    ).sort((a: any, b: any) => new Date(b.date).getTime() - new Date(a.date).getTime()).slice(0, 5),
  };
}

// ─── Clinicians ───

export async function getClinicianByEmail(email: string): Promise<any | null> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('clinicians').select('*').eq('email', email).single();
    if (error && error.code !== 'PGRST116') throw error;
    return data || null;
  }
  return null;
}

export async function createClinician(input: any): Promise<any> {
  if (isSupabaseEnabled()) {
    const sb = getSupabase()!;
    const { data, error } = await sb.from('clinicians').insert(input).select().single();
    if (error) throw error;
    return data;
  }
  return { id: '1', ...input, created_at: new Date().toISOString() };
}
