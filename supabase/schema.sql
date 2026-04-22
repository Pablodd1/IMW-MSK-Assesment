-- PhysioMotion Supabase Database Schema
-- Run this in Supabase SQL Editor to create all tables

-- ============================================
-- CLINICIANS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS clinicians (
  id SERIAL PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  title TEXT,
  license_number TEXT,
  license_state TEXT(2),
  npi_number TEXT(10),
  phone TEXT,
  clinic_name TEXT,
  role TEXT DEFAULT 'clinician',
  active INTEGER DEFAULT 1,
  last_login TIMESTAMP,
  last_activity TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clinicians_email ON clinicians(email);
CREATE INDEX IF NOT EXISTS idx_clinicians_role ON clinicians(role);

-- ============================================
-- PATIENTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS patients (
  id SERIAL PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  date_of_birth DATE,
  gender TEXT CHECK (gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
  email TEXT,
  phone TEXT,
  emergency_contact_name TEXT,
  emergency_contact_phone TEXT,
  address_line1 TEXT,
  address_line2 TEXT,
  city TEXT,
  state TEXT(2),
  zip_code TEXT(10),
  country TEXT DEFAULT 'US',
  height_cm NUMERIC(5,2),
  weight_kg NUMERIC(5,2),
  blood_type TEXT,
  insurance_provider TEXT,
  insurance_policy_number TEXT,
  insurance_group_number TEXT,
  patient_status TEXT DEFAULT 'active' CHECK (patient_status IN ('active', 'inactive', 'discharged')),
  referring_physician TEXT,
  primary_diagnosis TEXT,
  notes TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patients_status ON patients(patient_status);
CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(last_name, first_name);

-- ============================================
-- MEDICAL HISTORY TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS medical_history (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
  surgery_type TEXT CHECK (surgery_type IN ('pre_surgery', 'post_surgery', 'none', 'athletic_performance')),
  surgery_date DATE,
  surgery_description TEXT,
  conditions JSONB,
  medications JSONB,
  allergies JSONB,
  current_pain_level INTEGER CHECK (current_pain_level BETWEEN 0 AND 10),
  pain_location JSONB,
  pain_description TEXT,
  previous_pt_therapy TEXT,
  previous_chiropractic TEXT,
  previous_surgeries TEXT,
  activity_level TEXT CHECK (activity_level IN ('sedentary', 'light', 'moderate', 'active', 'very_active')),
  occupation TEXT,
  sports_activities JSONB,
  treatment_goals TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_medical_history_patient ON medical_history(patient_id);

-- ============================================
-- ASSESSMENTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS assessments (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
  clinician_id INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
  assessment_type TEXT CHECK (assessment_type IN ('initial', 'progress', 'discharge', 'athletic_performance')),
  assessment_status TEXT DEFAULT 'in_progress' CHECK (assessment_status IN ('in_progress', 'completed', 'cancelled')),
  session_date DATE,
  duration_minutes INTEGER,
  overall_score INTEGER,
  mobility_score INTEGER,
  stability_score INTEGER,
  movement_pattern_score INTEGER,
  subjective_findings TEXT,
  objective_findings TEXT,
  assessment_summary TEXT,
  plan TEXT,
  femto_mega_connected INTEGER DEFAULT 0,
  video_recorded INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_clinician ON assessments(clinician_id);
CREATE INDEX IF NOT EXISTS idx_assessments_status ON assessments(assessment_status);

-- ============================================
-- EXERCISES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS exercises (
  id SERIAL PRIMARY KEY,
  exercise_name TEXT NOT NULL,
  exercise_category TEXT CHECK (exercise_category IN ('strength', 'flexibility', 'balance', 'mobility', 'stability', 'cardio', 'functional')),
  target_muscles JSONB,
  target_joints JSONB,
  target_movements JSONB,
  difficulty_level TEXT CHECK (difficulty_level IN ('beginner', 'intermediate', 'advanced')),
  description TEXT,
  instructions TEXT,
  contraindications TEXT,
  demo_video_url TEXT,
  demo_image_url TEXT,
  reference_keypoints JSONB,
  acceptable_deviation NUMERIC(5,2),
  equipment_required JSONB,
  estimated_duration_seconds INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- PRESCRIBED EXERCISES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS prescribed_exercises (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
  assessment_id INTEGER REFERENCES assessments(id) ON DELETE SET NULL,
  exercise_id INTEGER REFERENCES exercises(id) ON DELETE CASCADE,
  sets INTEGER NOT NULL DEFAULT 3,
  repetitions INTEGER NOT NULL DEFAULT 10,
  hold_duration_seconds INTEGER,
  rest_between_sets_seconds INTEGER,
  times_per_week INTEGER NOT NULL DEFAULT 3,
  total_weeks INTEGER,
  prescription_status TEXT DEFAULT 'active' CHECK (prescription_status IN ('active', 'completed', 'discontinued', 'modified')),
  clinical_reason TEXT,
  target_deficiency TEXT,
  compliance_percentage NUMERIC(5,2),
  last_performed_at TIMESTAMP,
  prescribed_at TIMESTAMP DEFAULT NOW(),
  prescribed_by INTEGER REFERENCES clinicians(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_prescribed_patient ON prescribed_exercises(patient_id);
CREATE INDEX IF NOT EXISTS idx_prescribed_status ON prescribed_exercises(prescription_status);

-- ============================================
-- EXERCISE SESSIONS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS exercise_sessions (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
  prescribed_exercise_id INTEGER REFERENCES prescribed_exercises(id) ON DELETE SET NULL,
  session_date DATE,
  completed INTEGER DEFAULT 0,
  sets_completed INTEGER,
  reps_completed INTEGER,
  duration_seconds INTEGER,
  form_quality_score INTEGER CHECK (form_quality_score BETWEEN 0 AND 100),
  pose_accuracy_data JSONB,
  form_errors JSONB,
  compensation_patterns JSONB,
  pain_level_during INTEGER CHECK (pain_level_during BETWEEN 0 AND 10),
  difficulty_rating INTEGER CHECK (difficulty_rating BETWEEN 1 AND 5),
  patient_notes TEXT,
  recording_url TEXT,
  analyzed_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- BILLING CODES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS billing_codes (
  id SERIAL PRIMARY KEY,
  cpt_code TEXT NOT NULL UNIQUE,
  code_description TEXT NOT NULL,
  code_category TEXT CHECK (code_category IN ('evaluation', 'treatment', 'rpm', 'exercise', 'monitoring')),
  minimum_duration_minutes INTEGER,
  requires_documentation INTEGER DEFAULT 0,
  is_rpm_code INTEGER DEFAULT 0,
  rpm_time_requirement_minutes INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Insert common CPT codes
INSERT INTO billing_codes (cpt_code, code_description, code_category, minimum_duration_minutes, requires_documentation) VALUES
('97001', 'Physical therapy evaluation', 'evaluation', 30, 1),
('97002', 'PT re-evaluation', 'evaluation', 30, 1),
('97003', 'PT progress note', 'evaluation', 15, 1),
('97010', 'Therapeutic exercise', 'treatment', 15, 0),
('97012', 'Aquatic therapy', 'treatment', 15, 0),
('97014', 'Electrical stimulation', 'treatment', 15, 0),
('97016', 'Hydrotherapy', 'treatment', 15, 0),
('97018', 'Therapeutic ultrasound', 'treatment', 15, 0),
('97022', 'Electrical stimulation (unattended)', 'treatment', 15, 0),
('97032', 'Electrical stimulation (manual)', 'treatment', 15, 0),
('97034', 'Therapeutic activities', 'treatment', 15, 0),
('97035', 'TENS unit', 'treatment', 15, 0),
('97110', 'Therapeutic exercise', 'exercise', 15, 0),
('97112', 'Therapeutic activities', 'exercise', 15, 0),
('97116', 'Gait training', 'exercise', 15, 0),
('97530', 'Therapeutic activities (1 pt)', 'treatment', 15, 0),
('97535', 'ADL training', 'treatment', 15, 0),
('97542', 'Wheelchair training', 'treatment', 15, 0)
ON CONFLICT (cpt_code) DO NOTHING;

-- ============================================
-- BILLABLE EVENTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS billable_events (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
  assessment_id INTEGER REFERENCES assessments(id) ON DELETE SET NULL,
  exercise_session_id INTEGER REFERENCES exercise_sessions(id) ON DELETE SET NULL,
  cpt_code_id INTEGER REFERENCES billing_codes(id) ON DELETE CASCADE,
  service_date DATE NOT NULL,
  duration_minutes INTEGER,
  clinical_note TEXT,
  medical_necessity TEXT,
  billing_status TEXT DEFAULT 'pending' CHECK (billing_status IN ('pending', 'submitted', 'paid', 'denied')),
  provider_id INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
  provider_npi TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_billable_patient ON billable_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_billable_status ON billable_events(billing_status);

-- ============================================
-- AUDIT LOGS TABLE (HIPAA)
-- ============================================
CREATE TABLE IF NOT EXISTS audit_logs (
  id SERIAL PRIMARY KEY,
  clinician_id INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id TEXT,
  ip_address TEXT,
  user_agent TEXT,
  http_method TEXT,
  http_status INTEGER,
  duration_ms INTEGER,
  success INTEGER DEFAULT 1,
  details JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_clinician ON audit_logs(clinician_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);

-- ============================================
-- SEED DATA FOR TESTING
-- ============================================
INSERT INTO clinicians (email, password_hash, first_name, last_name, title, role) VALUES
('demo@physiomotion.com', '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqWqN8A.MF2', 'Demo', 'User', 'DPT', 'clinician'),
('admin@physiomotion.com', '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqWqN8A.MF2', 'Admin', 'Admin', 'MD', 'admin')
ON CONFLICT (email) DO NOTHING;

INSERT INTO exercises (exercise_name, exercise_category, difficulty_level, description, instructions, target_muscles) VALUES
('Deep Squat', 'mobility', 'intermediate', 'Full squat with proper form', 'Stand shoulder-width apart, lower hips below parallel, keep torso upright', '["glutes", "quadriceps", "hamstrings"]'),
('Hip Flexor Stretch', 'flexibility', 'beginner', 'Kneeling hip flexor stretch', 'Kneel on one knee, push hips forward, feel stretch in front hip', '["hip flexors", "rectus femoris"]'),
('Single Leg Balance', 'balance', 'intermediate', 'Balance on one leg', 'Stand on one leg, arms out, hold for 30 seconds', '["core", "hip abductors", "ankle stabilizers"]'),
('Shoulder Mobility', 'mobility', 'beginner', 'Arm circles and sweeps', 'Circle arms forward and backward, reach across body', '["deltoids", "rotator cuff", "upper back"]'),
('Plank Hold', 'stability', 'beginner', 'Core stabilization', 'Hold plank position, engage core, maintain straight line', '["core", "shoulders", "glutes"]'),
('Lunge Pattern', 'functional', 'intermediate', 'Walking lunges', 'Step forward into lunge, lower back knee, push through front heel', '["quadriceps", "glutes", "hamstrings"]'),
('Dead Bug', 'stability', 'intermediate', 'Core coordination', 'Lie on back, extend opposite arm/leg, keep spine neutral', '["core", "hip flexors"]'),
('Bird Dog', 'stability', 'beginner', 'Quadruped stability', 'On hands and knees, extend opposite arm and leg, hold', '["core", "back extensors"]')
ON CONFLICT DO NOTHING;

INSERT INTO patients (first_name, last_name, date_of_birth, gender, email, phone) VALUES
('John', 'Sample', '1985-03-15', 'male', 'john.sample@email.com', '555-0101'),
('Jane', 'Sample', '1990-07-22', 'female', 'jane.sample@email.com', '555-0102'),
('Bob', 'Test', '1978-11-08', 'male', 'bob.test@email.com', '555-0103')
ON CONFLICT DO NOTHING;