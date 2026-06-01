-- IMW-MSK PhysioMotion Database Schema
-- Run this in Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Clinicians / Users table
CREATE TABLE IF NOT EXISTS clinicians (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'clinician', -- clinician, admin, viewer
  license TEXT,
  clinic_id UUID,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_activity TIMESTAMPTZ,
  last_login TIMESTAMPTZ,
  active BOOLEAN DEFAULT TRUE
);

-- Patients table
CREATE TABLE IF NOT EXISTS patients (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  date_of_birth DATE,
  gender TEXT,
  email TEXT,
  phone TEXT,
  address TEXT,
  emergency_contact TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  medical_history JSONB DEFAULT '{}',
  progress_metrics JSONB DEFAULT '{"pain_trend": [], "functional_score_trend": []}'
);

-- Assessments table
CREATE TABLE IF NOT EXISTS assessments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  clinician_id UUID REFERENCES clinicians(id),
  assessment_type TEXT NOT NULL DEFAULT 'initial',
  assessment_date TIMESTAMPTZ DEFAULT NOW(),
  chief_complaint TEXT,
  pain_scale INTEGER DEFAULT 0,
  movement_test TEXT,
  goals TEXT[],
  clinical_notes TEXT,
  ai_analysis JSONB,
  status TEXT DEFAULT 'active', -- active, completed, cancelled
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tests (movement tests within an assessment)
CREATE TABLE IF NOT EXISTS tests (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  assessment_id UUID NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
  test_name TEXT NOT NULL,
  rationale TEXT,
  status TEXT DEFAULT 'pending', -- pending, completed, abnormal
  results JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Prescriptions / Exercise RX
CREATE TABLE IF NOT EXISTS prescriptions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  assessment_id UUID REFERENCES assessments(id) ON DELETE SET NULL,
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  exercise_id TEXT NOT NULL,
  exercise_name TEXT NOT NULL,
  sets INTEGER DEFAULT 3,
  reps INTEGER DEFAULT 10,
  frequency TEXT DEFAULT 'daily',
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Exercise Sessions (patient home compliance)
CREATE TABLE IF NOT EXISTS exercise_sessions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  session_date TIMESTAMPTZ DEFAULT NOW(),
  exercises_completed JSONB DEFAULT '[]',
  duration_minutes INTEGER DEFAULT 0,
  pain_level INTEGER DEFAULT 0,
  notes TEXT,
  adherence TEXT DEFAULT 'good' -- excellent, good, fair, poor
);

-- Exercise Library (reference data)
CREATE TABLE IF NOT EXISTS exercises (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  category TEXT, -- upper, lower, spine, core, balance
  difficulty TEXT DEFAULT 'beginner',
  target_joints TEXT[],
  contraindications TEXT[],
  cpt_code TEXT,
  video_url TEXT,
  image_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Movement Data (pose landmarks, depth data)
CREATE TABLE IF NOT EXISTS movement_data (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  assessment_id UUID REFERENCES assessments(id) ON DELETE CASCADE,
  session_id TEXT,
  movement_test TEXT,
  landmarks JSONB,
  depth_data JSONB,
  frame_count INTEGER,
  device_type TEXT, -- femto_mega, webcam, phone
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Notifications / Alerts log
CREATE TABLE IF NOT EXISTS notifications (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
  type TEXT NOT NULL, -- telegram, email, sms
  channel TEXT,
  message TEXT NOT NULL,
  status TEXT DEFAULT 'sent', -- sent, delivered, failed
  sent_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(last_name, first_name);
CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_date ON assessments(assessment_date);
CREATE INDEX IF NOT EXISTS idx_tests_assessment ON tests(assessment_id);
CREATE INDEX IF NOT EXISTS idx_prescriptions_patient ON prescriptions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_patient ON exercise_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_movement_patient ON movement_data(patient_id);
CREATE INDEX IF NOT EXISTS idx_clinicians_email ON clinicians(email);

-- Row Level Security (RLS) policies
ALTER TABLE patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE prescriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE exercise_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE movement_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- Simple RLS: all authenticated users can read/write everything
-- (Refine later for multi-tenant clinics)
CREATE POLICY "allow_all" ON patients FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON assessments FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON tests FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON prescriptions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON exercise_sessions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON movement_data FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON notifications FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON clinicians FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all" ON exercises FOR ALL USING (true) WITH CHECK (true);

-- Seed exercise library
INSERT INTO exercises (id, name, description, category, difficulty, target_joints, cpt_code)
VALUES
('pendulum', 'Pendulum (Codman)', 'Passive shoulder ROM exercise for acute phase', 'upper', 'beginner', ARRAY['shoulder'], '97110'),
('wall-slides', 'Wall Slides', 'Shoulder flexion against wall for recovery phase', 'upper', 'beginner', ARRAY['shoulder'], '97110'),
('quad-sets', 'Quad Sets', 'Isometric knee extension for acute phase', 'lower', 'beginner', ARRAY['knee'], '97110'),
('slr', 'Straight Leg Raises', 'Hip flexor strengthening', 'lower', 'beginner', ARRAY['hip','knee'], '97110'),
('cat-cow', 'Cat-Cow', 'Spinal mobility exercise', 'spine', 'beginner', ARRAY['spine','lumbar'], '97112'),
('dead-bug', 'Dead Bug', 'Core stability exercise', 'core', 'intermediate', ARRAY['spine','core'], '97110'),
('bridge', 'Glute Bridge', 'Hip and glute strengthening', 'lower', 'beginner', ARRAY['hip','glute'], '97110'),
('clamshell', 'Clamshell', 'Hip external rotator strengthening', 'lower', 'beginner', ARRAY['hip'], '97110'),
('plank', 'Plank', 'Core isometric hold', 'core', 'intermediate', ARRAY['core','spine'], '97110'),
('side-plank', 'Side Plank', 'Lateral core stability', 'core', 'intermediate', ARRAY['core','oblique'], '97110'),
('heel-raises', 'Heel Raises', 'Calf strengthening', 'lower', 'beginner', ARRAY['ankle'], '97110'),
('ankle-circles', 'Ankle Circles', 'Ankle ROM exercise', 'lower', 'beginner', ARRAY['ankle'], '97110'),
('neck-retraction', 'Chin Tuck / Neck Retraction', 'Cervical posture correction', 'spine', 'beginner', ARRAY['cervical'], '97112'),
('thoracic-rotation', 'Thoracic Rotation', 'Mid-back mobility', 'spine', 'intermediate', ARRAY['thoracic'], '97112'),
('hip-flexor-stretch', 'Hip Flexor Stretch', 'Hip flexor lengthening', 'lower', 'beginner', ARRAY['hip'], '97110'),
('hamstring-stretch', 'Hamstring Stretch', 'Posterior chain lengthening', 'lower', 'beginner', ARRAY['knee','hip'], '97110'),
('shoulder-external-rotation', 'Shoulder External Rotation', 'Rotator cuff strengthening', 'upper', 'intermediate', ARRAY['shoulder'], '97110')
ON CONFLICT (id) DO NOTHING;
