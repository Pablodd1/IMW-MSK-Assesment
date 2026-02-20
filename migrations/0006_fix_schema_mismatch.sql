-- Fix schema mismatches between src/index.tsx and 0001_initial_schema.sql
-- Adds missing tables and columns required by the application logic

-- Add missing columns to prescribed_exercises
ALTER TABLE prescribed_exercises ADD COLUMN patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE;
ALTER TABLE prescribed_exercises ADD COLUMN assessment_id INTEGER REFERENCES assessments(id) ON DELETE CASCADE;
ALTER TABLE prescribed_exercises ADD COLUMN repetitions INTEGER;
ALTER TABLE prescribed_exercises ADD COLUMN times_per_week INTEGER;
ALTER TABLE prescribed_exercises ADD COLUMN clinical_reason TEXT;
ALTER TABLE prescribed_exercises ADD COLUMN target_deficiency TEXT;
ALTER TABLE prescribed_exercises ADD COLUMN prescribed_by INTEGER;
ALTER TABLE prescribed_exercises ADD COLUMN compliance_percentage INTEGER DEFAULT 0;
ALTER TABLE prescribed_exercises ADD COLUMN last_performed_at DATETIME;

-- Add missing columns to exercise_sessions
ALTER TABLE exercise_sessions ADD COLUMN prescribed_exercise_id INTEGER REFERENCES prescribed_exercises(id) ON DELETE CASCADE;
ALTER TABLE exercise_sessions ADD COLUMN sets_completed INTEGER;
ALTER TABLE exercise_sessions ADD COLUMN reps_completed INTEGER;
ALTER TABLE exercise_sessions ADD COLUMN duration_seconds INTEGER;
ALTER TABLE exercise_sessions ADD COLUMN form_quality_score REAL;
ALTER TABLE exercise_sessions ADD COLUMN pose_accuracy_data TEXT; -- JSON
ALTER TABLE exercise_sessions ADD COLUMN pain_level_during INTEGER;
ALTER TABLE exercise_sessions ADD COLUMN difficulty_rating INTEGER;

-- Create missing billing_codes table
CREATE TABLE IF NOT EXISTS billing_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cpt_code TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL,
  default_duration_minutes INTEGER,
  unit_price REAL,
  category TEXT,
  requires_preauth BOOLEAN DEFAULT FALSE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Seed billing_codes with common CPT codes
INSERT OR IGNORE INTO billing_codes (cpt_code, description, default_duration_minutes, unit_price, category) VALUES
('97163', 'Physical therapy evaluation: high complexity', 45, 120.00, 'Evaluation'),
('97110', 'Therapeutic procedure, 1 or more areas, each 15 minutes; therapeutic exercises', 15, 45.00, 'Treatment'),
('97112', 'Neuromuscular reeducation of movement, balance, coordination, kinesthetic sense, posture, and/or proprioception for sitting and/or standing activities', 15, 45.00, 'Treatment'),
('98975', 'Remote therapeutic monitoring (e.g., respiratory system, musculoskeletal system, therapy adherence, therapy response); initial set-up and patient education on use of equipment', 20, 55.00, 'RTM'),
('98977', 'Remote therapeutic monitoring; device(s) supply with daily recording(s) or programmed alert(s) transmission, each 30 days', 0, 65.00, 'RTM');

-- Create missing billable_events table
CREATE TABLE IF NOT EXISTS billable_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  assessment_id INTEGER REFERENCES assessments(id),
  exercise_session_id INTEGER REFERENCES exercise_sessions(id),
  cpt_code_id INTEGER NOT NULL REFERENCES billing_codes(id),
  service_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  duration_minutes INTEGER,
  clinical_note TEXT,
  provider_id INTEGER REFERENCES clinicians(id),
  billing_status TEXT CHECK(billing_status IN ('pending', 'submitted', 'paid', 'denied')) DEFAULT 'pending',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for the newly created/modified tables
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_patient_id_new ON prescribed_exercises(patient_id);
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_assessment_id_new ON prescribed_exercises(assessment_id);
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_prescribed_exercise_id_new ON exercise_sessions(prescribed_exercise_id);
CREATE INDEX IF NOT EXISTS idx_billable_events_patient_id ON billable_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_billable_events_service_date ON billable_events(service_date DESC);
CREATE INDEX IF NOT EXISTS idx_billable_events_status ON billable_events(billing_status);
