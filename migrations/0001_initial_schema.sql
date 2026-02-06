-- PhysioAI Medical Assessment System Database Schema (Aligned with App Logic)

-- Patients table
CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  date_of_birth TEXT NOT NULL,
  gender TEXT,
  email TEXT,
  phone TEXT,
  emergency_contact_name TEXT,
  emergency_contact_phone TEXT,
  address_line1 TEXT,
  city TEXT,
  state TEXT,
  zip_code TEXT,
  height_cm REAL,
  weight_kg REAL,
  insurance_provider TEXT,
  assessment_reason TEXT,
  chief_complaint TEXT,
  pain_scale INTEGER,
  activity_level TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Medical History table
CREATE TABLE IF NOT EXISTS medical_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  surgery_type TEXT,
  surgery_date TEXT,
  conditions TEXT, -- JSON
  medications TEXT, -- JSON
  allergies TEXT, -- JSON
  current_pain_level INTEGER,
  pain_location TEXT, -- JSON
  activity_level TEXT,
  treatment_goals TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- Clinicians table
CREATE TABLE IF NOT EXISTS clinicians (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  title TEXT,
  license_number TEXT,
  license_state TEXT,
  npi_number TEXT,
  phone TEXT,
  clinic_name TEXT,
  role TEXT DEFAULT 'clinician',
  specialties TEXT, -- JSON
  active BOOLEAN DEFAULT TRUE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  last_login DATETIME
);

-- Assessments table
CREATE TABLE IF NOT EXISTS assessments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  clinician_id INTEGER,
  assessment_type TEXT,
  status TEXT DEFAULT 'in_progress',
  assessment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  -- Results
  subjective_findings TEXT,
  objective_findings TEXT,
  assessment_summary TEXT,
  plan TEXT,
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- Movement tests table
CREATE TABLE IF NOT EXISTS movement_tests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  assessment_id INTEGER NOT NULL,
  test_name TEXT NOT NULL,
  test_category TEXT,
  test_order INTEGER,
  instructions TEXT,
  
  status TEXT DEFAULT 'pending',
  test_status TEXT, -- Added to match 0002
  
  -- Results
  skeleton_data TEXT, -- JSON
  movement_quality_score REAL,
  deficiencies TEXT, -- JSON
  compensations_detected TEXT, -- JSON
  ai_recommendations TEXT, -- JSON
  
  completed_at DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE
);

-- Movement analysis table (Optional, implied by code join)
CREATE TABLE IF NOT EXISTS movement_analysis (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  test_id INTEGER NOT NULL,
  analysis_data TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (test_id) REFERENCES movement_tests(id) ON DELETE CASCADE
);

-- Exercises table
CREATE TABLE IF NOT EXISTS exercises (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  category TEXT,
  body_region TEXT,
  description TEXT,
  instructions TEXT,
  difficulty TEXT,
  demo_video_url TEXT,
  target_muscles TEXT, -- JSON
  default_sets INTEGER,
  default_reps INTEGER,
  default_hold_time INTEGER,
  default_frequency INTEGER,
  indications TEXT, -- JSON
  contraindications TEXT, -- JSON
  targeted_deficiencies TEXT, -- JSON
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Prescribed exercises table
CREATE TABLE IF NOT EXISTS prescribed_exercises (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  assessment_id INTEGER,
  exercise_id INTEGER NOT NULL,
  
  sets INTEGER,
  repetitions INTEGER,
  times_per_week INTEGER,
  clinical_reason TEXT,
  target_deficiency TEXT,
  prescribed_by INTEGER,
  
  status TEXT DEFAULT 'active',
  compliance_percentage REAL,
  last_performed_at DATETIME,
  
  prescribed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (assessment_id) REFERENCES assessments(id),
  FOREIGN KEY (exercise_id) REFERENCES exercises(id)
);

-- Exercise sessions table
CREATE TABLE IF NOT EXISTS exercise_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  prescribed_exercise_id INTEGER NOT NULL,
  
  sets_completed INTEGER,
  reps_completed INTEGER,
  duration_seconds INTEGER,
  
  form_quality_score REAL,
  pose_accuracy_data TEXT, -- JSON
  pain_level_during INTEGER,
  difficulty_rating INTEGER,
  completed BOOLEAN,
  
  session_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (prescribed_exercise_id) REFERENCES prescribed_exercises(id)
);

-- Billing Codes
CREATE TABLE IF NOT EXISTS billing_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cpt_code TEXT UNIQUE NOT NULL,
  code_description TEXT,
  rate REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Billable Events
CREATE TABLE IF NOT EXISTS billable_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  assessment_id INTEGER,
  exercise_session_id INTEGER,
  cpt_code_id INTEGER NOT NULL,
  
  service_date DATETIME,
  duration_minutes INTEGER,
  clinical_note TEXT,
  provider_id INTEGER,
  billing_status TEXT DEFAULT 'pending',
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id),
  FOREIGN KEY (cpt_code_id) REFERENCES billing_codes(id)
);

-- System settings
CREATE TABLE IF NOT EXISTS system_settings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  setting_key TEXT UNIQUE NOT NULL,
  setting_value TEXT NOT NULL,
  setting_type TEXT,
  description TEXT,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
