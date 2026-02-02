-- PhysioAI Medical Assessment System Database Schema
-- Comprehensive schema for patient management, assessments, and monitoring

-- Patients table - Complete demographics and medical history
CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,

  -- Basic Demographics
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  date_of_birth TEXT NOT NULL,
  gender TEXT CHECK(gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
  email TEXT UNIQUE,
  phone TEXT,

  -- Address Information
  address_line1 TEXT,
  address_line2 TEXT,
  city TEXT,
  state TEXT,
  zip_code TEXT,
  country TEXT DEFAULT 'USA',

  -- Emergency Contact
  emergency_contact_name TEXT,
  emergency_contact_phone TEXT,
  emergency_contact_relationship TEXT,

  -- Medical Information
  primary_physician TEXT,
  insurance_provider TEXT,
  insurance_policy_number TEXT,
  medical_history TEXT, -- JSON field for detailed history
  current_medications TEXT, -- JSON array
  allergies TEXT, -- JSON array

  -- Assessment Context
  assessment_reason TEXT CHECK(assessment_reason IN ('pre_surgery', 'post_surgery', 'athletic_performance', 'injury_recovery', 'general_wellness')),
  chief_complaint TEXT,
  pain_scale INTEGER CHECK(pain_scale BETWEEN 0 AND 10),
  activity_level TEXT CHECK(activity_level IN ('sedentary', 'light', 'moderate', 'active', 'very_active')),

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  last_visit DATETIME
);

-- Movement assessments table
CREATE TABLE IF NOT EXISTS assessments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  clinician_id INTEGER,

  -- Assessment metadata
  assessment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  assessment_type TEXT CHECK(assessment_type IN ('initial', 'progress', 'discharge', 'follow_up')),
  status TEXT CHECK(status IN ('in_progress', 'completed', 'reviewed', 'archived')) DEFAULT 'in_progress',

  -- FMS Tests performed
  tests_completed TEXT, -- JSON array of test names
  total_score INTEGER,

  -- Video/Image data
  video_urls TEXT, -- JSON array of R2 storage URLs
  camera_type TEXT CHECK(camera_type IN ('femto_mega', 'mobile', 'web')),

  -- Analysis results (stored as JSON)
  biomechanical_data TEXT, -- JSON: joint angles, ROM, movement patterns
  ai_analysis_results TEXT, -- JSON: AI-generated insights
  deficiencies_detected TEXT, -- JSON array

  -- Medical notes
  subjective_notes TEXT,
  objective_findings TEXT,
  assessment_notes TEXT,
  treatment_plan TEXT,

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  reviewed_at DATETIME,

  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- Functional Movement Screen (FMS) tests
CREATE TABLE IF NOT EXISTS movement_tests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  assessment_id INTEGER NOT NULL,

  -- Test identification
  test_name TEXT NOT NULL,
  test_category TEXT CHECK(test_category IN ('mobility', 'stability', 'flexibility', 'strength', 'balance', 'coordination')),
  test_order INTEGER,

  -- Instructions and media
  instructions TEXT NOT NULL,
  demo_video_url TEXT,
  expected_duration INTEGER, -- seconds

  -- Test results
  status TEXT CHECK(status IN ('pending', 'recording', 'completed', 'failed')) DEFAULT 'pending',
  score INTEGER CHECK(score BETWEEN 0 AND 3),

  -- Captured data
  video_url TEXT,
  capture_timestamp DATETIME,
  skeleton_data TEXT, -- JSON: 32-joint tracking from Femto Mega or MediaPipe

  -- Measurements
  rom_measurements TEXT, -- JSON: range of motion data
  joint_angles TEXT, -- JSON: specific joint angles
  movement_quality_score REAL,

  -- AI Analysis
  ai_feedback TEXT,
  deficiencies TEXT, -- JSON array
  compensations_detected TEXT, -- JSON array

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,

  FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE
);

-- Exercise library
CREATE TABLE IF NOT EXISTS exercises (
  id INTEGER PRIMARY KEY AUTOINCREMENT,

  -- Exercise identification
  name TEXT NOT NULL UNIQUE,
  category TEXT CHECK(category IN ('mobility', 'stability', 'strength', 'flexibility', 'balance', 'coordination', 'cardio')),
  body_region TEXT, -- JSON array: neck, shoulder, spine, hip, knee, ankle, etc.

  -- Exercise details
  description TEXT NOT NULL,
  instructions TEXT NOT NULL,
  demo_video_url TEXT,
  demo_image_url TEXT,

  -- Difficulty and modifications
  difficulty TEXT CHECK(difficulty IN ('beginner', 'intermediate', 'advanced')),
  prerequisites TEXT, -- JSON array of exercise IDs
  modifications TEXT, -- JSON array of easier/harder variations

  -- Default parameters
  default_sets INTEGER DEFAULT 3,
  default_reps INTEGER DEFAULT 10,
  default_hold_time INTEGER, -- seconds
  default_frequency INTEGER DEFAULT 3, -- times per week

  -- Clinical information
  indications TEXT, -- JSON array: conditions this helps
  contraindications TEXT, -- JSON array: when NOT to do this
  targeted_deficiencies TEXT, -- JSON array

  -- Metadata
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Exercise prescriptions
CREATE TABLE IF NOT EXISTS prescriptions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  assessment_id INTEGER NOT NULL,
  clinician_id INTEGER,

  -- Prescription metadata
  prescription_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  start_date DATE NOT NULL,
  end_date DATE,
  status TEXT CHECK(status IN ('active', 'completed', 'discontinued', 'modified')) DEFAULT 'active',

  -- Program details
  program_name TEXT,
  program_goals TEXT, -- JSON array
  frequency_per_week INTEGER DEFAULT 3,
  estimated_duration_minutes INTEGER DEFAULT 30,

  -- Clinical notes
  clinician_notes TEXT,
  patient_instructions TEXT,
  precautions TEXT,

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE
);

-- Prescribed exercises (join table with details)
CREATE TABLE IF NOT EXISTS prescribed_exercises (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prescription_id INTEGER NOT NULL,
  exercise_id INTEGER NOT NULL,

  -- Exercise parameters (can override defaults)
  sets INTEGER NOT NULL,
  reps INTEGER NOT NULL,
  hold_time INTEGER, -- seconds
  rest_time INTEGER, -- seconds between sets
  frequency_per_week INTEGER NOT NULL,

  -- Exercise order and grouping
  exercise_order INTEGER,
  superset_group INTEGER, -- group exercises to be done together

  -- Progression tracking
  progression_criteria TEXT,
  current_level TEXT CHECK(current_level IN ('beginner', 'intermediate', 'advanced')),

  -- Status
  status TEXT CHECK(status IN ('active', 'completed', 'skipped', 'modified')) DEFAULT 'active',

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (prescription_id) REFERENCES prescriptions(id) ON DELETE CASCADE,
  FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE
);

-- Exercise sessions (patient home tracking)
CREATE TABLE IF NOT EXISTS exercise_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  prescription_id INTEGER NOT NULL,

  -- Session metadata
  session_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  session_type TEXT CHECK(session_type IN ('home', 'clinic', 'supervised', 'telehealth')),
  device_type TEXT CHECK(device_type IN ('mobile', 'tablet', 'web')),

  -- Session details
  duration_minutes INTEGER,
  completed BOOLEAN DEFAULT FALSE,
  completion_percentage REAL,

  -- Quality metrics
  overall_form_score REAL, -- 0-100
  adherence_score REAL, -- 0-100
  pain_level INTEGER CHECK(pain_level BETWEEN 0 AND 10),

  -- Patient feedback
  difficulty_rating INTEGER CHECK(difficulty_rating BETWEEN 1 AND 5),
  notes TEXT,

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,

  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (prescription_id) REFERENCES prescriptions(id) ON DELETE CASCADE
);

-- Individual exercise performance tracking
CREATE TABLE IF NOT EXISTS exercise_performances (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  prescribed_exercise_id INTEGER NOT NULL,

  -- Performance data
  sets_completed INTEGER,
  reps_completed INTEGER,
  hold_time_achieved INTEGER,

  -- Quality assessment (from MediaPipe or Femto Mega)
  video_url TEXT,
  skeleton_data TEXT, -- JSON: pose tracking data
  form_score REAL, -- 0-100
  rom_achieved TEXT, -- JSON: measured range of motion

  -- AI Feedback
  ai_corrections TEXT, -- JSON array of form corrections
  compensations_detected TEXT, -- JSON array
  quality_rating TEXT CHECK(quality_rating IN ('excellent', 'good', 'fair', 'poor')),

  -- Patient feedback
  pain_during_exercise INTEGER CHECK(pain_during_exercise BETWEEN 0 AND 10),
  difficulty INTEGER CHECK(difficulty BETWEEN 1 AND 5),
  notes TEXT,

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (session_id) REFERENCES exercise_sessions(id) ON DELETE CASCADE,
  FOREIGN KEY (prescribed_exercise_id) REFERENCES prescribed_exercises(id) ON DELETE CASCADE
);

-- Remote Patient Monitoring (RPM) tracking
CREATE TABLE IF NOT EXISTS rpm_monitoring (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,

  -- Billing period
  billing_month TEXT NOT NULL, -- YYYY-MM format

  -- CPT code tracking
  cpt_98975_minutes INTEGER DEFAULT 0, -- Remote therapeutic monitoring treatment
  cpt_98976_minutes INTEGER DEFAULT 0, -- Additional 20 minutes
  cpt_98977_count INTEGER DEFAULT 0, -- Interactive communication (20+ min)
  cpt_98980_count INTEGER DEFAULT 0, -- Remote therapeutic monitoring setup
  cpt_98981_count INTEGER DEFAULT 0, -- Remote therapeutic monitoring device supply

  -- Activity metrics
  total_sessions INTEGER DEFAULT 0,
  total_minutes_monitored INTEGER DEFAULT 0,
  days_with_activity INTEGER DEFAULT 0,
  adherence_rate REAL, -- percentage

  -- Clinical review
  reviewed_by_clinician BOOLEAN DEFAULT FALSE,
  review_date DATETIME,
  review_notes TEXT,

  -- Billing status
  billable BOOLEAN DEFAULT FALSE,
  billed BOOLEAN DEFAULT FALSE,
  billing_date DATE,

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  UNIQUE(patient_id, billing_month)
);

-- Clinicians/Users table
CREATE TABLE IF NOT EXISTS clinicians (
  id INTEGER PRIMARY KEY AUTOINCREMENT,

  -- Basic information
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,

  -- Credentials
  title TEXT, -- DPT, DC, MD, etc.
  license_number TEXT,
  license_state TEXT,
  npi_number TEXT, -- National Provider Identifier

  -- Role and permissions
  role TEXT CHECK(role IN ('admin', 'clinician', 'assistant', 'viewer')) DEFAULT 'clinician',
  specialties TEXT, -- JSON array

  -- Contact
  phone TEXT,
  clinic_name TEXT,

  -- Status
  active BOOLEAN DEFAULT TRUE,

  -- Timestamps
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  last_login DATETIME
);

-- System settings and configurations
CREATE TABLE IF NOT EXISTS system_settings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  setting_key TEXT UNIQUE NOT NULL,
  setting_value TEXT NOT NULL,
  setting_type TEXT CHECK(setting_type IN ('string', 'number', 'boolean', 'json')),
  description TEXT,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email);
CREATE INDEX IF NOT EXISTS idx_patients_last_visit ON patients(last_visit);
CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_date ON assessments(assessment_date);
CREATE INDEX IF NOT EXISTS idx_assessments_status ON assessments(status);
CREATE INDEX IF NOT EXISTS idx_movement_tests_assessment ON movement_tests(assessment_id);
CREATE INDEX IF NOT EXISTS idx_prescriptions_patient ON prescriptions(patient_id);
CREATE INDEX IF NOT EXISTS idx_prescriptions_status ON prescriptions(status);
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_prescription ON prescribed_exercises(prescription_id);
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_patient ON exercise_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_prescription ON exercise_sessions(prescription_id);
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_date ON exercise_sessions(session_date);
CREATE INDEX IF NOT EXISTS idx_exercise_performances_session ON exercise_performances(session_id);
CREATE INDEX IF NOT EXISTS idx_rpm_monitoring_patient ON rpm_monitoring(patient_id);
CREATE INDEX IF NOT EXISTS idx_rpm_monitoring_month ON rpm_monitoring(billing_month);
CREATE INDEX IF NOT EXISTS idx_clinicians_email ON clinicians(email);
