-- Medical Movement Assessment Platform - Complete Database Schema

-- ============================================================================
-- PATIENT MANAGEMENT
-- ============================================================================

-- Patients table with comprehensive demographics
CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Personal Information
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  date_of_birth TEXT NOT NULL,
  gender TEXT CHECK(gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
  email TEXT UNIQUE,
  phone TEXT,
  emergency_contact_name TEXT,
  emergency_contact_phone TEXT,
  
  -- Address
  address_line1 TEXT,
  address_line2 TEXT,
  city TEXT,
  state TEXT,
  zip_code TEXT,
  country TEXT DEFAULT 'USA',
  
  -- Medical Information
  height_cm REAL,
  weight_kg REAL,
  blood_type TEXT,
  
  -- Insurance & Billing
  insurance_provider TEXT,
  insurance_policy_number TEXT,
  insurance_group_number TEXT,
  
  -- System Fields
  patient_status TEXT DEFAULT 'active' CHECK(patient_status IN ('active', 'inactive', 'discharged')),
  referring_physician TEXT,
  primary_diagnosis TEXT,
  notes TEXT,
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Medical History
CREATE TABLE IF NOT EXISTS medical_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  
  -- Pre/Post Surgery
  surgery_type TEXT CHECK(surgery_type IN ('pre_surgery', 'post_surgery', 'none', 'athletic_performance')),
  surgery_date TEXT,
  surgery_description TEXT,
  
  -- Medical Conditions
  conditions TEXT, -- JSON array of conditions
  medications TEXT, -- JSON array of medications
  allergies TEXT, -- JSON array of allergies
  
  -- Pain Assessment
  current_pain_level INTEGER CHECK(current_pain_level BETWEEN 0 AND 10),
  pain_location TEXT, -- JSON array
  pain_description TEXT,
  
  -- Previous Treatments
  previous_pt_therapy TEXT,
  previous_chiropractic TEXT,
  previous_surgeries TEXT,
  
  -- Lifestyle
  activity_level TEXT CHECK(activity_level IN ('sedentary', 'light', 'moderate', 'active', 'very_active')),
  occupation TEXT,
  sports_activities TEXT, -- JSON array
  
  -- Goals
  treatment_goals TEXT,
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- ============================================================================
-- ASSESSMENT SYSTEM
-- ============================================================================

-- Assessment Sessions
CREATE TABLE IF NOT EXISTS assessments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  clinician_id INTEGER,
  
  -- Assessment Type
  assessment_type TEXT NOT NULL CHECK(assessment_type IN ('initial', 'progress', 'discharge', 'athletic_performance')),
  assessment_status TEXT DEFAULT 'in_progress' CHECK(assessment_status IN ('in_progress', 'completed', 'cancelled')),
  
  -- Session Info
  session_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  duration_minutes INTEGER,
  
  -- Overall Scores
  overall_score REAL,
  mobility_score REAL,
  stability_score REAL,
  movement_pattern_score REAL,
  
  -- Clinical Notes
  subjective_findings TEXT,
  objective_findings TEXT,
  assessment_summary TEXT,
  plan TEXT,
  
  -- Camera Integration
  femto_mega_connected INTEGER DEFAULT 0, -- boolean
  video_recorded INTEGER DEFAULT 0, -- boolean
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- Functional Movement Screen (FMS) Tests
CREATE TABLE IF NOT EXISTS movement_tests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  assessment_id INTEGER NOT NULL,
  
  -- Test Information
  test_name TEXT NOT NULL,
  test_category TEXT CHECK(test_category IN ('mobility', 'stability', 'strength', 'balance', 'functional')),
  test_order INTEGER NOT NULL,
  
  -- Test Status
  test_status TEXT DEFAULT 'pending' CHECK(test_status IN ('pending', 'in_progress', 'completed', 'skipped')),
  
  -- Instructions
  instructions TEXT NOT NULL,
  demo_video_url TEXT,
  
  -- Timing
  started_at DATETIME,
  completed_at DATETIME,
  duration_seconds INTEGER,
  
  -- Camera Data
  camera_recording_url TEXT,
  skeleton_data TEXT, -- JSON from Femto Mega / MediaPipe
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE
);

-- Movement Analysis Results
CREATE TABLE IF NOT EXISTS movement_analysis (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  test_id INTEGER NOT NULL,
  
  -- Joint Angles (in degrees)
  joint_angles TEXT NOT NULL, -- JSON: {joint_name: {left: angle, right: angle}}
  
  -- Range of Motion
  rom_measurements TEXT, -- JSON: {joint: {flexion: deg, extension: deg, etc}}
  
  -- Asymmetry Detection
  left_right_asymmetry TEXT, -- JSON: {joint: difference_percentage}
  
  -- Movement Quality Scores (0-100)
  movement_quality_score REAL,
  stability_score REAL,
  compensation_detected INTEGER DEFAULT 0, -- boolean
  
  -- Deficiencies Detected
  deficiencies TEXT, -- JSON array of detected issues
  
  -- AI Analysis
  ai_confidence_score REAL, -- 0-1
  ai_recommendations TEXT, -- JSON array
  
  -- Biomechanical Data
  velocity_data TEXT, -- JSON
  acceleration_data TEXT, -- JSON
  trajectory_data TEXT, -- JSON
  
  analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (test_id) REFERENCES movement_tests(id) ON DELETE CASCADE
);

-- ============================================================================
-- EXERCISE PRESCRIPTION SYSTEM
-- ============================================================================

-- Exercise Library
CREATE TABLE IF NOT EXISTS exercise_library (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Exercise Info
  exercise_name TEXT NOT NULL UNIQUE,
  exercise_category TEXT CHECK(exercise_category IN ('strength', 'flexibility', 'balance', 'mobility', 'stability', 'cardio', 'functional')),
  
  -- Targets
  target_muscles TEXT, -- JSON array
  target_joints TEXT, -- JSON array
  target_movements TEXT, -- JSON array
  
  -- Difficulty
  difficulty_level TEXT CHECK(difficulty_level IN ('beginner', 'intermediate', 'advanced')),
  
  -- Instructions
  description TEXT NOT NULL,
  instructions TEXT NOT NULL,
  contraindications TEXT,
  
  -- Media
  demo_video_url TEXT,
  demo_image_url TEXT,
  
  -- MediaPipe Pose Reference
  reference_keypoints TEXT, -- JSON: expected body pose landmarks
  acceptable_deviation REAL, -- degrees of acceptable variation
  
  -- Metadata
  equipment_required TEXT, -- JSON array
  estimated_duration_seconds INTEGER,
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Prescribed Exercises for Patients
CREATE TABLE IF NOT EXISTS prescribed_exercises (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  assessment_id INTEGER,
  exercise_id INTEGER NOT NULL,
  
  -- Prescription Details
  sets INTEGER NOT NULL,
  repetitions INTEGER NOT NULL,
  hold_duration_seconds INTEGER,
  rest_between_sets_seconds INTEGER,
  
  -- Frequency
  times_per_week INTEGER NOT NULL,
  total_weeks INTEGER,
  
  -- Status
  prescription_status TEXT DEFAULT 'active' CHECK(prescription_status IN ('active', 'completed', 'discontinued', 'modified')),
  
  -- Clinical Reasoning
  clinical_reason TEXT, -- Why this exercise was prescribed
  target_deficiency TEXT, -- What deficiency this addresses
  
  -- Progress Tracking
  compliance_percentage REAL DEFAULT 0,
  last_performed_at DATETIME,
  
  prescribed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  prescribed_by INTEGER, -- clinician_id
  
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (assessment_id) REFERENCES assessments(id),
  FOREIGN KEY (exercise_id) REFERENCES exercise_library(id)
);

-- ============================================================================
-- REMOTE PATIENT MONITORING (RPM)
-- ============================================================================

-- Exercise Sessions (Home Monitoring)
CREATE TABLE IF NOT EXISTS exercise_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  prescribed_exercise_id INTEGER NOT NULL,
  
  -- Session Info
  session_date DATETIME DEFAULT CURRENT_TIMESTAMP,
  completed INTEGER DEFAULT 0, -- boolean
  
  -- Performance Data
  sets_completed INTEGER,
  reps_completed INTEGER,
  duration_seconds INTEGER,
  
  -- Quality Assessment from MediaPipe
  form_quality_score REAL, -- 0-100
  pose_accuracy_data TEXT, -- JSON from MediaPipe analysis
  
  -- Errors Detected
  form_errors TEXT, -- JSON array of errors detected
  compensation_patterns TEXT, -- JSON array
  
  -- Patient Feedback
  pain_level_during INTEGER CHECK(pain_level_during BETWEEN 0 AND 10),
  difficulty_rating INTEGER CHECK(difficulty_rating BETWEEN 1 AND 5),
  patient_notes TEXT,
  
  -- Media
  recording_url TEXT,
  
  analyzed_at DATETIME,
  
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (prescribed_exercise_id) REFERENCES prescribed_exercises(id) ON DELETE CASCADE
);

-- Real-time Monitoring Alerts
CREATE TABLE IF NOT EXISTS monitoring_alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  exercise_session_id INTEGER,
  
  -- Alert Type
  alert_type TEXT CHECK(alert_type IN ('form_error', 'pain_reported', 'non_compliance', 'progress_milestone', 'concern')),
  alert_severity TEXT CHECK(alert_severity IN ('low', 'medium', 'high', 'critical')),
  
  -- Details
  alert_message TEXT NOT NULL,
  alert_details TEXT, -- JSON
  
  -- Status
  alert_status TEXT DEFAULT 'new' CHECK(alert_status IN ('new', 'reviewed', 'resolved', 'dismissed')),
  reviewed_by INTEGER, -- clinician_id
  reviewed_at DATETIME,
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (exercise_session_id) REFERENCES exercise_sessions(id)
);

-- ============================================================================
-- MEDICAL BILLING & DOCUMENTATION
-- ============================================================================

-- CPT Codes for Billing
CREATE TABLE IF NOT EXISTS billing_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  cpt_code TEXT NOT NULL UNIQUE,
  code_description TEXT NOT NULL,
  code_category TEXT CHECK(code_category IN ('evaluation', 'treatment', 'rpm', 'exercise', 'monitoring')),
  
  -- Requirements
  minimum_duration_minutes INTEGER,
  requires_documentation INTEGER DEFAULT 1, -- boolean
  
  -- RPM Specific
  is_rpm_code INTEGER DEFAULT 0, -- boolean
  rpm_time_requirement_minutes INTEGER,
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Billable Events
CREATE TABLE IF NOT EXISTS billable_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  assessment_id INTEGER,
  exercise_session_id INTEGER,
  
  -- Billing Info
  cpt_code_id INTEGER NOT NULL,
  service_date DATETIME NOT NULL,
  duration_minutes INTEGER,
  
  -- Documentation
  clinical_note TEXT,
  medical_necessity TEXT,
  
  -- Status
  billing_status TEXT DEFAULT 'pending' CHECK(billing_status IN ('pending', 'submitted', 'paid', 'denied')),
  
  -- Provider Info
  provider_id INTEGER, -- clinician_id
  provider_npi TEXT,
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (assessment_id) REFERENCES assessments(id),
  FOREIGN KEY (exercise_session_id) REFERENCES exercise_sessions(id),
  FOREIGN KEY (cpt_code_id) REFERENCES billing_codes(id)
);

-- ============================================================================
-- CLINICIAN/USER MANAGEMENT
-- ============================================================================

-- Clinicians
CREATE TABLE IF NOT EXISTS clinicians (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  
  -- Personal Info
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  
  -- Credentials
  credential TEXT, -- PT, DPT, DC, MD, etc.
  license_number TEXT,
  npi_number TEXT,
  
  -- Specialization
  specialty TEXT CHECK(specialty IN ('physical_therapy', 'chiropractic', 'sports_medicine', 'orthopedics', 'other')),
  
  -- Account
  account_status TEXT DEFAULT 'active' CHECK(account_status IN ('active', 'inactive', 'suspended')),
  
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email);
CREATE INDEX IF NOT EXISTS idx_patients_status ON patients(patient_status);
CREATE INDEX IF NOT EXISTS idx_medical_history_patient ON medical_history(patient_id);

CREATE INDEX IF NOT EXISTS idx_assessments_patient ON assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_status ON assessments(assessment_status);
CREATE INDEX IF NOT EXISTS idx_assessments_date ON assessments(session_date);

CREATE INDEX IF NOT EXISTS idx_movement_tests_assessment ON movement_tests(assessment_id);
CREATE INDEX IF NOT EXISTS idx_movement_tests_status ON movement_tests(test_status);

CREATE INDEX IF NOT EXISTS idx_movement_analysis_test ON movement_analysis(test_id);

CREATE INDEX IF NOT EXISTS idx_exercise_library_category ON exercise_library(exercise_category);
CREATE INDEX IF NOT EXISTS idx_exercise_library_name ON exercise_library(exercise_name);

CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_patient ON prescribed_exercises(patient_id);
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_status ON prescribed_exercises(prescription_status);

CREATE INDEX IF NOT EXISTS idx_exercise_sessions_patient ON exercise_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_date ON exercise_sessions(session_date);
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_prescribed ON exercise_sessions(prescribed_exercise_id);

CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_patient ON monitoring_alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_status ON monitoring_alerts(alert_status);
CREATE INDEX IF NOT EXISTS idx_monitoring_alerts_severity ON monitoring_alerts(alert_severity);

CREATE INDEX IF NOT EXISTS idx_billable_events_patient ON billable_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_billable_events_date ON billable_events(service_date);
CREATE INDEX IF NOT EXISTS idx_billable_events_status ON billable_events(billing_status);

CREATE INDEX IF NOT EXISTS idx_clinicians_email ON clinicians(email);
