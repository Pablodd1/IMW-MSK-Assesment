CREATE TABLE IF NOT EXISTS patients (
  id SERIAL PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  date_of_birth TEXT NOT NULL,
  gender TEXT CHECK(gender IN ('male', 'female', 'other', 'prefer_not_to_say')),
  email TEXT UNIQUE,
  phone TEXT,
  height_cm REAL,
  weight_kg REAL,
  address_line1 TEXT,
  address_line2 TEXT,
  city TEXT,
  state TEXT,
  zip_code TEXT,
  country TEXT DEFAULT 'USA',
  emergency_contact_name TEXT,
  emergency_contact_phone TEXT,
  emergency_contact_relationship TEXT,
  primary_physician TEXT,
  referring_physician TEXT,
  insurance_provider TEXT,
  insurance_id TEXT,
  insurance_group TEXT,
  patient_status TEXT DEFAULT 'active',
  created_by_clinician_id INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_visit TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assessments (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  clinician_id INTEGER,
  assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  assessment_type TEXT CHECK(assessment_type IN ('initial', 'progress', 'discharge', 'follow_up')),
  status TEXT CHECK(status IN ('in_progress', 'completed', 'reviewed', 'archived')) DEFAULT 'in_progress',
  tests_completed TEXT,
  total_score INTEGER,
  video_urls TEXT,
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS movement_tests (
  id SERIAL PRIMARY KEY,
  assessment_id INTEGER NOT NULL REFERENCES assessments(id),
  test_name TEXT NOT NULL,
  test_category TEXT CHECK(test_category IN ('mobility', 'stability', 'flexibility', 'strength', 'balance', 'coordination')),
  test_order INTEGER,
  instructions TEXT NOT NULL,
  demo_video_url TEXT,
  expected_duration INTEGER,
  status TEXT CHECK(status IN ('pending', 'recording', 'completed', 'failed')) DEFAULT 'pending',
  test_status TEXT DEFAULT 'pending',
  video_url TEXT,
  raw_data TEXT,
  processed_data TEXT,
  score INTEGER,
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exercises (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  category TEXT CHECK(category IN ('mobility', 'stability', 'strength', 'flexibility', 'balance', 'coordination', 'cardio')),
  body_region TEXT,
  description TEXT NOT NULL,
  instructions TEXT NOT NULL,
  demo_video_url TEXT,
  demo_image_url TEXT,
  difficulty TEXT CHECK(difficulty IN ('beginner', 'intermediate', 'advanced')),
  modifications TEXT,
  cpt_code_reference TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prescriptions (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  assessment_id INTEGER NOT NULL REFERENCES assessments(id),
  clinician_id INTEGER,
  prescription_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  start_date DATE NOT NULL,
  end_date DATE,
  status TEXT CHECK(status IN ('active', 'completed', 'discontinued', 'modified')) DEFAULT 'active',
  program_name TEXT,
  program_goals TEXT,
  frequency_per_week INTEGER DEFAULT 3,
  duration_weeks INTEGER,
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prescribed_exercises (
  id SERIAL PRIMARY KEY,
  prescription_id INTEGER NOT NULL REFERENCES prescriptions(id),
  exercise_id INTEGER NOT NULL REFERENCES exercises(id),
  sets INTEGER NOT NULL,
  reps INTEGER NOT NULL,
  hold_time INTEGER,
  rest_time INTEGER,
  frequency_per_week INTEGER NOT NULL,
  exercise_order INTEGER,
  superset_group INTEGER,
  patient_id INTEGER REFERENCES patients(id),
  assessment_id INTEGER REFERENCES assessments(id),
  repetitions INTEGER,
  times_per_week INTEGER,
  clinical_reason TEXT,
  target_deficiency TEXT,
  prescribed_by INTEGER,
  compliance_percentage INTEGER DEFAULT 0,
  last_performed_at TIMESTAMP,
  pain_level_during INTEGER,
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exercise_sessions (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  prescription_id INTEGER NOT NULL REFERENCES prescriptions(id),
  session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  session_type TEXT CHECK(session_type IN ('home', 'clinic', 'supervised', 'telehealth')),
  device_type TEXT CHECK(device_type IN ('mobile', 'tablet', 'web')),
  duration_minutes INTEGER,
  completed BOOLEAN DEFAULT FALSE,
  completion_percentage REAL,
  form_score REAL,
  pain_level_reported INTEGER,
  perceived_exertion INTEGER,
  patient_feedback TEXT,
  prescribed_exercise_id INTEGER REFERENCES prescribed_exercises(id),
  sets_completed INTEGER,
  reps_completed INTEGER,
  duration_seconds INTEGER,
  form_quality_score REAL,
  pose_accuracy_data TEXT,
  pain_level_during INTEGER,
  difficulty_rating INTEGER,
  telemetry_data_path TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exercise_performances (
  id SERIAL PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES exercise_sessions(id),
  prescribed_exercise_id INTEGER NOT NULL REFERENCES prescribed_exercises(id),
  sets_completed INTEGER,
  reps_completed INTEGER,
  hold_time_achieved INTEGER,
  video_url TEXT,
  skeleton_data TEXT,
  form_score REAL,
  rom_achieved TEXT,
  compensations_detected TEXT,
  pain_reported INTEGER,
  perceived_exertion INTEGER,
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rpm_monitoring (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  billing_month TEXT NOT NULL,
  cpt_98975_minutes INTEGER DEFAULT 0,
  cpt_98976_minutes INTEGER DEFAULT 0,
  cpt_98977_count INTEGER DEFAULT 0,
  cpt_98980_count INTEGER DEFAULT 0,
  cpt_98981_count INTEGER DEFAULT 0,
  total_sessions INTEGER DEFAULT 0,
  total_duration_minutes INTEGER DEFAULT 0,
  days_with_activity INTEGER DEFAULT 0,
  alerts_generated INTEGER DEFAULT 0,
  status TEXT CHECK(status IN ('tracking', 'ready_to_bill', 'billed', 'paid')) DEFAULT 'tracking',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clinicians (
  id SERIAL PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  title TEXT,
  license_number TEXT,
  license_state TEXT,
  npi_number TEXT,
  clinic_name TEXT,
  phone TEXT,
  role TEXT DEFAULT 'clinician',
  active BOOLEAN DEFAULT TRUE,
  salt TEXT NOT NULL,
  last_activity TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_settings (
  id SERIAL PRIMARY KEY,
  setting_key TEXT UNIQUE NOT NULL,
  setting_value TEXT NOT NULL,
  setting_type TEXT CHECK(setting_type IN ('string', 'number', 'boolean', 'json')),
  description TEXT,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS billing_codes (
  id SERIAL PRIMARY KEY,
  cpt_code TEXT UNIQUE NOT NULL,
  code_description TEXT NOT NULL,
  code_category TEXT CHECK(code_category IN ('evaluation', 'treatment', 'rpm', 'exercise', 'monitoring')),
  minimum_duration_minutes INTEGER,
  requires_documentation BOOLEAN DEFAULT TRUE,
  is_timed_code BOOLEAN DEFAULT TRUE,
  rpm_device_supplied BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS billable_events (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
  assessment_id INTEGER REFERENCES assessments(id) ON DELETE SET NULL,
  exercise_session_id INTEGER REFERENCES exercise_sessions(id) ON DELETE SET NULL,
  cpt_code_id INTEGER REFERENCES billing_codes(id) ON DELETE RESTRICT,
  service_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  duration_minutes INTEGER,
  units INTEGER DEFAULT 1,
  documentation_ref TEXT,
  status TEXT CHECK(status IN ('pending', 'submitted', 'paid', 'denied')) DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS icd10_codes (
  id SERIAL PRIMARY KEY,
  code TEXT UNIQUE NOT NULL,
  description TEXT NOT NULL,
  category TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exercise_knowledge (
  id SERIAL PRIMARY KEY,
  concept TEXT UNIQUE NOT NULL,
  description TEXT NOT NULL,
  related_exercises TEXT,
  related_icd10 TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS medical_history (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
  surgery_type TEXT CHECK(surgery_type IN ('pre_surgery', 'post_surgery', 'none', 'athletic_performance')),
  surgery_date TEXT,
  surgery_description TEXT,
  conditions TEXT,
  medications TEXT,
  allergies TEXT,
  pain_level INTEGER CHECK(pain_level >= 0 AND pain_level <= 10),
  pain_locations TEXT,
  pain_description TEXT,
  patient_goals TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS movement_analysis (
  id SERIAL PRIMARY KEY,
  test_id INTEGER REFERENCES movement_tests(id) ON DELETE CASCADE,
  joint_angles TEXT,
  rom_measurements TEXT,
  asymmetry_detected TEXT,
  movement_quality_score REAL,
  detected_compensations TEXT,
  recommendations TEXT,
  deficiencies TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exercise_library (
  id SERIAL PRIMARY KEY,
  exercise_name TEXT NOT NULL,
  exercise_category TEXT CHECK(exercise_category IN ('strength', 'flexibility', 'balance', 'mobility', 'stability', 'cardio', 'functional')),
  target_muscles TEXT,
  target_joints TEXT,
  target_movements TEXT,
  description TEXT,
  instructions TEXT,
  video_url TEXT,
  thumbnail_url TEXT,
  difficulty_level TEXT CHECK(difficulty_level IN ('beginner', 'intermediate', 'advanced')),
  equipment_needed TEXT,
  cpt_code_reference TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring_alerts (
  id SERIAL PRIMARY KEY,
  patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
  exercise_session_id INTEGER REFERENCES exercise_sessions(id) ON DELETE SET NULL,
  alert_type TEXT CHECK(alert_type IN ('form_error', 'pain_reported', 'non_compliance', 'progress_milestone', 'concern')),
  alert_severity TEXT CHECK(alert_severity IN ('low', 'medium', 'high', 'critical')),
  alert_message TEXT NOT NULL,
  is_resolved BOOLEAN DEFAULT FALSE,
  resolved_by INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
  resolution_notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  resolved_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    clinician_id INTEGER REFERENCES clinicians(id),
    action TEXT NOT NULL,
    patient_id INTEGER REFERENCES patients(id),
    details TEXT,
    ip_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS phi_access_logs (
    id SERIAL PRIMARY KEY,
    clinician_id INTEGER REFERENCES clinicians(id),
    patient_id INTEGER REFERENCES patients(id),
    access_type TEXT NOT NULL,
    resource TEXT NOT NULL,
    ip_address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
