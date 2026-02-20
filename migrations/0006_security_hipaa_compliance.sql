-- Migration: Add Security & HIPAA Compliance Tables
-- Run this to add audit logging, salt columns, and other security enhancements

-- 1. Add salt column to clinicians for better password security
ALTER TABLE clinicians ADD COLUMN salt TEXT;

-- 2. Add last_activity column for session tracking
ALTER TABLE clinicians ADD COLUMN last_activity DATETIME;

-- 3. Add role column if not exists
ALTER TABLE clinicians ADD COLUMN role TEXT DEFAULT 'clinician' 
  CHECK(role IN ('admin', 'clinician', 'assistant', 'viewer'));

-- 4. Create audit_logs table for HIPAA compliance
CREATE TABLE IF NOT EXISTS audit_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clinician_id INTEGER,
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id TEXT,
  ip_address TEXT,
  user_agent TEXT,
  http_method TEXT,
  http_status INTEGER,
  duration_ms INTEGER,
  success INTEGER DEFAULT 1,
  details TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (clinician_id) REFERENCES clinicians(id) ON DELETE SET NULL
);

-- 5. Create indexes for audit_logs
CREATE INDEX IF NOT EXISTS idx_audit_clinician ON audit_logs(clinician_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at);

-- 6. Add missing columns to patients table
ALTER TABLE patients ADD COLUMN height_cm REAL;
ALTER TABLE patients ADD COLUMN weight_kg REAL;
ALTER TABLE patients ADD COLUMN blood_type TEXT;

-- 7. Add test_status column if missing
ALTER TABLE movement_tests ADD COLUMN status TEXT DEFAULT 'pending' 
  CHECK(status IN ('pending', 'in_progress', 'completed', 'skipped'));

-- 8. Create billing_codes table if not exists
CREATE TABLE IF NOT EXISTS billing_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cpt_code TEXT NOT NULL UNIQUE,
  code_description TEXT NOT NULL,
  code_category TEXT CHECK(code_category IN ('evaluation', 'treatment', 'rpm', 'exercise', 'monitoring')),
  minimum_duration_minutes INTEGER,
  requires_documentation INTEGER DEFAULT 1,
  is_rpm_code INTEGER DEFAULT 0,
  rpm_time_requirement_minutes INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 9. Create billable_events table if not exists
CREATE TABLE IF NOT EXISTS billable_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  assessment_id INTEGER,
  exercise_session_id INTEGER,
  cpt_code_id INTEGER NOT NULL,
  service_date DATE NOT NULL,
  duration_minutes INTEGER,
  clinical_note TEXT,
  medical_necessity TEXT,
  billing_status TEXT DEFAULT 'pending' 
    CHECK(billing_status IN ('pending', 'submitted', 'paid', 'denied')),
  provider_id INTEGER,
  provider_npi TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
  FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE SET NULL,
  FOREIGN KEY (exercise_session_id) REFERENCES exercise_sessions(id) ON DELETE SET NULL,
  FOREIGN KEY (cpt_code_id) REFERENCES billing_codes(id),
  FOREIGN KEY (provider_id) REFERENCES clinicians(id)
);

-- 10. Create indexes for billing
CREATE INDEX IF NOT EXISTS idx_billable_patient ON billable_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_billable_status ON billable_events(billing_status);
CREATE INDEX IF NOT EXISTS idx_billable_date ON billable_events(service_date);
