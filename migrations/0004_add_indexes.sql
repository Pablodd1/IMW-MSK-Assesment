-- Add performance indexes for frequently queried columns
-- This migration improves query performance for patient lookups, assessment retrieval, and monitoring queries

-- Patient indexes
CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email);
CREATE INDEX IF NOT EXISTS idx_patients_created_at ON patients(created_at DESC);

-- Assessment indexes
CREATE INDEX IF NOT EXISTS idx_assessments_patient_id ON assessments(patient_id);
CREATE INDEX IF NOT EXISTS idx_assessments_date ON assessments(assessment_date DESC);
CREATE INDEX IF NOT EXISTS idx_assessments_status ON assessments(status);
CREATE INDEX IF NOT EXISTS idx_assessments_patient_status ON assessments(patient_id, status);

-- Movement test indexes
CREATE INDEX IF NOT EXISTS idx_movement_tests_assessment_id ON movement_tests(assessment_id);
CREATE INDEX IF NOT EXISTS idx_movement_tests_status ON movement_tests(status);

-- Prescribed exercise indexes
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_patient_id ON prescribed_exercises(patient_id);
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_assessment_id ON prescribed_exercises(assessment_id);
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_status ON prescribed_exercises(status);
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_patient_status ON prescribed_exercises(patient_id, status);

-- Exercise session indexes
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_patient_id ON exercise_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_prescribed_exercise_id ON exercise_sessions(prescribed_exercise_id);
CREATE INDEX IF NOT EXISTS idx_exercise_sessions_date ON exercise_sessions(session_date DESC);

-- Exercise library indexes
CREATE INDEX IF NOT EXISTS idx_exercises_category ON exercises(category);
CREATE INDEX IF NOT EXISTS idx_exercises_difficulty ON exercises(difficulty);

-- Billable event indexes
CREATE INDEX IF NOT EXISTS idx_billable_events_patient_id ON billable_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_billable_events_service_date ON billable_events(service_date DESC);
CREATE INDEX IF NOT EXISTS idx_billable_events_status ON billable_events(billing_status);
