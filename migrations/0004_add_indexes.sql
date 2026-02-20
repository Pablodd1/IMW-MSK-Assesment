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

-- Prescribed exercise indexes (Using columns that exist in 0001_initial_schema.sql)
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_prescription_id ON prescribed_exercises(prescription_id);
CREATE INDEX IF NOT EXISTS idx_prescribed_exercises_status ON prescribed_exercises(status);

-- Exercise library indexes
CREATE INDEX IF NOT EXISTS idx_exercises_category ON exercises(category);
CREATE INDEX IF NOT EXISTS idx_exercises_difficulty ON exercises(difficulty);
