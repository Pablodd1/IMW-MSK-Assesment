-- Add height and weight columns to patients table
-- These were missing from the original schema but are used in the intake form

ALTER TABLE patients ADD COLUMN height_cm REAL;
ALTER TABLE patients ADD COLUMN weight_kg REAL;
