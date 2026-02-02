-- Add missing columns to prescribed_exercises
ALTER TABLE prescribed_exercises ADD COLUMN compliance_percentage REAL DEFAULT 0;
ALTER TABLE prescribed_exercises ADD COLUMN last_performed_at DATETIME;
ALTER TABLE prescribed_exercises ADD COLUMN prescribed_at DATETIME DEFAULT CURRENT_TIMESTAMP;
