-- Add test_status column alias for backwards compatibility
-- The schema uses 'status' but the code references 'test_status'

ALTER TABLE movement_tests ADD COLUMN test_status TEXT CHECK(test_status IN ('pending', 'recording', 'completed', 'failed')) DEFAULT 'pending';

-- Update existing rows to copy status to test_status
UPDATE movement_tests SET test_status = status WHERE test_status IS NULL;
