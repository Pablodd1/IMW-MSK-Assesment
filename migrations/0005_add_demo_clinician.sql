-- Add demo clinician account for testing
-- Password: demo123
-- Hash generated using SHA-256 with salt: 'physiomotion-salt-2025'

INSERT INTO clinicians (
    email,
    password_hash,
    first_name,
    last_name,
    title,
    license_number,
    license_state,
    npi_number,
    phone,
    clinic_name,
    role,
    active
) VALUES (
    'demo@physiomotion.com',
    'c0b87a0fe3f3c3d9e43cd5d4d4d1048506a5f9f77bc10110a6c418ed224813fa',
    'Demo',
    'Clinician',
    'DPT',
    'PT123456',
    'CA',
    '1234567890',
    '(555) 123-4567',
    'PhysioMotion Demo Clinic',
    'clinician',
    1
);
