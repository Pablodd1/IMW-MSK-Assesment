-- PhysioMotion Mock Patient Data Insertion
-- Complete profiles for testing the biomechanical analysis system

-- Delete existing test data first
DELETE FROM prescribed_exercises;
DELETE FROM prescriptions;
DELETE FROM exercise_sessions;
DELETE FROM exercise_performances;
DELETE FROM movement_tests;
DELETE FROM assessments;
DELETE FROM patients WHERE id > 1;

-- Emma Rodriguez (8-year-old with Developmental Coordination Disorder)
INSERT INTO patients (
  first_name, last_name, date_of_birth, gender, email, phone,
  address_line1, city, state, zip_code, country,
  emergency_contact_name, emergency_contact_phone, emergency_contact_relationship,
  primary_physician, insurance_provider, insurance_policy_number,
  medical_history, current_medications, allergies,
  assessment_reason, chief_complaint, pain_scale, activity_level,
  height_cm, weight_kg
) VALUES (
  'Emma', 'Rodriguez', '2017-03-15', 'female', 'parent.rodriguez@email.com', '(555) 234-5678',
  '456 Maple Street, Apt 3B', 'Springfield', 'IL', '62701', 'USA',
  'Maria Rodriguez (Mother)', '(555) 234-5679', 'parent',
  'Dr. Michael Chen (Pediatrics)', 'Blue Cross Blue Shield', 'BC123456789',
  '{"diagnoses": ["Developmental Coordination Disorder (DCD)", "Mild hypotonia", "Delayed gross motor development"], "past_medical_history": "Born full-term, normal delivery. Met early milestones slightly delayed (walked at 15 months). No major illnesses or hospitalizations. Up to date on vaccinations.", "family_history": {"mother": "History of ADHD", "father": "Healthy", "maternal_uncle": "Dyspraxia"}, "prior_therapy": {"occupational_therapy": "6 months (ongoing)", "physical_therapy": "3 sessions (discontinued due to insurance)"}, "social_history": {"living_situation": "Lives with both parents and younger brother (age 5)", "school": "Attends 3rd grade at local elementary school", "interests": "Enjoys art and reading, avoids sports activities"}}',
  '["None currently"]',
  '["No known drug allergies (NKDA)", "Mild seasonal allergies (pollen)"]',
  'general_wellness', 'Trouble with coordination and balance. Falls frequently during play and has difficulty with activities like catching a ball or riding a bicycle.', 1, 'light',
  127, 25.5
);

-- Marcus Thompson (28-year-old ACL reconstruction)
INSERT INTO patients (
  first_name, last_name, date_of_birth, gender, email, phone,
  address_line1, city, state, zip_code, country,
  emergency_contact_name, emergency_contact_phone, emergency_contact_relationship,
  primary_physician, insurance_provider, insurance_policy_number,
  medical_history, current_medications, allergies,
  assessment_reason, chief_complaint, pain_scale, activity_level,
  height_cm, weight_kg
) VALUES (
  'Marcus', 'Thompson', '1997-08-22', 'male', 'marcus.thompson@email.com', '(555) 789-0123',
  '892 Oak Avenue', 'Denver', 'CO', '80202', 'USA',
  'Jessica Thompson (Wife)', '(555) 789-4322', 'spouse',
  'Dr. Amanda Foster, MD (Orthopedic Surgery)', 'Aetna PPO', 'AT987654321',
  '{"surgery": {"procedure": "ACL reconstruction with hamstring autograft", "date": "December 20, 2025", "surgeon": "Dr. Amanda Foster, MD", "weeks_post_op": 6}, "injury_mechanism": "Non-contact injury during soccer match, felt pop and immediate instability", "athletic_history": "Played soccer since age 8, college varsity (Division II), currently semi-professional"}',
  '["Ibuprofen 600mg TID PRN", "Acetaminophen 500mg PRN (no opioids)"]',
  '["Penicillin (rash)"]',
  'post_surgery', 'Had ACL surgery 6 weeks ago and need to get back to playing soccer. Knee feels unstable and having trouble with exercises.', 4, 'moderate',
  185, 88
);

-- Margaret Chen (72-year-old with fall risk and multiple comorbidities)
INSERT INTO patients (
  first_name, last_name, date_of_birth, gender, email, phone,
  address_line1, city, state, zip_code, country,
  emergency_contact_name, emergency_contact_phone, emergency_contact_relationship,
  primary_physician, insurance_provider, insurance_policy_number,
  medical_history, current_medications, allergies,
  assessment_reason, chief_complaint, pain_scale, activity_level,
  height_cm, weight_kg
) VALUES (
  'Margaret', 'Chen', '1953-11-08', 'female', 'maggie.chen@email.com', '(555) 456-7890',
  'Senior Living Community, Apt 3C', 'Portland', 'OR', '97201', 'USA',
  'David Chen (Son)', '(555) 456-7891', 'child',
  'Dr. Robert Kim, MD (Geriatrics)', 'Medicare Part B + AARP Supplement', 'MEDICARE123456789A',
  '{"current_diagnoses": ["Osteoarthritis (bilateral knees - severe, lumbar spine - moderate)", "History of 2 falls in past month (HIGH FALL RISK)", "Sarcopenia (age-related muscle loss)", "Hypertension (controlled)", "Type 2 Diabetes Mellitus (HbA1c 6.8% - well-controlled)", "Osteoporosis (on bisphosphonate therapy)", "Peripheral neuropathy (mild, feet - diabetic)", "Mild cognitive impairment (early stage)", "Compression fracture L1 vertebra (2020 - healed)"], "fall_history": [{"date": "December 28, 2025", "location": "Bathroom while turning", "injuries": "Bruised left hip/elbow"}, {"date": "January 15, 2026", "location": "Tripped over rug edge in kitchen", "injuries": "Bruised right knee/palm"}], "living_situation": {"residence": "Senior living community (independent apartment, 3rd floor)", "assistance": "Uses cane for outdoor walking (inconsistent indoors)", "safety": "Has medical alert pendant"}}',
  '["Lisinopril 20mg (hypertension)", "Metformin 1000mg BID (diabetes)", "Atorvastatin 40mg (cholesterol)", "Alendronate 70mg weekly (osteoporosis)", "Calcium + Vitamin D3", "Aspirin 81mg", "Meloxicam 15mg PRN (knee pain - uses 4-5x/week)", "Acetaminophen 650mg TID PRN", "Gabapentin 300mg nightly (neuropathy)", "Multivitamin", "Vitamin B12 1000mcg (neuropathy)"]',
  '["None known"]',
  'general_wellness', 'Having more trouble getting around. Knees hurt, especially stairs. Fallen twice in the past month and afraid of falling again. Want to keep living independently.', 6, 'light',
  158, 68
);