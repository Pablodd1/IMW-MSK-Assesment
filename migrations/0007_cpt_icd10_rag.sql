-- ============================================================================
-- CPT CODES - Complete Physical Therapy Billing Database (2025 Medicare/Medicaid)
-- ============================================================================

CREATE TABLE IF NOT EXISTS cpt_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cpt_code TEXT NOT NULL UNIQUE,
  code_description TEXT NOT NULL,
  code_category TEXT NOT NULL CHECK(code_category IN (
    'evaluation', 'therapeutic', 'modalities', 'tests_measurements',
    'therapeutic_activities', 'neuromuscular', 'gait', 'manual',
    'group', 'rpm', 'rtm', 'telehealth', 'equipment', 'other'
  )),
  code_subcategory TEXT,
  time_minutes INTEGER,
  unit_type TEXT CHECK(unit_type IN ('per_15_min', 'per_session', 'per_encounter', 'per_hour')),
  medicare_2025_rate REAL,
  is_timed_code BOOLEAN DEFAULT 0,
  is_modality BOOLEAN DEFAULT 0,
  requires_supervision BOOLEAN DEFAULT 0,
  requires_license TEXT,
  commonly_used_for TEXT,
  documentation_requirements TEXT,
  modifiers TEXT,
  ncbi_edits TEXT,
  active BOOLEAN DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- PT Evaluation Codes (97161-97164)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97161', 'Physical Therapy Evaluation - Low Complexity', 'evaluation', 20, 'per_session', 98.01, 0, 'Straightforward patient with 1-2 elements affected', 'Document 2 or fewer elements from: body structure, function, activity limitation'),
('97162', 'Physical Therapy Evaluation - Moderate Complexity', 'evaluation', 30, 'per_session', 98.01, 0, 'Moderate complexity with 3+ elements affected', 'Document 3+ elements from: body structure, function, activity limitation'),
('97163', 'Physical Therapy Evaluation - High Complexity', 'evaluation', 45, 'per_session', 98.01, 0, 'High complexity with significant complexity', 'Document 4+ elements with significant complexity'),
('97164', 'Physical Therapy Re-Evaluation', 'evaluation', 20, 'per_session', 67.60, 0, 'Follow-up evaluation to assess progress', 'Document changes from initial eval, progress toward goals');

-- Therapeutic Exercise (97110)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97110', 'Therapeutic Exercise', 'therapeutic', 15, 'per_15_min', 28.79, 1, 'Exercises to improve strength, endurance, ROM, flexibility', 'Specify exercises, sets, reps, resistance');

-- Neuromuscular Re-education (97112)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97112', 'Neuromuscular Re-education', 'neuromuscular', 15, 'per_15_min', 32.02, 1, 'Restore motor patterns, proprioception, balance', 'Describe movement pattern addressed');

-- Gait Training (97116)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97116', 'Gait Training Therapy', 'gait', 15, 'per_15_min', 28.79, 1, 'Training for ambulation, assistive device use', 'Document gait pattern, assistive device, distance');

-- Manual Therapy (97140)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97140', 'Manual Therapy', 'manual', 15, 'per_15_min', 27.17, 1, 'Joint mobilization, manipulation, soft tissue', 'Specify technique, joint level, grade');

-- Therapeutic Activities (97530)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97530', 'Therapeutic Activities', 'therapeutic_activities', 15, 'per_15_min', 34.61, 1, 'Functional activities, ADL training', 'Describe functional task performed');

-- Self-Care/Home Management (97535)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97535', 'Self-Care/Home Management Training', 'therapeutic_activities', 15, 'per_15_min', 32.02, 1, 'ADL training, home exercise program', 'Document training provided');

-- Group Therapy (97150)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97150', 'Group Therapeutic Procedures', 'group', 15, 'per_session', 17.47, 0, '2+ patients in group', 'Document group size, activities');

-- Physical Performance Testing (97750)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97750', 'Physical Performance Test', 'tests_measurements', 15, 'per_15_min', 22.83, 1, 'ROM, strength, functional testing', 'Specify tests performed, results');

-- Aquatic Therapy (97113)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97113', 'Aquatic Therapy', 'therapeutic', 15, 'per_15_min', 32.02, 1, 'Therapeutic exercises in water', 'Document aquatic approach rationale');

-- Electrical Stimulation (97014)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97014', 'Electrical Stimulation (Unattended)', 'modalities', 15, 'per_session', 18.79, 0, 'TENS, NMES, electrical stimulation', 'Document indication, area treated');

-- Ultrasound (97035)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('97035', 'Ultrasound', 'modalities', 15, 'per_session', 18.79, 0, 'Therapeutic ultrasound', 'Document indication, area treated');

-- RPM Codes (99453, 99454, 99457, 99458)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('99453', 'Remote Patient Monitoring - Setup', 'rpm', 16, 'per_session', 0, 0, 'Initial setup and patient education', 'Document device provision, training time'),
('99454', 'Remote Patient Monitoring - Device Supply', 'rpm', 16, 'per_30_days', 50.00, 0, 'Monthly device supply (16+ days)', 'Document days device was available'),
('99457', 'Remote Patient Monitoring - Treatment Management', 'rpm', 20, 'per_calendar_month', 60.00, 1, 'First 20 minutes of monitoring', 'Document time spent on review/communication'),
('99458', 'Remote Patient Monitoring - Additional Time', 'rpm', 20, 'per_calendar_month', 40.00, 1, 'Each additional 20 minutes', 'Document cumulative time');

-- RTM Codes (98975, 98976, 98977, 98978)
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('98975', 'Remote Therapeutic Monitoring - Setup', 'rtm', 16, 'per_session', 0, 0, 'RTM equipment setup and education', 'Document equipment provided'),
('98976', 'Remote Therapeutic Monitoring - Device Supply', 'rtm', 16, 'per_30_days', 50.00, 0, 'Respiratory device supply (16+ days)', 'Document days device used'),
('98977', 'Remote Therapeutic Monitoring - Treatment', 'rtm', 20, 'per_calendar_month', 60.00, 1, 'First 20 minutes of RTM services', 'Document treatment time'),
('98978', 'Remote Therapeutic Monitoring - Additional', 'rtm', 20, 'per_calendar_month', 40.00, 1, 'Additional 20 minutes RTM', 'Document cumulative time');

-- Telehealth Codes
INSERT INTO cpt_codes (cpt_code, code_description, code_category, time_minutes, unit_type, medicare_2025_rate, is_timed_code, commonly_used_for, documentation_requirements) VALUES
('98960', 'Telehealth Education - Individual', 'telehealth', 30, 'per_session', 45.00, 0, 'Telehealth education services', 'Document virtual session'),
('99441', 'Telehealth Consult - 15 min', 'telehealth', 15, 'per_session', 35.00, 0, 'Virtual check-in, 15 min', 'Document time, topics discussed'),
('99442', 'Telehealth Consult - 25 min', 'telehealth', 25, 'per_session', 70.00, 0, 'Virtual check-in, 25 min', 'Document time, topics discussed'),
('99443', 'Telehealth Consult - 40 min', 'telehealth', 40, 'per_session', 95.00, 0, 'Virtual check-in, 40 min', 'Document time, topics discussed');

-- ============================================================================
-- ICD-10 DX CODES - Complete Diagnosis Codes for Physical Therapy
-- ============================================================================

CREATE TABLE IF NOT EXISTS icd10_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  icd10_code TEXT NOT NULL UNIQUE,
  code_description TEXT NOT NULL,
  code_category TEXT NOT NULL,
  body_region TEXT,
  chapter TEXT,
  is_billable BOOLEAN DEFAULT 1,
  requires_laterality BOOLEAN DEFAULT 0,
  requires_7th_char BOOLEAN DEFAULT 0,
  commonly_used_for TEXT,
  primary_treatment_approach TEXT,
  typical_cpt_codes TEXT,
  notes TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Low Back Pain (M54)
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, commonly_used_for, typical_cpt_codes) VALUES
('M54.5', 'Low back pain', 'pain', 'lumbar_spine', 'M45-M54', 'Non-specific LBP', '97110, 97140'),
('M54.16', 'Radiculopathy, lumbar region', 'radiculopathy', 'lumbar_spine', 'M45-M54', 'Lumbar radiculopathy', '97110, 97112, 97140'),
('M54.41', 'Lumbago with sciatica, right side', 'radiculopathy', 'lumbar_spine', 'M45-M54', 'Sciatica right', '97110, 97112, 97140'),
('M54.42', 'Lumbago with sciatica, left side', 'radiculopathy', 'lumbar_spine', 'M45-M54', 'Sciatica left', '97110, 97112, 97140'),
('M54.4', 'Low back pain with sciatica', 'radiculopathy', 'lumbar_spine', 'M45-M54', 'LBP with sciatica', '97110, 97112, 97140');

-- Neck Pain (M54)
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, commonly_used_for, typical_cpt_codes) VALUES
('M54.2', 'Cervicalgia', 'pain', 'cervical_spine', 'M45-M54', 'Neck pain', '97110, 97140'),
('M54.12', 'Radiculopathy, cervical region', 'radiculopathy', 'cervical_spine', 'M45-M54', 'Cervical radiculopathy', '97110, 97112, 97140');

-- Shoulder (M75)
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, requires_laterality, commonly_used_for, typical_cpt_codes) VALUES
('M75.100', 'Unspecified rotator cuff tear, unspecified shoulder', 'tear', 'shoulder', 'M70-M79', 1, 'RTC tear', '97110, 97140'),
('M75.101', 'Unspecified rotator cuff tear, right shoulder', 'tear', 'shoulder', 'M70-M79', 0, 'RTC tear R', '97110, 97140'),
('M75.102', 'Unspecified rotator cuff tear, left shoulder', 'tear', 'shoulder', 'M70-M79', 0, 'RTC tear L', '97110, 97140'),
('M75.51', 'Impingement syndrome of right shoulder', 'impingement', 'shoulder', 'M70-M79', 0, 'Shoulder impingement R', '97110, 97140'),
('M75.52', 'Impingement syndrome of left shoulder', 'impingement', 'shoulder', 'M70-M79', 0, 'Shoulder impingement L', '97110, 97140'),
('M75.81', 'Other shoulder lesions, right shoulder', 'other', 'shoulder', 'M70-M79', 0, 'Other shoulder R', '97110, 97140'),
('M75.82', 'Other shoulder lesions, left shoulder', 'other', 'shoulder', 'M70-M79', 0, 'Other shoulder L', '97110, 97140'),
('M75.10', 'Unspecified rotator cuff sprain/strain', 'strain', 'shoulder', 'M70-M79', 1, 'RTC strain', '97110, 97140');

-- Knee (M25, S83)
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, requires_laterality, commonly_used_for, typical_cpt_codes) VALUES
('M25.561', 'Pain in right knee', 'pain', 'knee', 'M20-M25', 0, 'Knee pain R', '97110, 97116'),
('M25.562', 'Pain in left knee', 'pain', 'knee', 'M20-M25', 0, 'Knee pain L', '97110, 97116'),
('M76.50', 'Unspecified Achilles tendinitis', 'tendinitis', 'knee', 'M70-M79', 1, 'Achilles tendinitis', '97110, 97140'),
('M76.51', 'Achilles tendinitis, right leg', 'tendinitis', 'knee', 'M70-M79', 0, 'Achilles R', '97110, 97140'),
('M76.52', 'Achilles tendinitis, left leg', 'tendinitis', 'knee', 'M70-M79', 0, 'Achilles L', '97110, 97140'),
('S83.401A', 'Sprain of unspecified ligament of right knee, initial', 'sprain', 'knee', 'S80-S89', 0, 'Knee ligament sprain R', '97110, 97116, 97140'),
('S83.402A', 'Sprain of unspecified ligament of left knee, initial', 'sprain', 'knee', 'S80-S89', 0, 'Knee ligament sprain L', '97110, 97116, 97140');

-- Hip (M25)
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, requires_laterality, commonly_used_for, typical_cpt_codes) VALUES
('M25.551', 'Pain in right hip', 'pain', 'hip', 'M20-M25', 0, 'Hip pain R', '97110, 97140'),
('M25.552', 'Pain in left hip', 'pain', 'hip', 'M20-M25', 0, 'Hip pain L', '97110, 97140'),
('M24.051', 'Contracture of right hip joint', 'contracture', 'hip', 'M20-M25', 0, 'Hip contracture R', '97110, 97140'),
('M24.052', 'Contracture of left hip joint', 'contracture', 'hip', 'M20-M25', 0, 'Hip contracture L', '97110, 97140');

-- Ankle/Foot
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, commonly_used_for, typical_cpt_codes) VALUES
('M25.571', 'Pain in right ankle', 'pain', 'ankle', 'M20-M25', 'Ankle pain R', '97110'),
('M25.572', 'Pain in left ankle', 'pain', 'ankle', 'M20-M25', 'Ankle pain L', '97110'),
('S93.401A', 'Sprain of unspecified ligament of right ankle, initial', 'sprain', 'ankle', 'S80-S89', 'Ankle sprain R', '97110, 97140'),
('S93.402A', 'Sprain of unspecified ligament of left ankle, initial', 'sprain', 'ankle', 'S80-S89', 'Ankle sprain L', '97110, 97140');

-- Wrist/Hand
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, commonly_used_for, typical_cpt_codes) VALUES
('M25.531', 'Pain in right wrist', 'pain', 'wrist', 'M20-M25', 'Wrist pain R', '97110'),
('M25.532', 'Pain in left wrist', 'pain', 'wrist', 'M20-M25', 'Wrist pain L', '97110'),
('M75.40', 'Unspecified carpal tunnel syndrome', 'nerve', 'wrist', 'M70-M79', 'Carpal tunnel', '97110, 97140');

-- Elbow
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, commonly_used_for, typical_cpt_codes) VALUES
('M25.521', 'Pain in right elbow', 'pain', 'elbow', 'M20-M25', 'Elbow pain R', '97110'),
('M25.522', 'Pain in left elbow', 'pain', 'elbow', 'M20-M25', 'Elbow pain L', '97110'),
('M77.10', 'Unspecified epicondylitis', 'tendinopathy', 'elbow', 'M70-M79', 'Epicondylitis', '97110, 97140');

-- Post-Surgical (Z47)
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, commonly_used_for, typical_cpt_codes) VALUES
('Z47.1', 'Aftercare following joint replacement surgery', 'post_surgical', 'general', 'Z70-Z99', 'Post joint replacement', '97110, 97140'),
('Z47.89', 'Encounter for other orthopedic aftercare', 'post_surgical', 'general', 'Z70-Z99', 'Other orthopedic aftercare', '97110, 97140'),
('Z96.641', 'Presence of right artificial hip joint', 'prosthesis', 'hip', 'Z80-Z99', 'Right hip prosthesis', '97110'),
('M96.1', 'Postlaminectomy syndrome', 'post_surgical', 'spine', 'M90-M99', 'Post back surgery', '97110, 97140');

-- Neurological
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, commonly_used_for, typical_cpt_codes) VALUES
('R26.2', 'Difficulty in walking', 'gait', 'general', 'R00-R99', 'Gait abnormality', '97116'),
('R26.81', 'Unsteadiness on feet', 'balance', 'general', 'R00-R99', 'Balance issues', '97110, 97116'),
('G57.91', 'Unspecified sciatica', 'nerve', 'leg', 'G50-G59', 'Sciatica', '97110, 97112'),
('R29.6', 'Repeated falls', 'fall_risk', 'general', 'R00-R99', 'Fall risk', '97110, 97116');

-- General Pain
INSERT INTO icd10_codes (icd10_code, code_description, code_category, body_region, chapter, commonly_used_for, typical_cpt_codes) VALUES
('M79.1', 'Myalgia', 'pain', 'general', 'M70-M79', 'Muscle pain', '97110'),
('M79.7', 'Fibromyalgia', 'pain', 'general', 'M70-M79', 'Fibromyalgia', '97110, 97140'),
('M62.81', 'Muscle weakness (generalized)', 'weakness', 'general', 'M60-M63', 'Generalized weakness', '97110'),
('R52', 'Pain, unspecified', 'pain', 'general', 'R00-R99', 'Unspecified pain', '97110');

-- ============================================================================
-- CLINICAL GUIDELINES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS clinical_guidelines (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  guideline_title TEXT NOT NULL,
  specialty TEXT NOT NULL CHECK(specialty IN ('physical_therapy', 'chiropractic', 'sports_medicine', 'occupational_therapy', 'sports_pt')),
  content TEXT NOT NULL,
  source TEXT,
  evidence_level TEXT CHECK(evidence_level IN ('A', 'B', 'C', 'D')),
  topic TEXT,
  body_region TEXT,
  recommendations TEXT,
  updated_at DATE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- FMS Guidelines
INSERT INTO clinical_guidelines (guideline_title, specialty, content, source, evidence_level, topic, body_region) VALUES
('FMS Deep Squat Scoring', 'physical_therapy', 'Score 3: Heels flat, knees past toes, torso parallel to tibia, dowel over feet. Score 2: Heels flat, knees past toes, torso above parallel. Score 1: Unable to perform with compensation or unable to complete.', 'Gray Cook FMS', 'A', 'mobility', 'full_body'),
('FMS Hurdle Step Scoring', 'physical_therapy', 'Score 3: Clear hurdle without compensation, touch heel down. Score 2: Clear hurdle with compensation. Score 1: Unable or shows compensation.', 'Gray Cook FMS', 'A', 'stability', 'hip'),
('FMS Inline Lunge Scoring', 'physical_therapy', 'Score 3: Dowel maintains contact with sacrum and head, torso upright, back knee touches ground. Score 2: Dowel contact lost or torso leans. Score 1: Unable to complete.', 'Gray Cook FMS', 'A', 'stability', 'hip'),
('Shoulder Mobility Scoring', 'physical_therapy', 'Score 3: Fists within 1 hand length. Score 2: Fists within 1.5 hand lengths. Score 1: Fists beyond 1.5 hand lengths.', 'Gray Cook FMS', 'A', 'mobility', 'shoulder');

-- Evidence-Based Protocols
INSERT INTO clinical_guidelines (guideline_title, specialty, content, source, evidence_level, topic, body_region) VALUES
('ACL Rehabilitation Protocol', 'sports_pt', 'Phase 1 (0-2 weeks): ROM 0-90, quad activation, no weight bearing. Phase 2 (2-6 weeks): ROM 0-125, progressive weight bearing. Phase 3 (6-12 weeks): Full ROM, strengthening. Phase 4 (12+ weeks): Return to sport progression.', 'APTA', 'A', 'rehabilitation', 'knee'),
('Low Back Pain Classification', 'physical_therapy', 'Based on treatment-based classification: Mobility impairment (flexion exercises), Stabilization (motor control), Directional preference (extension exercises), Specific exercise', 'Delitto et al 2015', 'A', 'classification', 'lumbar_spine'),
('Rotator Cuff Conservative Management', 'physical_therapy', 'Phase 1: Pain control, ROM (passive to active-assisted). Phase 2: Strengthening (isometric to isotonic). Phase 3: Functional progression. Duration: 6-12 weeks depending on severity.', 'AAOS', 'B', 'rehabilitation', 'shoulder'),
('Patellofemoral Pain Syndrome Treatment', 'physical_therapy', 'Quadriceps strengthening, hip abductor strengthening, patellar taping, activity modification. Evidence supports combined approach over single intervention.', 'Boling et al', 'A', 'rehabilitation', 'knee');

-- ============================================================================
-- CONTRAINDICATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS contraindications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  condition TEXT NOT NULL,
  exercise_name TEXT NOT NULL,
  risk_level TEXT NOT NULL CHECK(risk_level IN ('low', 'moderate', 'high', 'contraindicated')),
  reason TEXT,
  alternative_exercise TEXT,
  body_region TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO contraindications (condition, exercise_name, risk_level, reason, alternative_exercise, body_region) VALUES
('Rotator cuff tear', 'Shoulder flexion with resistance', 'contraindicated', 'May worsen tear', 'Pendulum exercises', 'shoulder'),
('Acute low back pain', 'Forward bending exercises', 'high', 'May increase pain', 'McKenzie extension', 'lumbar_spine'),
('Osteoporosis', 'Forward flexion exercises', 'contraindicated', 'Fracture risk', 'Extension exercises', 'spine'),
('ACL reconstruction', 'Open chain knee extension >90', 'contraindicated', ' Graft strain risk', 'Closed chain leg press', 'knee'),
('Ankylosing spondylitis', 'Deep flexion stretching', 'high', 'May cause fracture', 'Extension focused', 'spine'),
('Carpal tunnel syndrome', 'Wrist flexion exercises', 'contraindicated', 'Increases compression', 'Nerve gliding', 'wrist'),
('Hip replacement', 'Hip flexion >90 with resistance', 'contraindicated', 'Dislocation risk', 'Hip abduction', 'hip'),
('Patellar tendinopathy', 'Deep squat', 'moderate', 'Aggravates tendon', 'Shallow squat', 'knee'),
('Balance disorder', 'Single leg stance without support', 'high', 'Fall risk', 'Tandem stance with support', 'general');

-- ============================================================================
-- NORMATIVE JOINT ANGLES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS normative_joint_angles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  joint_name TEXT NOT NULL,
  movement TEXT NOT NULL,
  min_normal REAL NOT NULL,
  max_normal REAL NOT NULL,
  age_group TEXT CHECK(age_group IN ('18-30', '31-50', '51-70', '70+', 'all')),
  gender TEXT CHECK(gender IN ('male', 'female', 'neutral', 'all')),
  measurement_method TEXT,
  population_source TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO normative_joint_angles (joint_name, movement, min_normal, max_normal, age_group, gender, measurement_method) VALUES
('shoulder', 'flexion', 160, 180, 'all', 'all', 'goniometer'),
('shoulder', 'extension', 50, 60, 'all', 'all', 'goniometer'),
('shoulder', 'abduction', 150, 180, 'all', 'all', 'goniometer'),
('elbow', 'flexion', 130, 150, 'all', 'all', 'goniometer'),
('elbow', 'extension', 0, 10, 'all', 'all', 'goniometer'),
('hip', 'flexion', 110, 125, '18-30', 'neutral', 'goniometer'),
('hip', 'flexion', 100, 120, '31-50', 'neutral', 'goniometer'),
('hip', 'flexion', 90, 110, '51-70', 'neutral', 'goniometer'),
('hip', 'extension', 10, 30, 'all', 'all', 'goniometer'),
('hip', 'abduction', 30, 50, 'all', 'all', 'goniometer'),
('hip', 'adduction', 20, 30, 'all', 'all', 'goniometer'),
('knee', 'flexion', 130, 150, '18-30', 'neutral', 'goniometer'),
('knee', 'flexion', 120, 145, '31-50', 'neutral', 'goniometer'),
('knee', 'flexion', 110, 140, '51-70', 'neutral', 'goniometer'),
('knee', 'extension', 0, 10, 'all', 'all', 'goniometer'),
('ankle', 'dorsiflexion', 10, 25, 'all', 'all', 'goniometer'),
('ankle', 'plantarflexion', 45, 60, 'all', 'all', 'goniometer'),
('lumbar_spine', 'flexion', 40, 60, 'all', 'neutral', 'goniometer'),
('lumbar_spine', 'extension', 20, 35, 'all', 'neutral', 'goniometer'),
('lumbar_spine', 'lateral_flexion', 15, 30, 'all', 'neutral', 'goniometer'),
('cervical_spine', 'flexion', 45, 60, 'all', 'all', 'goniometer'),
('cervical_spine', 'extension', 45, 70, 'all', 'all', 'goniometer');

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_cpt_category ON cpt_codes(code_category);
CREATE INDEX IF NOT EXISTS idx_icd10_body_region ON icd10_codes(body_region);
CREATE INDEX IF NOT EXISTS idx_guidelines_specialty ON clinical_guidelines(specialty);
CREATE INDEX IF NOT EXISTS idx_normatives_joint ON normative_joint_angles(joint_name);
