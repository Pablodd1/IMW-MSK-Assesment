-- Comprehensive exercise prescriptions for mock patients
-- Based on individual biomechanical analysis results

-- Emma Rodriguez Exercise Prescription (8-year-old with DCD)
INSERT INTO prescriptions (
  patient_id, assessment_id, clinician_id, prescription_date, start_date, end_date, status,
  program_name, program_goals, frequency_per_week, estimated_duration_minutes,
  clinician_notes, patient_instructions, precautions
) VALUES (
  2, (SELECT id FROM assessments WHERE patient_id = 2 ORDER BY id DESC LIMIT 1), 1, '2026-01-31 10:30:00', '2026-02-01', '2026-03-28', 'active',
  'Pediatric Balance and Coordination Program',
  '["Improve balance and postural control", "Increase lower extremity strength", "Enhance ankle mobility", "Develop body awareness and proprioception", "Build confidence in movement", "Achieve age-appropriate motor skills"]',
  3, 45,
  '8-week comprehensive program for developmental coordination disorder. Focus on fun, engaging activities appropriate for 8-year-old. Parent involvement essential for carryover.',
  'Practice these exercises 3 times per week with parent supervision. Make it fun - use games, music, and rewards. Stop if you feel pain or get too tired.',
  'Parent supervision required. Use stable support when needed. Stop exercises if painful. Contact therapist if falls increase.'
);

-- Emma - Specific prescribed exercises
INSERT INTO prescribed_exercises (
  prescription_id, exercise_id, sets, reps, hold_time, rest_time, frequency_per_week, exercise_order, current_level, progression_criteria, status
) VALUES
((SELECT id FROM prescriptions WHERE patient_id = 2 ORDER BY id DESC LIMIT 1), 10, 3, 10, 30, 60, 3, 1, 'beginner', 'Progress to 60 seconds hold, then reduce support', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 2 ORDER BY id DESC LIMIT 1), 7, 3, 15, 30, 60, 3, 2, 'beginner', 'Increase dorsiflexion angle, progress to wall lean', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 2 ORDER BY id DESC LIMIT 1), 16, 3, 8, 0, 90, 3, 3, 'beginner', 'Progress to unsupported, increase depth', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 2 ORDER BY id DESC LIMIT 1), 5, 2, 10, 0, 60, 3, 4, 'beginner', 'Add resistance band, increase reps', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 2 ORDER BY id DESC LIMIT 1), 13, 2, 6, 5, 60, 3, 5, 'beginner', 'Increase hold time, add arm movements', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 2 ORDER BY id DESC LIMIT 1), 10, 2, 60, 0, 60, 3, 6, 'beginner', 'Progress to unstable surface, eyes closed', 'active');

-- Marcus Thompson Exercise Prescription (ACL Reconstruction)
INSERT INTO prescriptions (
  patient_id, assessment_id, clinician_id, prescription_date, start_date, end_date, status,
  program_name, program_goals, frequency_per_week, estimated_duration_minutes,
  clinician_notes, patient_instructions, precautions
) VALUES (
  3, (SELECT id FROM assessments WHERE patient_id = 3 ORDER BY id DESC LIMIT 1), 1, '2026-01-31 14:30:00', '2026-02-01', '2026-04-25', 'active',
  'ACL Reconstruction Phase 1 Rehabilitation',
  '["Restore quadriceps strength", "Improve proprioception and neuromuscular control", "Normalize movement patterns", "Progress ROM to full flexion", "Gradual return to sport-specific activities"]',
  2, 90,
  'Phase 1 (weeks 6-12 post-op): Focus on quadriceps strengthening and proprioception. Progress based on pain-free movement and physician clearance.',
  'Perform exercises 2x daily as prescribed. Stop if sharp pain occurs. Use ice after exercises if swollen. Progress gradually - do not rush.',
  'No running/jumping until cleared. Stop if sharp pain. Monitor for increased swelling. Follow physician restrictions.'
);

-- Marcus - Specific prescribed exercises
INSERT INTO prescribed_exercises (
  prescription_id, exercise_id, sets, reps, hold_time, rest_time, frequency_per_week, exercise_order, current_level, progression_criteria, status
) VALUES
((SELECT id FROM prescriptions WHERE patient_id = 3 ORDER BY id DESC LIMIT 1), 1, 3, 15, 10, 60, 14, 1, 'intermediate', 'Add ankle weights, increase to 20 reps', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 3 ORDER BY id DESC LIMIT 1), 7, 3, 20, 0, 60, 14, 2, 'intermediate', 'Add resistance band, increase reps', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 3 ORDER BY id DESC LIMIT 1), 5, 3, 10, 0, 60, 10, 3, 'intermediate', 'Add resistance band, increase steps', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 3 ORDER BY id DESC LIMIT 1), 16, 3, 12, 0, 90, 10, 4, 'intermediate', 'Progress to single leg, add weight', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 3 ORDER BY id DESC LIMIT 1), 10, 3, 30, 0, 60, 14, 5, 'intermediate', 'Progress to unstable surface, eyes closed', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 3 ORDER BY id DESC LIMIT 1), 17, 3, 15, 0, 60, 10, 6, 'advanced', 'Add weight, increase depth', 'active');

-- Margaret Chen Exercise Prescription (Fall Risk)
INSERT INTO prescriptions (
  patient_id, assessment_id, clinician_id, prescription_date, start_date, end_date, status,
  program_name, program_goals, frequency_per_week, estimated_duration_minutes,
  clinician_notes, patient_instructions, precautions
) VALUES (
  4, (SELECT id FROM assessments WHERE patient_id = 4 ORDER BY id DESC LIMIT 1), 1, '2026-01-31 16:30:00', '2026-02-01', '2026-05-01', 'active',
  'Fall Prevention and Balance Training Program',
  '["Reduce fall risk", "Improve lower extremity strength", "Enhance gait quality and speed", "Reduce pain", "Improve balance confidence", "Maintain independence in ADLs"]',
  3, 60,
  'Priority: Fall prevention. 12-week supervised program with emphasis on safety. All exercises with stable support initially. Family education essential.',
 'Perform exercises 3x weekly with stable support (kitchen counter). Wear medical alert pendant. Stop if dizzy or unsteady. Family member nearby initially.',
  'High fall risk - safety paramount. All exercises with support initially. Medical alert pendant required. Family supervision recommended first 2-3 weeks.'
);

-- Margaret - Specific prescribed exercises
INSERT INTO prescribed_exercises (
  prescription_id, exercise_id, sets, reps, hold_time, rest_time, frequency_per_week, exercise_order, current_level, progression_criteria, status
) VALUES
((SELECT id FROM prescriptions WHERE patient_id = 4 ORDER BY id DESC LIMIT 1), 10, 3, 10, 0, 60, 9, 1, 'beginner', 'Progress to no hands, increase time', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 4 ORDER BY id DESC LIMIT 1), 16, 3, 8, 0, 90, 9, 2, 'beginner', 'Progress to no hands, increase reps', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 4 ORDER BY id DESC LIMIT 1), 8, 2, 8, 0, 60, 6, 3, 'beginner', 'Increase step height, add weight', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 4 ORDER BY id DESC LIMIT 1), 5, 3, 10, 0, 60, 9, 4, 'beginner', 'Add resistance band, increase reps', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 4 ORDER BY id DESC LIMIT 1), 7, 3, 30, 0, 60, 9, 5, 'beginner', 'Progress to weight-bearing stretch', 'active'),
((SELECT id FROM prescriptions WHERE patient_id = 4 ORDER BY id DESC LIMIT 1), 10, 2, 60, 0, 60, 9, 6, 'beginner', 'Progress to eyes closed, unstable surface', 'active');