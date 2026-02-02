-- Marcus Thompson biomechanical analysis - ACL reconstruction case
-- 28-year-old male, 6 weeks post-op ACL reconstruction

-- Create assessment for Marcus Thompson
INSERT INTO assessments (
  patient_id, clinician_id, assessment_date, assessment_type, status,
  tests_completed, total_score, camera_type,
  biomechanical_data, ai_analysis_results, deficiencies_detected,
  subjective_notes, objective_findings, assessment_notes, treatment_plan
) VALUES (
  3, 1, '2026-01-31 14:00:00', 'initial', 'completed',
  '["Bilateral Squat Assessment", "Single Leg Squat - Right", "Single Leg Squat - Left", "Single Leg Balance - Right", "Single Leg Balance - Left", "Gait Analysis"]',
  54, 'femto_mega',
  '{"joint_angles": {"left_knee_flexion": 95, "right_knee_flexion": 125, "hip_flexion_left": 85, "hip_flexion_right": 110, "knee_anterior_translation": 15}, "rom_measurements": {"left_knee_flexion_normal": 135, "right_knee_flexion_normal": 135, "extension_lag": 0}, "balance_metrics": {"single_leg_right": 30, "single_leg_left": 12, "eyes_closed_right": 18, "eyes_closed_left": 3}, "strength_metrics": {"quad_atrophy": 3, "mmt_left": "3+/5", "mmt_right": "5/5"}, "gait_analysis": {"step_length_left": 58, "step_length_right": 72, "stance_time_left": 0.52, "stance_time_right": 0.78, "asymmetry": 24}}',
  '{"overall_quality_score": 54, "severity": "Fair", "clinical_significance": "Expected 6-week post-op findings with significant quad weakness and proprioceptive deficits", "ai_confidence": 94}',
  '["Severe Quadriceps Weakness (Post-surgical)", "Reduced Proprioception (Left Knee)", "Movement Asymmetry", "Dynamic Knee Stability", "Range of Motion Limitation"]',
  '28-year-old male status post ACL reconstruction (hamstring autograft) 6 weeks ago. Reports knee instability and difficulty with exercises. Semi-professional soccer player eager to return to sport.',
  'Significant quad atrophy (3cm) and strength deficit (MMT 3+/5 vs 5/5). Balance: single leg 30s right, 12s left (60% deficit). Gait: 24% step length asymmetry, antalgic pattern. Knee flexion: 95° left vs 125° right.',
  'Marcus demonstrates expected 6-week post-op findings with severe quadriceps weakness, reduced proprioception, and movement asymmetry. Movement quality score 54/100 is appropriate for this stage of recovery.',
  'Phase 1 rehabilitation (weeks 6-12): Focus on quadriceps strengthening, proprioception training, and gait normalization. Progress to return-to-sport criteria by 24-48 weeks.'
);

-- Marcus - Bilateral Squat Assessment
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 3 ORDER BY id DESC LIMIT 1),
  'Bilateral Squat Assessment', 'mobility', 1, 'Perform a bilateral squat to comfortable depth',
  'completed', 2, 58.0, 'Asymmetrical squat with 70/30 weight shift to right. Left knee limited to 95° flexion vs 125° right. Excessive anterior tibial translation +15mm.',
  '["Asymmetrical weight distribution", "Limited left knee flexion", "Excessive tibial translation", "Quad weakness"]','["Weight shift to right", "Shallow left knee flexion", "Compensatory hip strategy"]',
  '{"joints_tracked": 32, "tracking_confidence": 0.91, "weight_distribution": {"left": 30, "right": 70}, "knee_flexion": {"left": 95, "right": 125}, "anterior_translation": 15}',
  '{"knee_flexion_left": 95, "knee_flexion_right": 125, "hip_flexion_left": 85, "hip_flexion_right": 110}',
  '{"knee_flexion_normal": 135, "hip_flexion_normal": 110, "extension_lag": 0}',
  '2026-01-31 14:05:00'
);

-- Marcus - Single Leg Squat - Right (Non-surgical)
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 3 ORDER BY id DESC LIMIT 1),
  'Single Leg Squat - Right', 'strength', 2, 'Perform single leg squat on right (non-surgical) side',
  'completed', 3, 82.0, 'Good control on non-surgical side. 5 reps with good depth, minimal valgus, controlled descent.',
  '["Good strength and control", "Minimal valgus", "Appropriate depth"]','["Slight knee valgus", "Minor hip drop"]',
  '{"joints_tracked": 32, "tracking_confidence": 0.89, "reps_completed": 5, "valgus_angle": 3, "depth_score": 85}',
  '{"knee_flexion": 110, "hip_flexion": 95, "valgus_angle": 3}',
  '{"knee_flexion_target": 120, "valgus_normal": 0}',
  '2026-01-31 14:08:00'
);

-- Marcus - Single Leg Squat - Left (Surgical)
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 3 ORDER BY id DESC LIMIT 1),
  'Single Leg Squat - Left', 'strength', 3, 'Perform modified single leg squat on left (surgical) side',
  'completed', 1, 42.0, 'Significant weakness on surgical side. Required wall support, shallow squat (30° only), 8° knee valgus, 4/10 pain.',
  '["Severe quad weakness", "Limited knee flexion", "Excessive valgus", "Pain with movement"]','["Wall support required", "Shallow squat depth", "Knee valgus 8°", "Compensatory hip strategy"]',
  '{"joints_tracked": 32, "tracking_confidence": 0.87, "pain_level": 4, "valgus_angle": 8, "squat_depth": 30, "assistance_required": true}',
  '{"knee_flexion": 30, "hip_flexion": 45, "valgus_angle": 8, "pain_level": 4}',
  '{"knee_flexion_target": 90, "valgus_normal": 0, "pain_threshold": 3}',
  '2026-01-31 14:12:00'
);

-- Marcus - Single Leg Balance - Right
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 3 ORDER BY id DESC LIMIT 1),
  'Single Leg Balance - Right', 'balance', 4, 'Stand on right leg and maintain balance',
  'completed', 3, 88.0, 'Excellent balance on non-surgical side. Eyes open: 30 seconds, eyes closed: 18 seconds.',
  '["Excellent balance", "Good proprioception", "Stable control"]','["Minimal sway", "Good hip strategy"]',
  '{"joints_tracked": 32, "tracking_confidence": 0.90, "balance_time_eyes_open": 30, "balance_time_eyes_closed": 18, "stability_score": 88}',
  '{"balance_time_eyes_open": 30, "balance_time_eyes_closed": 18, "sway_amplitude": 5.2}',
  '{"normal_eyes_open": 30, "normal_eyes_closed": 15}',
  '2026-01-31 14:15:00'
);

-- Marcus - Single Leg Balance - Left (Surgical)
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 3 ORDER BY id DESC LIMIT 1),
  'Single Leg Balance - Left', 'balance', 5, 'Stand on left leg and maintain balance',
  'completed', 1, 45.0, 'Significant balance deficit on surgical side. Eyes open: 12 seconds (60% deficit), eyes closed: 3 seconds (83% deficit).',
  '["Significant balance deficit", "Proprioceptive loss", "Poor neuromuscular control"]','["Excessive sway", "Multiple foot touches", "Visual dependence"]',
  '{"joints_tracked": 32, "tracking_confidence": 0.85, "balance_time_eyes_open": 12, "balance_time_eyes_closed": 3, "stability_score": 45}',
  '{"balance_time_eyes_open": 12, "balance_time_eyes_closed": 3, "sway_amplitude": 22.1}',
  '{"normal_eyes_open": 30, "normal_eyes_closed": 15, "expected_6wk": 20}',
  '2026-01-31 14:18:00'
);

-- Marcus - Gait Analysis
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 3 ORDER BY id DESC LIMIT 1),
  'Gait Analysis', 'coordination', 6, 'Walk naturally for 10 meters at normal pace',
  'completed', 2, 62.0, 'Antalgic gait pattern with significant asymmetry. Step length: 58cm left vs 72cm right (24% difference). Stance time: 0.52s left vs 0.78s right.',
  '["Antalgic gait pattern", "Significant step length asymmetry", "Reduced knee flexion", "Compensatory hip strategy"]','["Shortened left step", "Reduced push-off", "Hip hiking", "Protective gait"]',
  '{"joints_tracked": 32, "tracking_confidence": 0.88, "step_length": {"left": 58, "right": 72}, "stance_time": {"left": 0.52, "right": 0.78}, "asymmetry": 24}',
  '{"step_length_left": 58, "step_length_right": 72, "stance_time_left": 0.52, "stance_time_right": 0.78}',
  '{"normal_step_length": 70, "normal_stance_time": 0.6, "acceptable_asymmetry": 10}',
  '2026-01-31 14:22:00'
);