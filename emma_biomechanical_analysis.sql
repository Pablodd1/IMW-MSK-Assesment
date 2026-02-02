-- Comprehensive biomechanical analysis results for mock patients
-- Emma Rodriguez (8-year-old with Developmental Coordination Disorder)

-- Create assessment for Emma Rodriguez
INSERT INTO assessments (
  patient_id, clinician_id, assessment_date, assessment_type, status,
  tests_completed, total_score, camera_type,
  biomechanical_data, ai_analysis_results, deficiencies_detected,
  subjective_notes, objective_findings, assessment_notes, treatment_plan
) VALUES (
  2, 1, '2026-01-31 10:00:00', 'initial', 'completed',
  '["Squat Test", "Single Leg Balance - Right", "Single Leg Balance - Left", "Forward Reach Test", "Heel-to-Toe Walk"]',
  38, 'web',
  '{"joint_angles": {"hip_flexion": 75, "knee_flexion": 85, "ankle_dorsiflexion": 10}, "rom_measurements": {"hip_flexion_normal": 110, "knee_flexion_normal": 120, "ankle_dorsiflexion_normal": 20}, "balance_metrics": {"single_leg_right": 3.2, "single_leg_left": 2.8, "forward_reach": 15}, "movement_patterns": {"squat_depth": "limited", "knee_alignment": "valgus", "trunk_lean": "excessive"}}',
  '{"overall_quality_score": 38, "severity": "Poor", "clinical_significance": "Significant developmental coordination deficits requiring intervention", "ai_confidence": 92}',
  '["Severe Balance Deficit", "Limited Lower Extremity Flexibility", "Muscle Weakness - Lower Extremities", "Poor Postural Control", "Reduced Proprioception"]',
  '8-year-old female with developmental coordination disorder. Parents report frequent falls during play, difficulty with ball skills, and inability to ride bicycle without training wheels. No pain complaints.',
  'Significant coordination deficits observed. Balance: single leg stance 3.2s right, 2.8s left (normal >10s). Squat: limited depth with 75° hip flexion, 85° knee flexion, 10° ankle dorsiflexion. Forward reach 15cm (normal 20-25cm). Gait: frequent step-offs during heel-to-toe walk.',
  'Emma demonstrates significant developmental coordination disorder with marked balance deficits, limited lower extremity flexibility, and poor postural control. Movement quality score 38/100 indicates substantial functional limitations requiring comprehensive therapeutic intervention.',
  '8-week comprehensive therapy program focusing on balance training, lower extremity strengthening, ankle mobility, and proprioception development. Parent education and home exercise program essential.'
);

-- Create detailed movement tests for Emma
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 2 ORDER BY id DESC LIMIT 1),
  'Squat Test', 'mobility', 1, 'Stand with feet shoulder-width apart and perform a squat to comfortable depth',
  'completed', 1, 45.0, 'Poor squat mechanics with limited depth, excessive forward lean, and knee valgus. Hip flexion only 75° vs normal 110-120°.',
  '["Limited ankle dorsiflexion (10° vs normal 20-25°)", "Poor hip control with valgus collapse", "Excessive trunk lean 45°"]', '["Knee valgus during descent", "Heel elevation", "Forward trunk lean"]', '["Arms extended for balance", "Wide base of support"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.85, "key_angles": {"hip_flexion": 75, "knee_flexion": 85, "ankle_dorsiflexion": 10, "trunk_lean": 45}}',
  '{"hip_flexion": 75, "knee_flexion": 85, "ankle_dorsiflexion": 10, "trunk_lean": 45}',
  '{"hip_flexion_normal": 110, "knee_flexion_normal": 120, "ankle_dorsiflexion_normal": 20}',
  '2026-01-31 10:05:00'
);

INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 2 ORDER BY id DESC LIMIT 1),
  'Single Leg Balance - Right', 'balance', 2, 'Stand on right leg and maintain balance as long as possible',
  'completed', 1, 30.0, 'Poor balance with multiple corrections required. Balance time 3.2 seconds vs normal >10 seconds.',
  '["Severe balance deficit", "Requires visual dependence", "Poor proprioception"]', '["Arms extended", "Hip hiking", "Frequent foot touches"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.82, "balance_time": 3.2, "stability_score": 30}',
  '{"balance_time": 3.2, "sway_amplitude": 15.5}',
  '{"normal_balance_time": 10, "age_expected": 8}',
  '2026-01-31 10:08:00'
);

INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 2 ORDER BY id DESC LIMIT 1),
  'Single Leg Balance - Left', 'balance', 3, 'Stand on left leg and maintain balance as long as possible',
  'completed', 1, 28.0, 'Worse balance than right side. Balance time 2.8 seconds with immediate loss of balance.',
  '["Severe balance deficit", "Dominant side weakness", "Poor motor control"]','["Immediate foot touches", "Excessive sway", "Visual dependence"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.80, "balance_time": 2.8, "stability_score": 28}',
  '{"balance_time": 2.8, "sway_amplitude": 18.2}',
  '{"normal_balance_time": 10, "age_expected": 8}',
  '2026-01-31 10:10:00'
);

INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 2 ORDER BY id DESC LIMIT 1),
  'Forward Reach Test', 'balance', 4, 'Reach forward as far as possible without losing balance',
  'completed', 1, 52.0, 'Limited reach distance with fear of falling evident. Reach 15cm vs normal 20-25cm.',
  '["Limited reach distance", "Fear of falling", "Poor anticipatory control"]','["Hesitation", "Partial reach", "Safety seeking"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.83, "reach_distance": 15, "balance_confidence": 52}',
  '{"reach_distance": 15, "trunk_flexion": 25}',
  '{"normal_reach": 22, "age_expected": 20}',
  '2026-01-31 10:12:00'
);

INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 2 ORDER BY id DESC LIMIT 1),
  'Heel-to-Toe Walk', 'coordination', 5, 'Walk heel-to-toe along a straight line for 3 meters',
  'completed', 1, 35.0, 'Poor coordination with 7 step-offs, arms extended, and eyes looking down. Completed in 10 seconds.',
  '["Poor coordination", "Visual dependence", "Balance instability"]','["7 step-offs", "Arms extended", "Eyes down", "Slow cautious gait"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.79, "step_offs": 7, "completion_time": 10}',
  '{"step_offs": 7, "completion_time": 10, "gait_stability": 35}',
  '{"normal_step_offs": 0, "normal_time": 6}',
  '2026-01-31 10:15:00'
);