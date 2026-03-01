-- Margaret Chen biomechanical analysis - Fall risk and elderly assessment
-- 72-year-old female with multiple comorbidities and high fall risk

-- Create assessment for Margaret Chen
INSERT INTO assessments (
  patient_id, clinician_id, assessment_date, assessment_type, status,
  tests_completed, total_score, camera_type,
  biomechanical_data, ai_analysis_results, deficiencies_detected,
  subjective_notes, objective_findings, assessment_notes, treatment_plan
) VALUES (
  4, 1, '2026-01-31 16:00:00', 'initial', 'completed',
  '["Sit-to-Stand Assessment", "Modified Squat", "Static Balance - Eyes Open", "Semi-Tandem Stance", "Gait Analysis with Cane", "Stair Climbing"]',
  32, 'web',
  '{"clinical_tests": {"berg_balance_scale": 38, "tug_test": 18.5, "chair_stand_30sec": 6, "gait_speed": 0.62}, "joint_angles": {"knee_extension_deficit": -9, "ankle_dorsiflexion": 6, "hip_flexion": 85}, "balance_metrics": {"static_balance": 11, "semi_tandem": 4, "single_leg": 0}, "pain_levels": {"rest": 3, "movement": 6, "stairs": 7, "squat": 8}, "strength_measures": {"quad_strength": "3/5", "hip_abductors": "3/5", "grip_strength": "normal"}}',
  '{"overall_quality_score": 32, "severity": "Poor", "clinical_significance": "Severe fall risk with multiple contributing factors requiring immediate intervention", "ai_confidence": 96, "fall_risk_category": "HIGH", "steadi_score": 12}',
  '["Severe Lower Extremity Weakness", "Impaired Balance (Multi-factorial)", "Gait Dysfunction", "Joint ROM Limitations", "Chronic Pain", "Sensory Deficits", "Fear of Falling", "Postural Abnormalities"]',
  '72-year-old female with multiple comorbidities including severe bilateral knee osteoarthritis, history of 2 falls in past month, sarcopenia, diabetes with neuropathy, and osteoporosis. Lives alone in senior community. Reports knee pain and fear of falling.',
  'High fall risk patient with multiple red flags: Berg 38/56 (high risk), TUG 18.5s (high risk), 30-sec chair stand 6 reps (poor), gait speed 0.62 m/s (severely impaired). Strength: quad 3/5, hip abductors 3/5. ROM: knee extension -9° deficit, ankle dorsiflexion 6°.',
  'Margaret presents with severe fall risk (12/12 STEADI risk factors), significant lower extremity weakness, multi-factorial balance impairment, and multiple comorbidities complicating rehabilitation. Movement quality score 32/100 indicates substantial functional limitations.',
  'Priority: Fall prevention. 12-week supervised PT program focusing on balance training, lower extremity strengthening, gait training, and safety education. Critical safety measures required due to high fall risk.'
);

-- Margaret - Sit-to-Stand Assessment
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 4 ORDER BY id DESC LIMIT 1),
  'Sit-to-Stand Assessment', 'mobility', 1, 'Rise from chair without using armrests, repeat 5 times',
  'completed', 1, 35.0, 'Severe difficulty with transfers. Required armrests for all reps, excessive trunk lean 55°, multiple attempts, visible tremor.',
  '["Severe lower extremity weakness", "Poor transfer mechanics", "Functional limitation", "Safety concern"]','["Armrests required", "Excessive trunk lean", "Multiple attempts", "Tremor observed"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.81, "reps_completed": 5, "trunk_lean": 55, "tremor_score": 7}',
  '{"trunk_lean": 55, "tremor_score": 7, "completion_time": 28}',
  '{"normal_trunk_lean": 15, "normal_time": 12, "armrests": "none"}',
  '2026-01-31 16:05:00'
);

-- Margaret - Modified Squat
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 4 ORDER BY id DESC LIMIT 1),
  'Modified Squat', 'mobility', 2, 'Perform partial squat with counter support',
  'completed', 1, 28.0, 'Very limited due to pain and weakness. Maximum depth 30° knee flexion only. Pain 8/10 at end range.',
  '["Severe ROM limitation", "High pain levels", "Muscle weakness", "Joint stiffness"]','["Shallow squat depth", "Counter support required", "Pain avoidance", "Limited movement"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.79, "pain_level": 8, "squat_depth": 30, "support_required": true}',
  '{"squat_depth": 30, "pain_level": 8, "support_level": "counter"}',
  '{"normal_squat_depth": 90, "pain_threshold": 3, "target_depth": 60}',
  '2026-01-31 16:08:00'
);

-- Margaret - Static Balance - Eyes Open
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 4 ORDER BY id DESC LIMIT 1),
  'Static Balance - Eyes Open', 'balance', 3, 'Stand with feet together and maintain balance',
  'completed', 1, 42.0, 'Poor static balance with multiple corrections required. Maintained only 11 seconds vs target 30 seconds.',
  '["Poor static balance", "Visual dependence", "Proprioceptive loss", "Postural instability"]','["Multiple balance corrections", "Standby assistance required", "Wide base of support"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.83, "balance_time": 11, "corrections": 8, "stability_score": 42}',
  '{"balance_time": 11, "corrections": 8, "sway_amplitude": 12.5}',
  '{"normal_balance_time": 30, "normal_corrections": 2, "target_time": 25}',
  '2026-01-31 16:12:00'
);

-- Margaret - Semi-Tandem Stance
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 4 ORDER BY id DESC LIMIT 1),
  'Semi-Tandem Stance', 'balance', 4, 'Stand with one foot partially in front of the other',
  'completed', 1, 25.0, 'Unable to maintain semi-tandem position. Immediate loss of balance after 4 seconds, required therapist assistance.',
  '["Severe balance impairment", "Inability to maintain narrow base", "High fall risk"]','["Immediate loss of balance", "Therapist assistance required", "Unable to achieve position"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.80, "balance_time": 4, "assistance_required": true, "fall_risk_score": 25}',
  '{"balance_time": 4, "stance_width": "semi-tandem", "assistance_level": "therapist"}',
  '{"normal_balance_time": 30, "target_time": 15, "acceptable_assistance": "none"}',
  '2026-01-31 16:15:00'
);

-- Margaret - Gait Analysis with Cane
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 4 ORDER BY id DESC LIMIT 1),
  'Gait Analysis with Cane', 'coordination', 5, 'Walk 10 meters using cane at normal pace',
  'completed', 1, 38.0, 'Severely impaired gait pattern. Speed 0.625 m/s (severely impaired), step length 35-38cm (Normal: 55-65cm), widened step width 18cm, shuffling steps.',
  '["Severely impaired gait", "Reduced step length", "Widened base of support", "Shuffling pattern"]','["Shortened steps", "Reduced foot clearance", "Forward head posture", "Slow cautious gait"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.82, "gait_speed": 0.625, "step_length": {"left": 38, "right": 35}, "step_width": 18, "double_support": 45}',
  '{"gait_speed": 0.625, "step_length_left": 38, "step_length_right": 35, "step_width": 18, "double_support_time": 45}',
  '{"normal_gait_speed": 1.2, "normal_step_length": 60, "normal_step_width": 10, "normal_double_support": 25}',
  '2026-01-31 16:18:00'
);

-- Margaret - Stair Climbing
INSERT INTO movement_tests (
  assessment_id, test_name, test_category, test_order, instructions,
  status, score, movement_quality_score, ai_feedback, deficiencies, compensations_detected,
  skeleton_data, joint_angles, rom_measurements, completed_at
) VALUES (
  (SELECT id FROM assessments WHERE patient_id = 4 ORDER BY id DESC LIMIT 1),
  'Stair Climbing', 'mobility', 6, 'Climb 4 steps using handrail',
  'completed', 1, 32.0, 'Very slow and fearful stair negotiation. Ascent: 22 seconds, descent: 28 seconds. Unable to perform step-over-step pattern.',
  '["Severe functional limitation", "Fear of falling", "Poor stair negotiation", "Safety concerns"]','["Very slow ascent/descent", "Step-to pattern only", "Handrail + standby required", "Fearful movement"]',
  '{"joints_tracked": 33, "tracking_confidence": 0.80, "ascent_time": 22, "descent_time": 28, "steps_completed": 4, "safety_required": true}',
  '{"ascent_time": 22, "descent_time": 28, "steps": 4, "pattern": "step-to"}',
  '{"normal_ascent": 8, "normal_descent": 10, "normal_pattern": "step-over-step", "safety": "independent"}',
  '2026-01-31 16:22:00'
);