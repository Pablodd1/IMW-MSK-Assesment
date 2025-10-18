-- Seed Data for Medical Movement Assessment Platform

-- ============================================================================
-- EXERCISE LIBRARY
-- ============================================================================

INSERT OR IGNORE INTO exercise_library (exercise_name, exercise_category, target_muscles, target_joints, target_movements, difficulty_level, description, instructions, contraindications, equipment_required, estimated_duration_seconds) VALUES

-- Mobility Exercises
('Deep Squat', 'mobility', '["quadriceps","glutes","hamstrings","calves"]', '["hip","knee","ankle"]', '["hip_flexion","knee_flexion","ankle_dorsiflexion"]', 'intermediate', 
'Full-depth squat assessment and exercise for lower body mobility and stability.', 
'Stand with feet shoulder-width apart. Lower your body by bending knees and hips, keeping chest up. Descend until thighs are parallel or below. Return to standing.', 
'Knee injury, hip replacement (recent), severe arthritis', 
'[]', 60),

('Overhead Squat', 'mobility', '["quadriceps","glutes","shoulders","core"]', '["hip","knee","ankle","shoulder"]', '["hip_flexion","knee_flexion","shoulder_flexion"]', 'advanced',
'Full-body mobility assessment evaluating ankle, hip, and shoulder mobility with core stability.',
'Stand with arms extended overhead, shoulder-width apart. Perform a deep squat while maintaining arms overhead. Keep chest up and heels down.',
'Shoulder impingement, rotator cuff injury, knee instability',
'["dowel_rod"]', 60),

('Hip Flexor Stretch', 'flexibility', '["hip_flexors","psoas","iliacus"]', '["hip"]', '["hip_extension"]', 'beginner',
'Stretching exercise to improve hip flexor mobility and reduce anterior pelvic tilt.',
'Kneel on one knee with other foot forward. Push hips forward while keeping back straight until stretch is felt in front of hip. Hold position.',
'Hip replacement (recent), severe hip arthritis',
'["yoga_mat"]', 90),

-- Stability Exercises
('Single Leg Balance', 'balance', '["glutes","core","ankle_stabilizers"]', '["hip","knee","ankle"]', '["hip_stabilization","ankle_stabilization"]', 'beginner',
'Balance assessment to evaluate proprioception and lower extremity stability.',
'Stand on one leg with hands on hips. Maintain balance without touching down with other foot. Focus on a fixed point.',
'Recent ankle sprain, severe balance disorders, vertigo',
'[]', 60),

('Plank Hold', 'stability', '["core","shoulders","glutes"]', '["shoulder","spine","hip"]', '["core_stabilization","shoulder_stabilization"]', 'intermediate',
'Core stability exercise to assess and improve trunk control and endurance.',
'Position on forearms and toes, body in straight line from head to heels. Hold position without sagging or hiking hips.',
'Shoulder injury, lower back pain (acute), wrist injury',
'["yoga_mat"]', 60),

-- Strength Exercises
('Romanian Deadlift', 'strength', '["hamstrings","glutes","erector_spinae"]', '["hip","knee"]', '["hip_extension","hip_hinge"]', 'intermediate',
'Hip hinge pattern exercise to strengthen posterior chain and assess movement quality.',
'Stand holding weight at thighs. Push hips back, lowering weight while keeping back straight. Lower until hamstrings stretch, then return by driving hips forward.',
'Lower back injury (acute), hamstring tear, herniated disc',
'["dumbbells"]', 90),

('Single Leg RDL', 'strength', '["hamstrings","glutes","core"]', '["hip","ankle"]', '["hip_extension","hip_hinge","balance"]', 'advanced',
'Unilateral hip hinge with balance component to assess asymmetries and stability.',
'Stand on one leg. Hinge at hip while extending other leg behind. Lower torso while maintaining straight back. Return to standing.',
'Hamstring injury, severe balance impairment, ankle instability',
'["dumbbells"]', 90),

-- Functional Exercises
('Sit to Stand', 'functional', '["quadriceps","glutes","core"]', '["hip","knee","ankle"]', '["hip_extension","knee_extension"]', 'beginner',
'Functional movement pattern assessing lower body strength and coordination.',
'Sit in chair with feet flat. Stand up without using hands by driving through heels. Control descent when sitting back down.',
'Recent hip/knee replacement, severe osteoarthritis',
'["chair"]', 45),

('Lunge Pattern', 'functional', '["quadriceps","glutes","hip_flexors"]', '["hip","knee","ankle"]', '["hip_flexion","knee_flexion","hip_stabilization"]', 'intermediate',
'Dynamic stability and strength assessment in split stance position.',
'Step forward into lunge, lowering back knee toward ground. Front knee should stay over ankle. Push back to starting position.',
'Knee injury, hip replacement (recent), balance impairment',
'[]', 60),

('Step Up', 'functional', '["quadriceps","glutes","hamstrings"]', '["hip","knee","ankle"]', '["hip_extension","knee_extension"]', 'intermediate',
'Functional lower body exercise assessing unilateral strength and control.',
'Place one foot on elevated surface. Step up by driving through heel, bringing other foot to platform. Step down with control.',
'Knee instability, hip pain, balance disorders',
'["step_platform"]', 60),

-- Shoulder/Upper Body
('Shoulder Flexion', 'mobility', '["deltoids","rotator_cuff"]', '["shoulder"]', '["shoulder_flexion"]', 'beginner',
'Shoulder mobility assessment and exercise for overhead reaching.',
'Stand or sit with arms at sides. Raise arms forward and overhead as high as possible. Lower with control.',
'Rotator cuff tear, shoulder impingement (acute), frozen shoulder',
'[]', 45),

('Shoulder External Rotation', 'strength', '["rotator_cuff","posterior_deltoid"]', '["shoulder"]', '["shoulder_external_rotation"]', 'beginner',
'Rotator cuff strengthening for shoulder stability and injury prevention.',
'Lie on side with elbow bent 90 degrees. Rotate forearm upward while keeping elbow against body. Lower with control.',
'Rotator cuff tear (acute), shoulder dislocation (recent)',
'["resistance_band"]', 60),

-- Core Exercises
('Dead Bug', 'stability', '["core","hip_flexors"]', '["spine","hip"]', '["core_stabilization","hip_flexion"]', 'beginner',
'Core stability exercise with coordination component.',
'Lie on back with arms extended toward ceiling, knees bent 90 degrees. Lower opposite arm and leg while maintaining lower back contact with floor. Alternate sides.',
'Lower back pain (acute), hip flexor strain',
'["yoga_mat"]', 90),

('Bird Dog', 'stability', '["core","glutes","shoulders"]', '["spine","hip","shoulder"]', '["core_stabilization","hip_extension","shoulder_flexion"]', 'intermediate',
'Core and hip stability exercise with balance component.',
'Start on hands and knees. Extend opposite arm and leg while maintaining level hips and shoulders. Hold, then return. Alternate sides.',
'Shoulder injury, lower back pain (acute), knee injury',
'["yoga_mat"]', 75),

-- Flexibility/Mobility
('Thoracic Rotation', 'mobility', '["thoracic_spine","obliques"]', '["spine"]', '["thoracic_rotation"]', 'beginner',
'Thoracic spine mobility exercise to improve rotation and reduce compensatory movements.',
'Kneel with one hand behind head. Rotate upper body toward ceiling, following elbow with eyes. Return to start. Repeat on both sides.',
'Recent spine surgery, severe osteoporosis, acute back pain',
'["yoga_mat"]', 60),

('Cat-Cow Stretch', 'mobility', '["erector_spinae","core"]', '["spine"]', '["spinal_flexion","spinal_extension"]', 'beginner',
'Spinal mobility exercise to improve flexion and extension range of motion.',
'Start on hands and knees. Arch back while lifting head (cow). Round back while tucking chin (cat). Flow smoothly between positions.',
'Acute back injury, herniated disc, severe osteoporosis',
'["yoga_mat"]', 90);

-- ============================================================================
-- CPT CODES FOR MEDICAL BILLING
-- ============================================================================

INSERT OR IGNORE INTO billing_codes (cpt_code, code_description, code_category, minimum_duration_minutes, requires_documentation, is_rpm_code, rpm_time_requirement_minutes) VALUES

-- Physical Therapy Evaluation
('97161', 'Physical therapy evaluation - Low complexity', 'evaluation', 20, 1, 0, NULL),
('97162', 'Physical therapy evaluation - Moderate complexity', 'evaluation', 30, 1, 0, NULL),
('97163', 'Physical therapy evaluation - High complexity', 'evaluation', 45, 1, 0, NULL),

-- Re-evaluation
('97164', 'Physical therapy re-evaluation', 'evaluation', 20, 1, 0, NULL),

-- Therapeutic Exercises
('97110', 'Therapeutic exercises - Development of strength, endurance, ROM, flexibility', 'treatment', 15, 1, 0, NULL),

-- Neuromuscular Re-education
('97112', 'Neuromuscular re-education of movement, balance, coordination, kinesthetic sense', 'treatment', 15, 1, 0, NULL),

-- Gait Training
('97116', 'Gait training therapy', 'treatment', 15, 1, 0, NULL),

-- Manual Therapy
('97140', 'Manual therapy techniques - Mobilization, manipulation, manual traction', 'treatment', 15, 1, 0, NULL),

-- Therapeutic Activities
('97530', 'Therapeutic activities - Direct patient contact for dynamic activities', 'treatment', 15, 1, 0, NULL),

-- Self-care/Home Management Training
('97535', 'Self-care/home management training', 'treatment', 15, 1, 0, NULL),

-- Remote Physiologic Monitoring (RPM)
('99453', 'RPM - Initial setup and patient education', 'rpm', 16, 1, 1, 16),
('99454', 'RPM - Device supply with daily recording and transmission', 'rpm', 0, 1, 1, 16),
('99457', 'RPM - Treatment management services, first 20 minutes', 'rpm', 20, 1, 1, 20),
('99458', 'RPM - Additional 20 minutes of monitoring', 'rpm', 20, 1, 1, 20),

-- Remote Therapeutic Monitoring (RTM) - Added 2022
('98975', 'RTM - Initial setup and patient education', 'rpm', 16, 1, 1, 16),
('98976', 'RTM - Device supply for muscle, bone, or joint condition', 'rpm', 0, 1, 1, 16),
('98977', 'RTM - Treatment management, first 20 minutes', 'rpm', 20, 1, 1, 20),
('98978', 'RTM - Additional 20 minutes of monitoring', 'rpm', 20, 1, 1, 20),

-- Functional Capacity Evaluation
('97750', 'Physical performance test or measurement', 'evaluation', 15, 1, 0, NULL);

-- ============================================================================
-- SAMPLE CLINICIAN
-- ============================================================================

INSERT OR IGNORE INTO clinicians (first_name, last_name, email, credential, license_number, npi_number, specialty) VALUES
('Dr. Sarah', 'Johnson', 'sjohnson@clinic.com', 'DPT', 'PT12345', '1234567890', 'physical_therapy'),
('Dr. Michael', 'Chen', 'mchen@clinic.com', 'DC', 'DC54321', '0987654321', 'chiropractic');
