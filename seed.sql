-- Seed data for PhysioAI Medical Assessment System

-- Insert comprehensive exercise library
INSERT INTO exercises (name, category, body_region, description, instructions, difficulty, default_sets, default_reps, default_hold_time, default_frequency, indications, contraindications, targeted_deficiencies) VALUES

-- Shoulder exercises
('Shoulder External Rotation', 'mobility', '["shoulder"]', 'External rotation exercise to improve rotator cuff strength and shoulder stability', '1. Lie on your side with affected arm on top\n2. Bend elbow 90 degrees\n3. Keeping elbow at side, rotate arm up\n4. Slowly lower back down\n5. Complete prescribed reps', 'beginner', 3, 15, NULL, 5, '["rotator cuff weakness", "shoulder instability", "impingement syndrome"]', '["acute shoulder dislocation", "recent surgery", "severe pain"]', '["limited shoulder ROM", "rotator cuff weakness", "poor scapular control"]'),

('Scapular Wall Slides', 'mobility', '["shoulder", "spine"]', 'Improves scapular mobility and thoracic extension', '1. Stand with back against wall\n2. Arms at 90-90 position\n3. Slide arms overhead keeping contact with wall\n4. Return to start position\n5. Focus on squeezing shoulder blades', 'beginner', 3, 12, NULL, 5, '["poor posture", "shoulder pain", "scapular dyskinesis"]', '["shoulder impingement during movement", "acute injury"]', '["scapular dyskinesis", "thoracic hyperkyphosis", "limited overhead reach"]'),

('YTW Exercise', 'strength', '["shoulder", "spine"]', 'Strengthens posterior shoulder and scapular stabilizers', '1. Lie face down or bend forward at hips\n2. Y: Arms overhead at 45-degree angle\n3. T: Arms straight out to sides\n4. W: Elbows bent, squeeze shoulder blades\n5. Hold each position briefly', 'intermediate', 3, 10, 3, 4, '["rounded shoulders", "weak upper back", "postural dysfunction"]', '["acute shoulder injury", "severe pain"]', '["weak scapular stabilizers", "forward head posture", "rounded shoulders"]'),

-- Hip exercises  
('Hip Flexor Stretch', 'flexibility', '["hip"]', 'Stretches hip flexors to improve hip extension', '1. Kneel on one knee, other foot forward\n2. Keep torso upright\n3. Gently push hips forward\n4. Feel stretch in front of hip\n5. Hold for prescribed time', 'beginner', 3, NULL, 30, 5, '["tight hip flexors", "anterior pelvic tilt", "low back pain"]', '["hip replacement", "acute groin strain", "hip fracture"]', '["limited hip extension", "anterior pelvic tilt", "lumbar hyperlordosis"]'),

('Clamshells', 'strength', '["hip"]', 'Strengthens hip abductors and external rotators', '1. Lie on side with knees bent\n2. Keep feet together\n3. Lift top knee while keeping hips stable\n4. Slowly lower back down\n5. Complete prescribed reps', 'beginner', 3, 15, NULL, 5, '["hip weakness", "IT band syndrome", "knee pain"]', '["acute hip pain", "recent hip surgery"]', '["weak hip abductors", "gluteus medius weakness", "trendelenburg gait"]'),

('Single Leg Bridge', 'strength', '["hip", "spine"]', 'Strengthens glutes and hamstrings while improving stability', '1. Lie on back, knees bent\n2. Lift one leg straight\n3. Push through heel of standing leg\n4. Lift hips off ground\n5. Hold and lower slowly', 'intermediate', 3, 12, 2, 4, '["weak glutes", "low back pain", "hamstring weakness"]', '["acute low back pain", "hip flexor pain"]', '["weak glute activation", "poor hip extension", "hamstring weakness"]'),

-- Knee exercises
('Terminal Knee Extension', 'strength', '["knee"]', 'Strengthens quadriceps, especially VMO', '1. Loop resistance band behind knee\n2. Slightly bend knee\n3. Straighten knee fully against resistance\n4. Hold briefly\n5. Slowly return to start', 'beginner', 3, 15, 2, 5, '["patellofemoral pain", "post-ACL surgery", "quad weakness"]', '["acute knee injury", "knee effusion"]', '["quadriceps weakness", "VMO atrophy", "limited knee extension"]'),

('Step Downs', 'strength', '["knee", "hip", "ankle"]', 'Functional exercise for knee control and eccentric strength', '1. Stand on step with one leg\n2. Slowly lower opposite heel toward ground\n3. Keep knee aligned over toes\n4. Touch floor lightly\n5. Push back up', 'intermediate', 3, 10, NULL, 4, '["patellofemoral pain", "ACL prevention", "general knee weakness"]', '["acute knee injury", "severe pain"]', '["poor knee control", "weak eccentric strength", "dynamic valgus"]'),

-- Ankle exercises
('Ankle Alphabet', 'mobility', '["ankle"]', 'Improves ankle mobility in all planes', '1. Sit with leg extended\n2. Use ankle to write alphabet in air\n3. Make large letters\n4. Complete A-Z slowly\n5. Switch ankles', 'beginner', 2, 1, NULL, 5, '["ankle stiffness", "post ankle sprain", "limited mobility"]', '["acute ankle injury", "severe swelling"]', '["limited ankle dorsiflexion", "reduced ankle mobility", "post-immobilization stiffness"]'),

('Single Leg Balance', 'balance', '["ankle", "knee", "hip"]', 'Improves proprioception and balance', '1. Stand on one leg\n2. Keep knee slightly bent\n3. Hold steady position\n4. Progress by closing eyes\n5. Advanced: stand on unstable surface', 'beginner', 3, NULL, 30, 5, '["ankle instability", "balance deficits", "fall prevention"]', '["acute injury", "severe instability"]', '["poor proprioception", "ankle instability", "balance deficits"]'),

-- Core exercises
('Dead Bug', 'stability', '["spine"]', 'Core stability exercise emphasizing anti-extension', '1. Lie on back, knees at 90 degrees\n2. Extend opposite arm and leg\n3. Keep low back flat on floor\n4. Alternate sides slowly\n5. Breathe throughout', 'intermediate', 3, 10, NULL, 4, '["low back pain", "core weakness", "postural instability"]', '["acute low back pain", "herniated disc symptoms"]', '["weak core stability", "poor motor control", "anterior pelvic tilt"]'),

('Bird Dog', 'stability', '["spine", "hip", "shoulder"]', 'Multi-planar stability exercise', '1. Start on hands and knees\n2. Extend opposite arm and leg\n3. Keep spine neutral\n4. Hold steady position\n5. Alternate sides', 'intermediate', 3, 10, 5, 4, '["low back pain", "core weakness", "balance deficits"]', '["acute back pain", "shoulder pain during weight bearing"]', '["poor spinal stability", "weak core", "balance deficits"]'),

('Plank', 'stability', '["spine"]', 'Isometric core strengthening', '1. Forearms on ground, elbows under shoulders\n2. Extend legs, balance on toes\n3. Keep body in straight line\n4. Engage core, don''t let hips sag\n5. Hold for prescribed time', 'intermediate', 3, NULL, 30, 4, '["core weakness", "athletic training", "postural support"]', '["shoulder injury", "acute low back pain", "wrist pain"]', '["weak core endurance", "poor postural control"]'),

-- Spine mobility
('Cat-Cow Stretch', 'mobility', '["spine"]', 'Improves spinal mobility and segmental movement', '1. Start on hands and knees\n2. Arch back, look up (cow)\n3. Round back, tuck chin (cat)\n4. Move slowly between positions\n5. Breathe with movement', 'beginner', 3, 10, NULL, 5, '["spine stiffness", "low back pain", "postural dysfunction"]', '["acute spinal injury", "spondylolisthesis"]', '["limited spinal mobility", "poor segmental movement"]'),

('Thoracic Rotation', 'mobility', '["spine"]', 'Improves thoracic spine rotation', '1. Lie on side with knees bent\n2. Top arm extended forward\n3. Rotate arm across body to opposite side\n4. Follow hand with eyes\n5. Return slowly', 'beginner', 3, 10, NULL, 5, '["thoracic stiffness", "shoulder pain", "poor rotation"]', '["acute rib injury", "severe pain"]', '["limited thoracic rotation", "hyperkyphosis"]'),

-- Functional movements
('Squat', 'strength', '["hip", "knee", "ankle"]', 'Fundamental lower body strength exercise', '1. Feet shoulder-width apart\n2. Lower hips back and down\n3. Keep chest up, knees aligned\n4. Descend to comfortable depth\n5. Push through heels to stand', 'beginner', 3, 12, NULL, 3, '["general weakness", "functional training", "athletic performance"]', '["acute knee or hip injury", "severe pain"]', '["poor squat mechanics", "weak lower body", "limited ankle dorsiflexion"]'),

('Romanian Deadlift', 'strength', '["hip", "spine"]', 'Strengthens posterior chain', '1. Stand with feet hip-width\n2. Hold weights in front of thighs\n3. Hinge at hips, keep back straight\n4. Lower weights along legs\n5. Feel stretch in hamstrings, return to stand', 'intermediate', 3, 10, NULL, 3, '["weak posterior chain", "hamstring weakness", "poor hip hinge"]', '["acute low back pain", "hamstring strain"]', '["weak hamstrings", "poor hip hinge pattern", "weak glutes"]');

-- Insert default system settings
INSERT INTO system_settings (setting_key, setting_value, setting_type, description) VALUES
('min_session_duration', '10', 'number', 'Minimum session duration in minutes for RPM billing'),
('rpm_monthly_threshold', '16', 'number', 'Minimum days of monitoring required for RPM billing'),
('rpm_communication_time', '20', 'number', 'Minimum interactive communication time in minutes for CPT 98977'),
('default_session_duration', '30', 'number', 'Default exercise session duration in minutes'),
('form_score_threshold', '70', 'number', 'Minimum form score (0-100) for good quality'),
('max_pain_level', '7', 'number', 'Maximum pain level to continue exercise'),
('mediapipe_model_complexity', '1', 'number', 'MediaPipe pose model complexity (0-2)'),
('femto_mega_fps', '30', 'number', 'Femto Mega camera frames per second'),
('skeleton_confidence_threshold', '0.7', 'number', 'Minimum confidence for skeleton tracking');

-- Insert demo clinician (password: demo123)
INSERT OR REPLACE INTO clinicians (email, password_hash, first_name, last_name, title, license_number, license_state, npi_number, role, specialties) VALUES
('demo@physiomotion.com', 'pbkdf2:100000:f977a28edc51f5fdcb2496074ddc02d5:3c2ed044c4f2ccba5a739211ab1f8dca5df3c1b7825553a871cd72cf491c61de', 'Demo', 'Clinician', 'DPT', 'PT12345', 'CA', '1234567890', 'admin', '["orthopedics", "sports medicine", "manual therapy"]');

-- Insert demo patient
INSERT INTO patients (first_name, last_name, date_of_birth, gender, email, phone, address_line1, city, state, zip_code, assessment_reason, chief_complaint, pain_scale, activity_level) VALUES
('John', 'Doe', '1985-06-15', 'male', 'john.doe@example.com', '555-0123', '123 Main St', 'San Francisco', 'CA', '94102', 'post_surgery', 'Right shoulder pain and limited ROM after rotator cuff repair', 4, 'light');
