// TypeScript Type Definitions for Medical Movement Assessment Platform

import type { D1Database, R2Bucket } from '@cloudflare/workers-types';

export type Bindings = {
  DB: D1Database;
  R2: R2Bucket;
}

// ============================================================================
// PATIENT TYPES
// ============================================================================

export interface Patient {
  id?: number;
  // Personal Information
  first_name: string;
  last_name: string;
  date_of_birth: string;
  gender: 'male' | 'female' | 'other' | 'prefer_not_to_say';
  email?: string;
  phone?: string;
  emergency_contact_name?: string;
  emergency_contact_phone?: string;
  
  // Address
  address_line1?: string;
  address_line2?: string;
  city?: string;
  state?: string;
  zip_code?: string;
  country?: string;
  
  // Medical Information
  height_cm?: number;
  weight_kg?: number;
  blood_type?: string;
  
  // Insurance & Billing
  insurance_provider?: string;
  insurance_policy_number?: string;
  insurance_group_number?: string;
  
  // System Fields
  patient_status?: 'active' | 'inactive' | 'discharged';
  referring_physician?: string;
  primary_diagnosis?: string;
  notes?: string;
  
  created_at?: string;
  updated_at?: string;
}

export interface MedicalHistory {
  id?: number;
  patient_id: number;
  
  // Pre/Post Surgery
  surgery_type?: 'pre_surgery' | 'post_surgery' | 'none' | 'athletic_performance';
  surgery_date?: string;
  surgery_description?: string;
  
  // Medical Conditions (stored as JSON strings)
  conditions?: string; // JSON array
  medications?: string; // JSON array
  allergies?: string; // JSON array
  
  // Pain Assessment
  current_pain_level?: number; // 0-10
  pain_location?: string; // JSON array
  pain_description?: string;
  
  // Previous Treatments
  previous_pt_therapy?: string;
  previous_chiropractic?: string;
  previous_surgeries?: string;
  
  // Lifestyle
  activity_level?: 'sedentary' | 'light' | 'moderate' | 'active' | 'very_active';
  occupation?: string;
  sports_activities?: string; // JSON array
  
  // Goals
  treatment_goals?: string;
  
  created_at?: string;
  updated_at?: string;
}

// ============================================================================
// ASSESSMENT TYPES
// ============================================================================

export interface Assessment {
  id?: number;
  patient_id: number;
  clinician_id?: number;
  
  // Assessment Type
  assessment_type: 'initial' | 'progress' | 'discharge' | 'athletic_performance';
  assessment_status?: 'in_progress' | 'completed' | 'cancelled';
  
  // Session Info
  session_date?: string;
  duration_minutes?: number;
  
  // Overall Scores
  overall_score?: number;
  mobility_score?: number;
  stability_score?: number;
  movement_pattern_score?: number;
  
  // Clinical Notes (SOAP Format)
  subjective_findings?: string;
  objective_findings?: string;
  assessment_summary?: string;
  plan?: string;
  
  // Camera Integration
  femto_mega_connected?: number; // boolean
  video_recorded?: number; // boolean
  
  created_at?: string;
  updated_at?: string;
}

export interface MovementTest {
  id?: number;
  assessment_id: number;
  
  // Test Information
  test_name: string;
  test_category?: 'mobility' | 'stability' | 'strength' | 'balance' | 'functional';
  test_order: number;
  
  // Test Status
  test_status?: 'pending' | 'in_progress' | 'completed' | 'skipped';
  
  // Instructions
  instructions: string;
  demo_video_url?: string;
  
  // Timing
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  
  // Camera Data
  camera_recording_url?: string;
  skeleton_data?: string; // JSON
  
  created_at?: string;
}

export interface MovementAnalysis {
  id?: number;
  test_id: number;
  
  // Joint Angles
  joint_angles: string; // JSON
  
  // Range of Motion
  rom_measurements?: string; // JSON
  
  // Asymmetry Detection
  left_right_asymmetry?: string; // JSON
  
  // Movement Quality Scores
  movement_quality_score?: number;
  stability_score?: number;
  compensation_detected?: number; // boolean
  
  // Deficiencies
  deficiencies?: string; // JSON array
  
  // AI Analysis
  ai_confidence_score?: number;
  ai_recommendations?: string; // JSON array
  
  // Biomechanical Data
  velocity_data?: string; // JSON
  acceleration_data?: string; // JSON
  trajectory_data?: string; // JSON
  
  analyzed_at?: string;
}

// ============================================================================
// EXERCISE TYPES
// ============================================================================

export interface Exercise {
  id?: number;
  
  // Exercise Info
  exercise_name: string;
  exercise_category: 'strength' | 'flexibility' | 'balance' | 'mobility' | 'stability' | 'cardio' | 'functional';
  
  // Targets
  target_muscles?: string; // JSON array
  target_joints?: string; // JSON array
  target_movements?: string; // JSON array
  
  // Difficulty
  difficulty_level?: 'beginner' | 'intermediate' | 'advanced';
  
  // Instructions
  description: string;
  instructions: string;
  contraindications?: string;
  
  // Media
  demo_video_url?: string;
  demo_image_url?: string;
  
  // MediaPipe Pose Reference
  reference_keypoints?: string; // JSON
  acceptable_deviation?: number;
  
  // Metadata
  equipment_required?: string; // JSON array
  estimated_duration_seconds?: number;
  
  created_at?: string;
  updated_at?: string;
}

export interface PrescribedExercise {
  id?: number;
  patient_id: number;
  assessment_id?: number;
  exercise_id: number;
  
  // Prescription Details
  sets: number;
  repetitions: number;
  hold_duration_seconds?: number;
  rest_between_sets_seconds?: number;
  
  // Frequency
  times_per_week: number;
  total_weeks?: number;
  
  // Status
  prescription_status?: 'active' | 'completed' | 'discontinued' | 'modified';
  
  // Clinical Reasoning
  clinical_reason?: string;
  target_deficiency?: string;
  
  // Progress Tracking
  compliance_percentage?: number;
  last_performed_at?: string;
  
  prescribed_at?: string;
  prescribed_by?: number; // clinician_id
}

// ============================================================================
// MONITORING TYPES
// ============================================================================

export interface ExerciseSession {
  id?: number;
  patient_id: number;
  prescribed_exercise_id: number;
  
  // Session Info
  session_date?: string;
  completed?: number; // boolean
  
  // Performance Data
  sets_completed?: number;
  reps_completed?: number;
  duration_seconds?: number;
  
  // Quality Assessment
  form_quality_score?: number; // 0-100
  pose_accuracy_data?: string; // JSON
  
  // Errors Detected
  form_errors?: string; // JSON array
  compensation_patterns?: string; // JSON array
  
  // Patient Feedback
  pain_level_during?: number; // 0-10
  difficulty_rating?: number; // 1-5
  patient_notes?: string;
  
  // Media
  recording_url?: string;
  
  analyzed_at?: string;
}

export interface MonitoringAlert {
  id?: number;
  patient_id: number;
  exercise_session_id?: number;
  
  // Alert Type
  alert_type: 'form_error' | 'pain_reported' | 'non_compliance' | 'progress_milestone' | 'concern';
  alert_severity: 'low' | 'medium' | 'high' | 'critical';
  
  // Details
  alert_message: string;
  alert_details?: string; // JSON
  
  // Status
  alert_status?: 'new' | 'reviewed' | 'resolved' | 'dismissed';
  reviewed_by?: number; // clinician_id
  reviewed_at?: string;
  
  created_at?: string;
}

// ============================================================================
// BILLING TYPES
// ============================================================================

export interface BillingCode {
  id?: number;
  cpt_code: string;
  code_description: string;
  code_category: 'evaluation' | 'treatment' | 'rpm' | 'exercise' | 'monitoring';
  
  // Requirements
  minimum_duration_minutes?: number;
  requires_documentation?: number; // boolean
  
  // RPM Specific
  is_rpm_code?: number; // boolean
  rpm_time_requirement_minutes?: number;
  
  created_at?: string;
}

export interface BillableEvent {
  id?: number;
  patient_id: number;
  assessment_id?: number;
  exercise_session_id?: number;
  
  // Billing Info
  cpt_code_id: number;
  service_date: string;
  duration_minutes?: number;
  
  // Documentation
  clinical_note?: string;
  medical_necessity?: string;
  
  // Status
  billing_status?: 'pending' | 'submitted' | 'paid' | 'denied';
  
  // Provider Info
  provider_id?: number; // clinician_id
  provider_npi?: string;
  
  created_at?: string;
}

// ============================================================================
// CLINICIAN TYPES
// ============================================================================

export interface Clinician {
  id?: number;
  
  // Personal Info
  first_name: string;
  last_name: string;
  email: string;
  
  // Credentials
  credential?: string; // PT, DPT, DC, MD, etc.
  license_number?: string;
  npi_number?: string;
  
  // Specialization
  specialty?: 'physical_therapy' | 'chiropractic' | 'sports_medicine' | 'orthopedics' | 'other';
  
  // Account
  account_status?: 'active' | 'inactive' | 'suspended';
  
  created_at?: string;
  updated_at?: string;
}

// ============================================================================
// POSE TRACKING TYPES (MediaPipe & Femto Mega)
// ============================================================================

export interface PoseLandmark {
  x: number; // Normalized [0-1]
  y: number; // Normalized [0-1]
  z: number; // Depth (Femto Mega provides real depth)
  visibility?: number; // Confidence [0-1]
}

export interface SkeletonData {
  timestamp: number;
  landmarks: {
    // 33 key points from MediaPipe Pose / Azure Kinect Body Tracking
    nose: PoseLandmark;
    left_eye_inner: PoseLandmark;
    left_eye: PoseLandmark;
    left_eye_outer: PoseLandmark;
    right_eye_inner: PoseLandmark;
    right_eye: PoseLandmark;
    right_eye_outer: PoseLandmark;
    left_ear: PoseLandmark;
    right_ear: PoseLandmark;
    mouth_left: PoseLandmark;
    mouth_right: PoseLandmark;
    left_shoulder: PoseLandmark;
    right_shoulder: PoseLandmark;
    left_elbow: PoseLandmark;
    right_elbow: PoseLandmark;
    left_wrist: PoseLandmark;
    right_wrist: PoseLandmark;
    left_pinky: PoseLandmark;
    right_pinky: PoseLandmark;
    left_index: PoseLandmark;
    right_index: PoseLandmark;
    left_thumb: PoseLandmark;
    right_thumb: PoseLandmark;
    left_hip: PoseLandmark;
    right_hip: PoseLandmark;
    left_knee: PoseLandmark;
    right_knee: PoseLandmark;
    left_ankle: PoseLandmark;
    right_ankle: PoseLandmark;
    left_heel: PoseLandmark;
    right_heel: PoseLandmark;
    left_foot_index: PoseLandmark;
    right_foot_index: PoseLandmark;
  };
}

export interface JointAngle {
  joint_name: string;
  left_angle?: number;
  right_angle?: number;
  bilateral_difference?: number;
  normal_range: [number, number]; // [min, max]
  status: 'normal' | 'limited' | 'excessive';
}

export interface BiomechanicalAnalysis {
  joint_angles: JointAngle[];
  movement_quality_score: number; // 0-100
  detected_compensations: string[];
  recommendations: string[];
  deficiencies: Array<{
    area: string;
    severity: 'mild' | 'moderate' | 'severe';
    description: string;
    recommended_exercises: number[]; // exercise_ids
  }>;
}

// ============================================================================
// API REQUEST/RESPONSE TYPES
// ============================================================================

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T = any> {
  success: boolean;
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    total_pages: number;
  };
}

// ============================================================================
// FUNCTIONAL MOVEMENT SCREEN (FMS) TESTS
// ============================================================================

export const FMS_TESTS = [
  {
    name: 'Deep Squat',
    category: 'mobility',
    instructions: 'Stand with feet shoulder-width apart. Hold arms overhead with dowel rod. Perform a deep squat keeping heels down, knees out, and arms overhead.',
    demo_video_url: '/videos/deep-squat-demo.mp4',
    order: 1
  },
  {
    name: 'Hurdle Step',
    category: 'stability',
    instructions: 'Stand facing hurdle at knee height. Step over hurdle with one leg while maintaining balance. Touch heel down and return to start.',
    demo_video_url: '/videos/hurdle-step-demo.mp4',
    order: 2
  },
  {
    name: 'Inline Lunge',
    category: 'stability',
    instructions: 'Stand with feet in narrow split stance. Hold dowel rod behind back touching head and sacrum. Lower back knee to ground while maintaining upright torso.',
    demo_video_url: '/videos/inline-lunge-demo.mp4',
    order: 3
  },
  {
    name: 'Shoulder Mobility',
    category: 'mobility',
    instructions: 'Make fists with thumbs inside. Reach one hand over shoulder and other up back. Measure distance between fists.',
    demo_video_url: '/videos/shoulder-mobility-demo.mp4',
    order: 4
  },
  {
    name: 'Active Straight Leg Raise',
    category: 'mobility',
    instructions: 'Lie on back with one leg extended. Raise other leg with straight knee as high as possible while keeping opposite leg flat.',
    demo_video_url: '/videos/aslr-demo.mp4',
    order: 5
  },
  {
    name: 'Trunk Stability Push-up',
    category: 'stability',
    instructions: 'Lie prone with hands at prescribed position. Push body up as unit without allowing hips to sag or pike.',
    demo_video_url: '/videos/pushup-demo.mp4',
    order: 6
  },
  {
    name: 'Rotary Stability',
    category: 'stability',
    instructions: 'Start on hands and knees. Extend opposite arm and leg. Touch elbow to knee under body and return to extended position.',
    demo_video_url: '/videos/rotary-stability-demo.mp4',
    order: 7
  }
] as const;
