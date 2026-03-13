// Biomechanical Analysis Utilities
// Joint angle calculations, ROM analysis, and deficiency detection

import type { PoseLandmark, JointAngle, BiomechanicalAnalysis, SkeletonData } from '../types';

/**
 * Calculate angle between three points in 3D space
 * @param a - First point (e.g., shoulder)
 * @param b - Middle point (vertex, e.g., elbow)
 * @param c - Third point (e.g., wrist)
 * @returns Angle in degrees
 */
export function calculateAngle(a: PoseLandmark, b: PoseLandmark, c: PoseLandmark): number {
  // Vector BA
  const ba = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: a.z - b.z
  };

  // Vector BC
  const bc = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: c.z - b.z
  };

  // Dot product
  const dotProduct = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;

  // Magnitudes
  const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
  const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z);

  // Angle in radians
  const angleRad = Math.acos(Math.max(-1, Math.min(1, dotProduct / (magBA * magBC))));

  // Convert to degrees
  const angleDeg = angleRad * (180 / Math.PI);

  return Math.round(angleDeg * 10) / 10; // Round to 1 decimal place
}

/**
 * Calculate distance between two points
 */
function calculateDistance(a: PoseLandmark, b: PoseLandmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Calculate all joint angles from skeleton data
 */
export function calculateJointAngles(skeleton: SkeletonData): Record<string, JointAngle> {
  const { landmarks } = skeleton;

  const jointAngles: Record<string, JointAngle> = {};

/**
 * Calculate all joint angles from skeleton data - Optimized with pre-computed mappings
 */
export function calculateJointAngles(skeleton: SkeletonData): Record<string, JointAngle> {
  const { landmarks } = skeleton;
  const jointAngles: Record<string, JointAngle> = {};

  // Define joint configurations to loop through
  const configs = [
    { key: 'left_shoulder_flexion', name: 'Left Shoulder Flexion', p1: landmarks.left_hip, p2: landmarks.left_shoulder, p3: landmarks.left_elbow, range: [0, 180], threshold: 150 },
    { key: 'right_shoulder_flexion', name: 'Right Shoulder Flexion', p1: landmarks.right_hip, p2: landmarks.right_shoulder, p3: landmarks.right_elbow, range: [0, 180], threshold: 150 },
    { key: 'left_elbow_flexion', name: 'Left Elbow Flexion', p1: landmarks.left_shoulder, p2: landmarks.left_elbow, p3: landmarks.left_wrist, range: [0, 150], threshold: 130 },
    { key: 'right_elbow_flexion', name: 'Right Elbow Flexion', p1: landmarks.right_shoulder, p2: landmarks.right_elbow, p3: landmarks.right_wrist, range: [0, 150], threshold: 130 },
    { key: 'left_hip_flexion', name: 'Left Hip Flexion', p1: landmarks.left_shoulder, p2: landmarks.left_hip, p3: landmarks.left_knee, range: [0, 120], threshold: 90 },
    { key: 'right_hip_flexion', name: 'Right Hip Flexion', p1: landmarks.right_shoulder, p2: landmarks.right_hip, p3: landmarks.right_knee, range: [0, 120], threshold: 90 },
    { key: 'left_knee_flexion', name: 'Left Knee Flexion', p1: landmarks.left_hip, p2: landmarks.left_knee, p3: landmarks.left_ankle, range: [0, 135], threshold: 120 },
    { key: 'right_knee_flexion', name: 'Right Knee Flexion', p1: landmarks.right_hip, p2: landmarks.right_knee, p3: landmarks.right_ankle, range: [0, 135], threshold: 120 },
  ];

  for (const config of configs) {
    try {
      const angle = calculateAngle(config.p1, config.p2, config.p3);
      jointAngles[config.key] = {
        joint_name: config.name,
        left_angle: config.key.startsWith('left') ? angle : undefined,
        right_angle: config.key.startsWith('right') ? angle : undefined,
        normal_range: config.range as [number, number],
        status: angle >= config.threshold ? 'normal' : 'limited'
      };
    } catch (e) {
      console.error(`Error calculating ${config.name}:`, e);
    }
  }

  // ... (Keep existing bilateral difference logic)


  // Ankle Dorsiflexion (Left)
  try {
    const leftAnkleAngle = calculateAngle(
      landmarks.left_knee,
      landmarks.left_ankle,
      landmarks.left_foot_index
    );
    jointAngles.left_ankle_dorsiflexion = {
      joint_name: 'Left Ankle Dorsiflexion',
      left_angle: leftAnkleAngle,
      normal_range: [70, 110],
      status: leftAnkleAngle >= 85 && leftAnkleAngle <= 105 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating left ankle angle:', e);
  }

  // Ankle Dorsiflexion (Right)
  try {
    const rightAnkleAngle = calculateAngle(
      landmarks.right_knee,
      landmarks.right_ankle,
      landmarks.right_foot_index
    );
    jointAngles.right_ankle_dorsiflexion = {
      joint_name: 'Right Ankle Dorsiflexion',
      right_angle: rightAnkleAngle,
      normal_range: [70, 110],
      status: rightAnkleAngle >= 85 && rightAnkleAngle <= 105 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating right ankle angle:', e);
  }

  // Calculate bilateral differences
  for (const key in jointAngles) {
    const joint = jointAngles[key];
    if (joint.left_angle !== undefined && joint.right_angle !== undefined) {
      joint.bilateral_difference = Math.abs(joint.left_angle - joint.right_angle);
    }
  }

  return jointAngles;
}

const JOINT_ANGLE_PAIRS: Array<[string, string, string]> = [
  ['left_shoulder_flexion', 'right_shoulder_flexion', 'shoulder'],
  ['left_elbow_flexion', 'right_elbow_flexion', 'elbow'],
  ['left_hip_flexion', 'right_hip_flexion', 'hip'],
  ['left_knee_flexion', 'right_knee_flexion', 'knee'],
  ['left_ankle_dorsiflexion', 'right_ankle_dorsiflexion', 'ankle']
];

/**
 * Detect asymmetries between left and right sides
 */
export function detectAsymmetries(jointAngles: Record<string, JointAngle>): Record<string, number> {
  const asymmetries: Record<string, number> = {};
  
  for (const [leftKey, rightKey, name] of JOINT_ANGLE_PAIRS) {
    const leftJoint = jointAngles[leftKey];
    const rightJoint = jointAngles[rightKey];

    if (leftJoint?.left_angle !== undefined && rightJoint?.right_angle !== undefined) {
      const difference = Math.abs(leftJoint.left_angle - rightJoint.right_angle);
      const percentage = (difference / Math.max(leftJoint.left_angle, rightJoint.right_angle)) * 100;

      if (percentage > 10) { // More than 10% difference is significant
        asymmetries[name] = Math.round(percentage * 10) / 10;
      }
    }
  }

  return asymmetries;
}

/**
 * Detect compensation patterns in movement
 */
export function detectCompensations(skeleton: SkeletonData, jointAngles: Record<string, JointAngle>): string[] {
  const compensations: string[] = [];
  const { landmarks } = skeleton;

  try {
    // Check for knee valgus (knees caving in during squat)
    const leftKneeX = landmarks.left_knee.x;
    const rightKneeX = landmarks.right_knee.x;
    const leftAnkleX = landmarks.left_ankle.x;
    const rightAnkleX = landmarks.right_ankle.x;

    // Check for knee valgus (knees caving in during squat)
    // In front view, knee should be relatively aligned with ankle.
    // Medial deviation of knee relative to hip-ankle line indicates valgus.
    const leftHipX = landmarks.left_hip.x;
    const rightHipX = landmarks.right_hip.x;

    // For left leg, if knee X is greater than ankle X (more medial in most views)
    if (leftKneeX > leftAnkleX + 0.05) {
      compensations.push('Left knee valgus detected - knee tracking medial to ankle');
    }
    // For right leg, if knee X is less than ankle X
    if (rightKneeX < rightAnkleX - 0.05) {
      compensations.push('Right knee valgus detected - knee tracking medial to ankle');
    }

    // Check for excessive forward lean (torso angle)
    const hipToShoulder = {
      x: landmarks.left_shoulder.x - landmarks.left_hip.x,
      y: landmarks.left_shoulder.y - landmarks.left_hip.y
    };
    const forwardLeanAngle = Math.atan2(hipToShoulder.x, Math.abs(hipToShoulder.y)) * (180 / Math.PI);

    if (forwardLeanAngle > 30) {
      compensations.push('Excessive forward trunk lean - core weakness or hip mobility limitation');
    }

    // Check for heel lift during squat
    const leftHeelToToe = Math.abs(landmarks.left_heel.y - landmarks.left_foot_index.y);
    const rightHeelToToe = Math.abs(landmarks.right_heel.y - landmarks.right_foot_index.y);

    if (leftHeelToToe > 0.15) {
      compensations.push('Left heel lifting - ankle dorsiflexion limitation');
    }
    if (rightHeelToToe > 0.15) {
      compensations.push('Right heel lifting - ankle dorsiflexion limitation');
    }

    // Check for uneven shoulder height
    const shoulderHeightDiff = Math.abs(landmarks.left_shoulder.y - landmarks.right_shoulder.y);
    if (shoulderHeightDiff > 0.08) {
      compensations.push('Shoulder height asymmetry - possible lateral trunk lean or shoulder dysfunction');
    }

    // Check for hip asymmetry
    const hipHeightDiff = Math.abs(landmarks.left_hip.y - landmarks.right_hip.y);
    if (hipHeightDiff > 0.08) {
      compensations.push('Pelvic obliquity - one hip higher than the other');
    }

  } catch (e) {
    console.error('Error detecting compensations:', e);
  }

  return compensations;
}

/**
 * Calculate movement quality score based on joint angles and compensations
 */
export function calculateMovementQualityScore(
  jointAngles: Record<string, JointAngle>,
  compensations: string[]
): number {
  let score = 100;

  // Deduct points for limited ROM
  for (const key in jointAngles) {
    const joint = jointAngles[key];
    if (joint.status === 'limited') {
      score -= 10;
    } else if (joint.status === 'excessive') {
      score -= 5;
    }
  }

  // Deduct points for asymmetries
  for (const key in jointAngles) {
    const joint = jointAngles[key];
    if (joint.bilateral_difference && joint.bilateral_difference > 10) {
      score -= 8;
    }
  }

  // Deduct points for compensations
  score -= compensations.length * 7;

  return Math.max(0, Math.min(100, score));
}

/**
 * Generate deficiency report with recommended exercises
 */
export function generateDeficiencies(
  jointAngles: Record<string, JointAngle>,
  compensations: string[],
  asymmetries: Record<string, number>
): Array<{
  area: string;
  severity: 'mild' | 'moderate' | 'severe';
  description: string;
  recommended_exercises: number[];
}> {
  const deficiencies = [];

  // Check ankle mobility
  const leftAnkle = jointAngles.left_ankle_dorsiflexion;
  const rightAnkle = jointAngles.right_ankle_dorsiflexion;

  if ((leftAnkle?.status === 'limited' || rightAnkle?.status === 'limited')) {
    deficiencies.push({
      area: 'Ankle Dorsiflexion',
      severity: 'moderate' as const,
      description: 'Limited ankle dorsiflexion range of motion. This can lead to heel lifting during squats and compensation patterns up the kinetic chain.',
      recommended_exercises: [3] // Hip Flexor Stretch also helps ankles
    });
  }

  // Check hip mobility
  const leftHip = jointAngles.left_hip_flexion;
  const rightHip = jointAngles.right_hip_flexion;

  if ((leftHip?.status === 'limited' || rightHip?.status === 'limited')) {
    deficiencies.push({
      area: 'Hip Flexion',
      severity: 'moderate' as const,
      description: 'Limited hip flexion range of motion. This affects squat depth and can cause excessive forward trunk lean.',
      recommended_exercises: [1, 3] // Deep Squat, Hip Flexor Stretch
    });
  }

  // Check shoulder mobility
  const leftShoulder = jointAngles.left_shoulder_flexion;
  const rightShoulder = jointAngles.right_shoulder_flexion;

  if ((leftShoulder?.status === 'limited' || rightShoulder?.status === 'limited')) {
    deficiencies.push({
      area: 'Shoulder Flexion',
      severity: 'mild' as const,
      description: 'Limited overhead shoulder mobility. This can affect overhead movements and daily activities.',
      recommended_exercises: [11, 12] // Shoulder exercises
    });
  }

  // Check for stability issues from compensations
  if (compensations.some(c => c.includes('knee valgus') || c.includes('balance'))) {
    deficiencies.push({
      area: 'Lower Extremity Stability',
      severity: 'moderate' as const,
      description: 'Instability and poor movement control in lower extremity. Knee valgus and balance deficits detected.',
      recommended_exercises: [4, 5] // Single Leg Balance, Plank Hold
    });
  }

  // Check for core stability
  if (compensations.some(c => c.includes('trunk lean') || c.includes('core'))) {
    deficiencies.push({
      area: 'Core Stability',
      severity: 'moderate' as const,
      description: 'Weak core stability causing compensatory trunk movements and poor postural control.',
      recommended_exercises: [5, 13, 14] // Plank, Dead Bug, Bird Dog
    });
  }

  // Check for significant asymmetries
  if (Object.keys(asymmetries).length > 0) {
    const maxAsymmetry = Math.max(...Object.values(asymmetries));
    deficiencies.push({
      area: 'Bilateral Asymmetry',
      severity: maxAsymmetry > 20 ? 'severe' : maxAsymmetry > 15 ? 'moderate' : 'mild',
      description: `Significant differences between left and right sides detected. Asymmetries found in: ${Object.keys(asymmetries).join(', ')}. This increases injury risk and affects functional performance.`,
      recommended_exercises: [7, 8] // Single Leg RDL, Lunge Pattern
    });
  }

  return deficiencies;
}

/**
 * Perform complete biomechanical analysis
 */
export function performBiomechanicalAnalysis(skeleton: SkeletonData): BiomechanicalAnalysis {
  // Calculate all joint angles
  const jointAngles = calculateJointAngles(skeleton);

  // Detect asymmetries
  const asymmetries = detectAsymmetries(jointAngles);

  // Detect compensations
  const compensations = detectCompensations(skeleton, jointAngles);

  // Calculate movement quality score
  const movementQualityScore = calculateMovementQualityScore(jointAngles, compensations);

  // Generate deficiencies and recommendations
  const deficiencies = generateDeficiencies(jointAngles, compensations, asymmetries);

  // Generate AI recommendations
  const recommendations = generateRecommendations(jointAngles, compensations, deficiencies);

  return {
    joint_angles: Object.values(jointAngles),
    movement_quality_score: movementQualityScore,
    detected_compensations: compensations,
    recommendations,
    deficiencies
  };
}

/**
 * Generate AI recommendations based on analysis
 */
function generateRecommendations(
  jointAngles: Record<string, JointAngle>,
  compensations: string[],
  deficiencies: any[]
): string[] {
  const recommendations: string[] = [];

  // Add general recommendations
  recommendations.push('Continue with prescribed exercise program focusing on identified deficiencies');

  // Add specific recommendations based on findings
  if (compensations.length > 3) {
    recommendations.push('Multiple compensation patterns detected - recommend reducing exercise complexity until fundamental movement patterns improve');
  }

  if (deficiencies.some(d => d.severity === 'severe')) {
    recommendations.push('Severe deficiencies identified - recommend twice-weekly supervised therapy sessions');
  }

  if (Object.keys(jointAngles).some(key => jointAngles[key].status === 'limited')) {
    recommendations.push('Perform daily mobility work for 10-15 minutes focusing on identified ROM limitations');
  }

  recommendations.push('Re-assess movement quality in 2-3 weeks to track progress');

  return recommendations;
}
