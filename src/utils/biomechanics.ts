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
  const angleRad = Math.acos(dotProduct / (magBA * magBC));
  
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
  
  // Shoulder Flexion/Extension (Left)
  try {
    const leftShoulderAngle = calculateAngle(
      landmarks.left_hip,
      landmarks.left_shoulder,
      landmarks.left_elbow
    );
    jointAngles.left_shoulder_flexion = {
      joint_name: 'Left Shoulder Flexion',
      left_angle: leftShoulderAngle,
      normal_range: [0, 180],
      status: leftShoulderAngle >= 150 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating left shoulder angle:', e);
  }
  
  // Shoulder Flexion/Extension (Right)
  try {
    const rightShoulderAngle = calculateAngle(
      landmarks.right_hip,
      landmarks.right_shoulder,
      landmarks.right_elbow
    );
    jointAngles.right_shoulder_flexion = {
      joint_name: 'Right Shoulder Flexion',
      right_angle: rightShoulderAngle,
      normal_range: [0, 180],
      status: rightShoulderAngle >= 150 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating right shoulder angle:', e);
  }
  
  // Elbow Flexion (Left)
  try {
    const leftElbowAngle = calculateAngle(
      landmarks.left_shoulder,
      landmarks.left_elbow,
      landmarks.left_wrist
    );
    jointAngles.left_elbow_flexion = {
      joint_name: 'Left Elbow Flexion',
      left_angle: leftElbowAngle,
      normal_range: [0, 150],
      status: leftElbowAngle >= 130 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating left elbow angle:', e);
  }
  
  // Elbow Flexion (Right)
  try {
    const rightElbowAngle = calculateAngle(
      landmarks.right_shoulder,
      landmarks.right_elbow,
      landmarks.right_wrist
    );
    jointAngles.right_elbow_flexion = {
      joint_name: 'Right Elbow Flexion',
      right_angle: rightElbowAngle,
      normal_range: [0, 150],
      status: rightElbowAngle >= 130 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating right elbow angle:', e);
  }
  
  // Hip Flexion (Left)
  try {
    const leftHipAngle = calculateAngle(
      landmarks.left_shoulder,
      landmarks.left_hip,
      landmarks.left_knee
    );
    jointAngles.left_hip_flexion = {
      joint_name: 'Left Hip Flexion',
      left_angle: leftHipAngle,
      normal_range: [0, 120],
      status: leftHipAngle >= 90 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating left hip angle:', e);
  }
  
  // Hip Flexion (Right)
  try {
    const rightHipAngle = calculateAngle(
      landmarks.right_shoulder,
      landmarks.right_hip,
      landmarks.right_knee
    );
    jointAngles.right_hip_flexion = {
      joint_name: 'Right Hip Flexion',
      right_angle: rightHipAngle,
      normal_range: [0, 120],
      status: rightHipAngle >= 90 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating right hip angle:', e);
  }
  
  // Knee Flexion (Left)
  try {
    const leftKneeAngle = calculateAngle(
      landmarks.left_hip,
      landmarks.left_knee,
      landmarks.left_ankle
    );
    jointAngles.left_knee_flexion = {
      joint_name: 'Left Knee Flexion',
      left_angle: leftKneeAngle,
      normal_range: [0, 135],
      status: leftKneeAngle >= 120 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating left knee angle:', e);
  }
  
  // Knee Flexion (Right)
  try {
    const rightKneeAngle = calculateAngle(
      landmarks.right_hip,
      landmarks.right_knee,
      landmarks.right_ankle
    );
    jointAngles.right_knee_flexion = {
      joint_name: 'Right Knee Flexion',
      right_angle: rightKneeAngle,
      normal_range: [0, 135],
      status: rightKneeAngle >= 120 ? 'normal' : 'limited'
    };
  } catch (e) {
    console.error('Error calculating right knee angle:', e);
  }
  
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

/**
 * Detect asymmetries between left and right sides
 */
export function detectAsymmetries(jointAngles: Record<string, JointAngle>): Record<string, number> {
  const asymmetries: Record<string, number> = {};
  
  const jointPairs = [
    ['left_shoulder_flexion', 'right_shoulder_flexion', 'shoulder'],
    ['left_elbow_flexion', 'right_elbow_flexion', 'elbow'],
    ['left_hip_flexion', 'right_hip_flexion', 'hip'],
    ['left_knee_flexion', 'right_knee_flexion', 'knee'],
    ['left_ankle_dorsiflexion', 'right_ankle_dorsiflexion', 'ankle']
  ];
  
  for (const [leftKey, rightKey, name] of jointPairs) {
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
    
    if (Math.abs(leftKneeX - leftAnkleX) < 0.05) {
      compensations.push('Left knee valgus detected - knee tracking inward');
    }
    if (Math.abs(rightKneeX - rightAnkleX) < 0.05) {
      compensations.push('Right knee valgus detected - knee tracking inward');
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
      severity: (maxAsymmetry > 20 ? 'severe' : maxAsymmetry > 15 ? 'moderate' : 'mild') as 'severe' | 'moderate' | 'mild',
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
