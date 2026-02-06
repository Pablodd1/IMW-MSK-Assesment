import { detectAsymmetries } from '../src/utils/biomechanics';
import type { JointAngle } from '../src/types';

const jointAngles: Record<string, JointAngle> = {
  left_shoulder_flexion: {
    joint_name: 'Left Shoulder Flexion',
    left_angle: 160,
    normal_range: [0, 180],
    status: 'normal'
  },
  right_shoulder_flexion: {
    joint_name: 'Right Shoulder Flexion',
    right_angle: 140, // Asymmetry: abs(160-140)/160 = 20/160 = 12.5% > 10%
    normal_range: [0, 180],
    status: 'limited'
  },
  left_elbow_flexion: {
    joint_name: 'Left Elbow Flexion',
    left_angle: 140,
    normal_range: [0, 150],
    status: 'normal'
  },
  right_elbow_flexion: {
    joint_name: 'Right Elbow Flexion',
    right_angle: 140,
    normal_range: [0, 150],
    status: 'normal'
  },
  left_hip_flexion: {
    joint_name: 'Left Hip Flexion',
    left_angle: 100,
    normal_range: [0, 120],
    status: 'normal'
  },
  right_hip_flexion: {
    joint_name: 'Right Hip Flexion',
    right_angle: 80, // Asymmetry: abs(100-80)/100 = 20/100 = 20% > 10%
    normal_range: [0, 120],
    status: 'limited'
  },
  left_knee_flexion: {
    joint_name: 'Left Knee Flexion',
    left_angle: 130,
    normal_range: [0, 135],
    status: 'normal'
  },
  right_knee_flexion: {
    joint_name: 'Right Knee Flexion',
    right_angle: 130,
    normal_range: [0, 135],
    status: 'normal'
  },
  left_ankle_dorsiflexion: {
    joint_name: 'Left Ankle Dorsiflexion',
    left_angle: 90,
    normal_range: [70, 110],
    status: 'normal'
  },
  right_ankle_dorsiflexion: {
    joint_name: 'Right Ankle Dorsiflexion',
    right_angle: 90,
    normal_range: [70, 110],
    status: 'normal'
  }
};

const ITERATIONS = 1_000_000;

console.log(`Running benchmark for detectAsymmetries with ${ITERATIONS} iterations...`);

const start = performance.now();

for (let i = 0; i < ITERATIONS; i++) {
  detectAsymmetries(jointAngles);
}

const end = performance.now();
const duration = end - start;

console.log(`Total time: ${duration.toFixed(2)}ms`);
console.log(`Average time per call: ${(duration / ITERATIONS).toFixed(6)}ms`);

// Verification
const result = detectAsymmetries(jointAngles);
console.log('Result:', JSON.stringify(result, null, 2));

// Expect asymmetry in shoulder and hip
if (result['shoulder'] && result['hip'] && !result['elbow']) {
    console.log('Verification PASSED');
} else {
    throw new Error('Verification FAILED');
}
