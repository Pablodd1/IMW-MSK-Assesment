import { test } from 'node:test';
import assert from 'node:assert';
import { detectAsymmetries } from './biomechanics.ts';
import type { JointAngle } from '../types.ts';

test('detectAsymmetries', async (t) => {
  await t.test('should return empty object when no asymmetries are present (all <= 10%)', () => {
    const jointAngles: Record<string, JointAngle> = {
      left_shoulder_flexion: { joint_name: 'Left Shoulder Flexion', left_angle: 100, normal_range: [0, 180], status: 'normal' },
      right_shoulder_flexion: { joint_name: 'Right Shoulder Flexion', right_angle: 95, normal_range: [0, 180], status: 'normal' },
    };
    // (100 - 95) / 100 * 100 = 5%
    const result = detectAsymmetries(jointAngles);
    assert.deepStrictEqual(result, {});
  });

  await t.test('should detect shoulder asymmetry when difference is > 10%', () => {
    const jointAngles: Record<string, JointAngle> = {
      left_shoulder_flexion: { joint_name: 'Left Shoulder Flexion', left_angle: 100, normal_range: [0, 180], status: 'normal' },
      right_shoulder_flexion: { joint_name: 'Right Shoulder Flexion', right_angle: 85, normal_range: [0, 180], status: 'normal' },
    };
    // (100 - 85) / 100 * 100 = 15%
    const result = detectAsymmetries(jointAngles);
    assert.deepStrictEqual(result, { shoulder: 15 });
  });

  await t.test('should detect elbow asymmetry when difference is > 10%', () => {
    const jointAngles: Record<string, JointAngle> = {
      left_elbow_flexion: { joint_name: 'Left Elbow Flexion', left_angle: 90, normal_range: [0, 150], status: 'normal' },
      right_elbow_flexion: { joint_name: 'Right Elbow Flexion', right_angle: 110, normal_range: [0, 150], status: 'normal' },
    };
    // (110 - 90) / 110 * 100 = 18.1818... -> 18.2%
    const result = detectAsymmetries(jointAngles);
    assert.deepStrictEqual(result, { elbow: 18.2 });
  });

  await t.test('should handle boundary case of exactly 10%', () => {
    const jointAngles: Record<string, JointAngle> = {
      left_hip_flexion: { joint_name: 'Left Hip Flexion', left_angle: 100, normal_range: [0, 120], status: 'normal' },
      right_hip_flexion: { joint_name: 'Right Hip Flexion', right_angle: 90, normal_range: [0, 120], status: 'normal' },
    };
    // (100 - 90) / 100 * 100 = 10%
    const result = detectAsymmetries(jointAngles);
    assert.deepStrictEqual(result, {});
  });

  await t.test('should handle slightly above 10%', () => {
    const jointAngles: Record<string, JointAngle> = {
      left_knee_flexion: { joint_name: 'Left Knee Flexion', left_angle: 100, normal_range: [0, 135], status: 'normal' },
      right_knee_flexion: { joint_name: 'Right Knee Flexion', right_angle: 89.9, normal_range: [0, 135], status: 'normal' },
    };
    // (100 - 89.9) / 100 * 100 = 10.1%
    const result = detectAsymmetries(jointAngles);
    assert.deepStrictEqual(result, { knee: 10.1 });
  });

  await t.test('should handle missing data', () => {
    const jointAngles: Record<string, JointAngle> = {
      left_ankle_dorsiflexion: { joint_name: 'Left Ankle Dorsiflexion', left_angle: 90, normal_range: [70, 110], status: 'normal' },
      // right_ankle_dorsiflexion is missing
    };
    const result = detectAsymmetries(jointAngles);
    assert.deepStrictEqual(result, {});
  });

  await t.test('should detect multiple asymmetries', () => {
    const jointAngles: Record<string, JointAngle> = {
      left_shoulder_flexion: { joint_name: 'Left Shoulder Flexion', left_angle: 100, normal_range: [0, 180], status: 'normal' },
      right_shoulder_flexion: { joint_name: 'Right Shoulder Flexion', right_angle: 80, normal_range: [0, 180], status: 'normal' },
      left_hip_flexion: { joint_name: 'Left Hip Flexion', left_angle: 120, normal_range: [0, 120], status: 'normal' },
      right_hip_flexion: { joint_name: 'Right Hip Flexion', right_angle: 100, normal_range: [0, 120], status: 'normal' },
    };
    // Shoulder: (100 - 80) / 100 * 100 = 20%
    // Hip: (120 - 100) / 120 * 100 = 16.666... -> 16.7%
    const result = detectAsymmetries(jointAngles);
    assert.deepStrictEqual(result, { shoulder: 20, hip: 16.7 });
  });

  await t.test('should round percentage to 1 decimal place', () => {
    const jointAngles: Record<string, JointAngle> = {
      left_elbow_flexion: { joint_name: 'Left Elbow Flexion', left_angle: 100, normal_range: [0, 150], status: 'normal' },
      right_elbow_flexion: { joint_name: 'Right Elbow Flexion', right_angle: 88.4, normal_range: [0, 150], status: 'normal' },
    };
    // (100 - 88.4) / 100 * 100 = 11.6%
    const result = detectAsymmetries(jointAngles);
    assert.deepStrictEqual(result, { elbow: 11.6 });
  });
});
