import { test } from 'node:test';
import assert from 'node:assert';
import { EnhancedJointTracker } from '../src/utils/enhanced-tracking.ts';
import type { PoseLandmark, SkeletonData } from '../src/types.ts';

test('calculateAngleWithUncertainty calculates 2D angle correctly', () => {
    // 90 degree angle (L shape)
    const a = { x: 0, y: 1, z: 0, visibility: 0.9 };
    const b = { x: 0, y: 0, z: 0, visibility: 0.9 }; // Vertex
    const c = { x: 1, y: 0, z: 0, visibility: 0.9 };

    const result = EnhancedJointTracker.calculateAngleWithUncertainty(a, b, c);

    // Should be roughly 90 degrees
    assert.ok(Math.abs(result.angle - 90) < 1, `Expected ~90, got ${result.angle}`);
    // Confidence should be 90 (0.9 * 100)
    assert.equal(result.confidence, 90);
});

test('calculateClinicalJointAngles calculates correct clinical angles', () => {
    // Simulate elbow at 90 degrees
    const landmarks: Record<string, PoseLandmark> = {
        'left_shoulder': { x: 0, y: 1, z: 0, visibility: 0.9 },
        'left_elbow': { x: 0, y: 0, z: 0, visibility: 0.9 },
        'left_wrist': { x: 1, y: 0, z: 0, visibility: 0.9 },
        'left_hip': { x: 0, y: 2, z: 0, visibility: 0.9 }, // Needed for shoulder flexion
        'left_knee': { x: 0, y: 3, z: 0, visibility: 0.9 },
        'left_ankle': { x: 0, y: 4, z: 0, visibility: 0.9 }
    };

    const skeleton: SkeletonData = { landmarks };
    const angles = EnhancedJointTracker.calculateClinicalJointAngles(skeleton);

    // Left elbow flexion should be ~90 degrees
    assert.ok(angles.left_elbow_flexion !== undefined);
    assert.ok(Math.abs(angles.left_elbow_flexion.left_angle - 90) < 1);
});

test('Joint filter maintains isolated state per axis and joint', () => {
    const tracker = new EnhancedJointTracker.JointTracker({
        smoothingWindow: 5,
        temporalSmoothing: true,
        outlierRemoval: false
    });

    // Provide steady sequence to fill windows
    for (let i = 0; i < 5; i++) {
      const l1: Record<string, PoseLandmark> = {
          'left_shoulder': { x: 1, y: 1, z: 1, visibility: 0.9 }
      };
      tracker.processLandmarks(l1);
    }

    // Provide jump on X axis only
    const l2: Record<string, PoseLandmark> = {
        'left_shoulder': { x: 10, y: 1, z: 1, visibility: 0.9 }
    };
    const r2 = tracker.processLandmarks(l2);

    // X should be smoothed (not exactly 10, not exactly 1)
    // Y and Z should remain very close to 1
    assert.ok(r2['left_shoulder'].x > 1 && r2['left_shoulder'].x < 10, 'X should be smoothed');
    assert.ok(Math.abs(r2['left_shoulder'].y - 1) < 0.1, 'Y should be unaffected');
    assert.ok(Math.abs((r2['left_shoulder'].z || 0) - 1) < 0.1, 'Z should be unaffected');
});

test('Quality Gate Check: Low confidence triggers flag', () => {
    // If visibility is below threshold, it shouldn't be valid
    const tracker = new EnhancedJointTracker.JointTracker({
        confidenceThreshold: 0.8
    });

    const l: Record<string, PoseLandmark> = {
        'left_shoulder': { x: 1, y: 1, z: 1, visibility: 0.5 }
    };
    const r = tracker.processLandmarks(l);

    assert.equal(r['left_shoulder'].isValid, false);
});
