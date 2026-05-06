import { test } from 'node:test';
import assert from 'node:assert';
import { EnhancedJointTracker } from '../src/utils/enhanced-tracking.ts';
import type { PoseLandmark } from '../src/types.ts';

test('JointTracker filtering isolation', () => {
  const tracker = new EnhancedJointTracker.JointTracker({
    smoothingWindow: 5,
    temporalSmoothing: true,
    outlierRemoval: true
  });

  const l1: Record<string, PoseLandmark> = {
    'left_shoulder': { x: 1, y: 1, z: 1, visibility: 0.9 },
    'right_shoulder': { x: 1, y: 1, z: 1, visibility: 0.9 }
  };

  const r1 = tracker.processLandmarks(l1);
  assert.equal(r1['left_shoulder'].isValid, true);
});
