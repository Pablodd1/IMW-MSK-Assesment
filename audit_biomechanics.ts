
import { calculateAngle, calculateJointAngles, detectAsymmetries, detectCompensations, calculateMovementQualityScore, generateDeficiencies } from './src/utils/biomechanics';
import type { SkeletonData } from './src/types';

// Mock Skeleton Data (Simulated Squat with Issues)
const mockSkeleton: SkeletonData = {
  timestamp: Date.now(),
  landmarks: {
    // Left side (normal)
    left_hip: { x: 0.1, y: 0.5, z: 0 },
    left_knee: { x: 0.1, y: 0.8, z: 0.2 }, // Knee forward
    left_ankle: { x: 0.1, y: 1.0, z: 0 },
    left_foot_index: { x: 0.1, y: 1.1, z: 0.1 },

    // Right side (valgus/inward knee)
    right_hip: { x: -0.1, y: 0.5, z: 0 },
    right_knee: { x: -0.05, y: 0.8, z: 0.2 }, // Knee caving in (x closer to 0)
    right_ankle: { x: -0.1, y: 1.0, z: 0 },
    right_foot_index: { x: -0.1, y: 1.1, z: 0.1 },

    // Upper body (leaning forward)
    left_shoulder: { x: 0.1, y: 0.2, z: 0.3 }, // Leaning forward (z depth)
    right_shoulder: { x: -0.1, y: 0.2, z: 0.3 },
    left_elbow: { x: 0.2, y: 0.3, z: 0.2 },
    right_elbow: { x: -0.2, y: 0.3, z: 0.2 },
    left_wrist: { x: 0.2, y: 0.4, z: 0.1 },
    right_wrist: { x: -0.2, y: 0.4, z: 0.1 },

    // Filler for required fields
    nose: { x: 0, y: 0, z: 0 },
    left_eye_inner: { x: 0, y: 0, z: 0 },
    left_eye: { x: 0, y: 0, z: 0 },
    left_eye_outer: { x: 0, y: 0, z: 0 },
    right_eye_inner: { x: 0, y: 0, z: 0 },
    right_eye: { x: 0, y: 0, z: 0 },
    right_eye_outer: { x: 0, y: 0, z: 0 },
    left_ear: { x: 0, y: 0, z: 0 },
    right_ear: { x: 0, y: 0, z: 0 },
    mouth_left: { x: 0, y: 0, z: 0 },
    mouth_right: { x: 0, y: 0, z: 0 },
    left_pinky: { x: 0, y: 0, z: 0 },
    right_pinky: { x: 0, y: 0, z: 0 },
    left_index: { x: 0, y: 0, z: 0 },
    right_index: { x: 0, y: 0, z: 0 },
    left_thumb: { x: 0, y: 0, z: 0 },
    right_thumb: { x: 0, y: 0, z: 0 },
    left_heel: { x: 0, y: 0, z: 0 },
    right_heel: { x: 0, y: 0, z: 0 },
  }
};

console.log("Starting Biomechanics Audit...");

// 1. Test Angle Calculation
const angle = calculateAngle(
    {x: 1, y: 0, z: 0}, // A
    {x: 0, y: 0, z: 0}, // B (Vertex)
    {x: 0, y: 1, z: 0}  // C
);
console.log(`Test 90 Degree Angle: ${angle}° (Expected: 90)`);
if (Math.abs(angle - 90) > 0.1) console.error("❌ Angle calculation failed");
else console.log("✅ Angle calculation passed");

// 2. Test Joint Angles
const angles = calculateJointAngles(mockSkeleton);
console.log("\nCalculated Joint Angles:");
console.log(`Left Knee Flexion: ${angles.left_knee_flexion?.left_angle}°`);
console.log(`Right Knee Flexion: ${angles.right_knee_flexion?.right_angle}°`);

// 3. Test Compensation Detection
const compensations = detectCompensations(mockSkeleton, angles);
console.log("\nDetected Compensations:");
compensations.forEach(c => console.log(`- ${c}`));

const hasValgus = compensations.some(c => c.includes("valgus"));
if (hasValgus) console.log("✅ Knee Valgus correctly detected");
else console.error("❌ Failed to detect Knee Valgus");

// 4. Test Scoring
const score = calculateMovementQualityScore(angles, compensations);
console.log(`\nMovement Quality Score: ${score}/100`);

// 5. Test Deficiencies
const asymmetries = detectAsymmetries(angles);
const deficiencies = generateDeficiencies(angles, compensations, asymmetries);
console.log("\nGenerated Deficiencies:");
deficiencies.forEach(d => console.log(`- ${d.area}: ${d.description} (${d.severity})`));

console.log("\nAudit Complete.");
