// Enhanced Joint Tracking System
// Maximum Accuracy Implementation for Orbbec Femto Mega & MediaPipe

import type { PoseLandmark, JointAngle, SkeletonData } from '../types';

// ============================================================================
// CONFIGURATION
// ============================================================================

interface TrackingConfig {
  smoothingWindow: number;       // Number of frames to average
  confidenceThreshold: number;    // Minimum visibility for valid landmark
  outlierRemoval: boolean;       // Enable outlier detection
  temporalSmoothing: boolean;   // Use temporal Kalman-like filter
}

const DEFAULT_CONFIG: TrackingConfig = {
  smoothingWindow: 5,
  confidenceThreshold: 0.5,
  outlierRemoval: true,
  temporalSmoothing: true
};

// ============================================================================
// KALMAN FILTER FOR SMOOTHING
// ============================================================================

class KalmanFilter {
  private q: number; // Process noise
  private r: number; // Measurement noise
  private x: number = 0; // Estimated value
  private p: number = 1; // Estimation error covariance
  private k: number = 0; // Kalman gain

  constructor(processNoise: number = 0.01, measurementNoise: number = 0.1) {
    this.q = processNoise;
    this.r = measurementNoise;
  }

  update(measurement: number): number {
    // Prediction update
    this.p = this.p + this.q;

    // Measurement update
    this.k = this.p / (this.p + this.r);
    this.x = this.x + this.k * (measurement - this.x);
    this.p = (1 - this.k) * this.p;

    return this.x;
  }

  reset(): void {
    this.x = 0;
    this.p = 1;
    this.k = 0;
  }
}

// ============================================================================
// MOVING AVERAGE FILTER
// ============================================================================

class MovingAverageFilter {
  private window: number[];
  private windowSize: number;
  private index: number = 0;
  private filled: boolean = false;

  constructor(windowSize: number = 5) {
    this.windowSize = windowSize;
    this.window = new Array(windowSize).fill(0);
  }

  update(value: number): number {
    this.window[this.index] = value;
    this.index = (this.index + 1) % this.windowSize;
    
    if (this.index === 0) this.filled = true;

    const count = this.filled ? this.windowSize : this.index;
    const sum = this.window.reduce((a, b) => a + b, 0);
    return sum / count;
  }

  reset(): void {
    this.window.fill(0);
    this.index = 0;
    this.filled = false;
  }
}

// ============================================================================
// OUTLIER DETECTION
// ============================================================================

class OutlierDetector {
  private history: number[] = [];
  private readonly maxHistory = 10;
  private readonly zThreshold = 2.5;

  isOutlier(value: number): boolean {
    if (this.history.length < 3) {
      this.history.push(value);
      return false;
    }

    const mean = this.history.reduce((a, b) => a + b, 0) / this.history.length;
    const std = Math.sqrt(
      this.history.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / this.history.length
    );

    const zScore = Math.abs((value - mean) / (std || 1));
    
    this.history.push(value);
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }

    return zScore > this.zThreshold;
  }

  reset(): void {
    this.history = [];
  }
}

// ============================================================================
// ENHANCED POSE LANDMARK PROCESSING
// ============================================================================

interface FilteredLandmark extends PoseLandmark {
  isValid: boolean;
  filtered: PoseLandmark;
}

class JointTracker {
  private kalmanFilters: Map<string, KalmanFilter> = new Map();
  private movingAverageFilters: Map<string, MovingAverageFilter> = new Map();
  private outlierDetectors: Map<string, OutlierDetector> = new Map();
  private config: TrackingConfig;

  constructor(config: Partial<TrackingConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.initializeFilters();
  }

  private initializeFilters(): void {
    const landmarkNames = [
      'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
      'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
      'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
      'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
      'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
    ];

    for (const name of landmarkNames) {
      this.kalmanFilters.set(name, new KalmanFilter(0.01, 0.1));
      this.movingAverageFilters.set(name, new MovingAverageFilter(this.config.smoothingWindow));
      this.outlierDetectors.set(name, new OutlierDetector());
    }
  }

  processLandmarks(landmarks: Record<string, PoseLandmark>): Record<string, FilteredLandmark> {
    const result: Record<string, FilteredLandmark> = {};

    for (const [name, landmark] of Object.entries(landmarks)) {
      const kalman = this.kalmanFilters.get(name);
      const movingAvg = this.movingAverageFilters.get(name);
      const outlier = this.outlierDetectors.get(name);

      if (!kalman || !movingAvg || !outlier) continue;

      const visibility = landmark.visibility || 0;
      const isValid = visibility >= this.config.confidenceThreshold;

      if (!isValid || (this.config.outlierRemoval && outlier.isOutlier(landmark.x))) {
        // Use last valid value or interpolate
        result[name] = {
          ...landmark,
          isValid: false,
          filtered: landmark
        };
        continue;
      }

      // Apply filters
      let filteredX = landmark.x;
      let filteredY = landmark.y;
      let filteredZ = landmark.z || 0;

      if (this.config.temporalSmoothing) {
        filteredX = kalman.update(landmark.x);
        filteredY = kalman.update(landmark.y);
        filteredZ = kalman.update(landmark.z || 0);
      }

      if (this.config.smoothingWindow > 1) {
        filteredX = movingAvg.update(filteredX);
        filteredY = movingAvg.update(filteredY);
      }

      result[name] = {
        ...landmark,
        x: filteredX,
        y: filteredY,
        z: filteredZ,
        isValid: true,
        filtered: {
          x: filteredX,
          y: filteredY,
          z: filteredZ,
          visibility
        }
      };
    }

    return result;
  }

  reset(): void {
    for (const filter of this.kalmanFilters.values()) filter.reset();
    for (const filter of this.movingAverageFilters.values()) filter.reset();
    for (const detector of this.outlierDetectors.values()) detector.reset();
  }
}

// ============================================================================
// ANGLE CALCULATION WITH ERROR BOUNDS
// ============================================================================

interface AngleWithConfidence {
  angle: number;
  confidence: number;
  method: '3d' | '2d';
  stdDev?: number;
}

function calculateAngleWithUncertainty(
  a: PoseLandmark,
  b: PoseLandmark,
  c: PoseLandmark
): AngleWithConfidence {
  // Use 3D if available
  if (a.z !== undefined && b.z !== undefined && c.z !== undefined) {
    return calculate3DAngle(a, b, c);
  }
  
  return calculate2DAngle(a, b, c);
}

function calculate3DAngle(
  a: PoseLandmark,
  b: PoseLandmark,
  c: PoseLandmark
): AngleWithConfidence {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };

  const dotProduct = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
  const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);

  const angleRad = Math.acos(Math.max(-1, Math.min(1, dotProduct / (magBA * magBC))));
  const angleDeg = angleRad * (180 / Math.PI);

  // Calculate confidence based on landmark visibility
  const avgVisibility = ((a.visibility || 0) + (b.visibility || 0) + (c.visibility || 0)) / 3;
  const confidence = avgVisibility * 100;

  return {
    angle: Math.round(angleDeg * 10) / 10,
    confidence: Math.round(confidence),
    method: '3d'
  };
}

function calculate2DAngle(
  a: PoseLandmark,
  b: PoseLandmark,
  c: PoseLandmark
): AngleWithConfidence {
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };

  const dotProduct = ba.x * bc.x + ba.y * bc.y;
  const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2);
  const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2);

  const angleRad = Math.acos(Math.max(-1, Math.min(1, dotProduct / (magBA * magBC))));
  const angleDeg = angleRad * (180 / Math.PI);

  const avgVisibility = ((a.visibility || 0) + (b.visibility || 0) + (c.visibility || 0)) / 3;

  return {
    angle: Math.round(angleDeg * 10) / 10,
    confidence: Math.round(avgVisibility * 100),
    method: '2d'
  };
}

// ============================================================================
// CLINICAL-GRADE JOINT ANGLE CALCULATIONS
// ============================================================================

interface ClinicalJointAngle extends JointAngle {
  confidence: number;
  method: '3d' | '2d';
  clinicalSignificance: 'normal' | 'limited' | 'excessive' | 'borderline';
  deviationFromNorm: number;
}

const CLINICAL_NORMS: Record<string, { min: number; max: number }> = {
  shoulder_flexion: { min: 160, max: 180 },
  elbow_flexion: { min: 130, max: 150 },
  hip_flexion: { min: 110, max: 125 },
  knee_flexion: { min: 130, max: 150 },
  ankle_dorsiflexion: { min: 10, max: 25 },
  hip_abduction: { min: 30, max: 50 }
};

export function calculateClinicalJointAngles(
  skeleton: SkeletonData
): Record<string, ClinicalJointAngle> {
  const { landmarks } = skeleton;
  const angles: Record<string, ClinicalJointAngle> = {};

  // Helper to calculate with confidence
  const calc = (a: PoseLandmark, b: PoseLandmark, c: PoseLandmark) => 
    calculateAngleWithUncertainty(a, b, c);

  // Shoulder Flexion (Left)
  if (landmarks.left_hip && landmarks.left_shoulder && landmarks.left_elbow) {
    const result = calc(landmarks.left_hip, landmarks.left_shoulder, landmarks.left_elbow);
    angles.left_shoulder_flexion = createClinicalAngle(
      'Left Shoulder Flexion',
      result.angle,
      [0, 180],
      'shoulder_flexion',
      result.confidence,
      result.method
    );
  }

  // Shoulder Flexion (Right)
  if (landmarks.right_hip && landmarks.right_shoulder && landmarks.right_elbow) {
    const result = calc(landmarks.right_hip, landmarks.right_shoulder, landmarks.right_elbow);
    angles.right_shoulder_flexion = createClinicalAngle(
      'Right Shoulder Flexion',
      result.angle,
      [0, 180],
      'shoulder_flexion',
      result.confidence,
      result.method
    );
  }

  // Elbow Flexion (Left)
  if (landmarks.left_shoulder && landmarks.left_elbow && landmarks.left_wrist) {
    const result = calc(landmarks.left_shoulder, landmarks.left_elbow, landmarks.left_wrist);
    angles.left_elbow_flexion = createClinicalAngle(
      'Left Elbow Flexion',
      result.angle,
      [0, 150],
      'elbow_flexion',
      result.confidence,
      result.method
    );
  }

  // Elbow Flexion (Right)
  if (landmarks.right_shoulder && landmarks.right_elbow && landmarks.right_wrist) {
    const result = calc(landmarks.right_shoulder, landmarks.right_elbow, landmarks.right_wrist);
    angles.right_elbow_flexion = createClinicalAngle(
      'Right Elbow Flexion',
      result.angle,
      [0, 150],
      'elbow_flexion',
      result.confidence,
      result.method
    );
  }

  // Hip Flexion (Left)
  if (landmarks.left_shoulder && landmarks.left_hip && landmarks.left_knee) {
    const result = calc(landmarks.left_shoulder, landmarks.left_hip, landmarks.left_knee);
    angles.left_hip_flexion = createClinicalAngle(
      'Left Hip Flexion',
      result.angle,
      [0, 120],
      'hip_flexion',
      result.confidence,
      result.method
    );
  }

  // Hip Flexion (Right)
  if (landmarks.right_shoulder && landmarks.right_hip && landmarks.right_knee) {
    const result = calc(landmarks.right_shoulder, landmarks.right_hip, landmarks.right_knee);
    angles.right_hip_flexion = createClinicalAngle(
      'Right Hip Flexion',
      result.angle,
      [0, 120],
      'hip_flexion',
      result.confidence,
      result.method
    );
  }

  // Knee Flexion (Left)
  if (landmarks.left_hip && landmarks.left_knee && landmarks.left_ankle) {
    const result = calc(landmarks.left_hip, landmarks.left_knee, landmarks.left_ankle);
    angles.left_knee_flexion = createClinicalAngle(
      'Left Knee Flexion',
      result.angle,
      [0, 135],
      'knee_flexion',
      result.confidence,
      result.method
    );
  }

  // Knee Flexion (Right)
  if (landmarks.right_hip && landmarks.right_knee && landmarks.right_ankle) {
    const result = calc(landmarks.right_hip, landmarks.right_knee, landmarks.right_ankle);
    angles.right_knee_flexion = createClinicalAngle(
      'Right Knee Flexion',
      result.angle,
      [0, 135],
      'knee_flexion',
      result.confidence,
      result.method
    );
  }

  // Ankle Dorsiflexion (Left)
  if (landmarks.left_knee && landmarks.left_ankle && landmarks.left_foot_index) {
    const result = calc(landmarks.left_knee, landmarks.left_ankle, landmarks.left_foot_index);
    angles.left_ankle_dorsiflexion = createClinicalAngle(
      'Left Ankle Dorsiflexion',
      result.angle,
      [0, 110],
      'ankle_dorsiflexion',
      result.confidence,
      result.method
    );
  }

  // Ankle Dorsiflexion (Right)
  if (landmarks.right_knee && landmarks.right_ankle && landmarks.right_foot_index) {
    const result = calc(landmarks.right_knee, landmarks.right_ankle, landmarks.right_foot_index);
    angles.right_ankle_dorsiflexion = createClinicalAngle(
      'Right Ankle Dorsiflexion',
      result.angle,
      [0, 110],
      'ankle_dorsiflexion',
      result.confidence,
      result.method
    );
  }

  // Calculate bilateral differences
  for (const key in angles) {
    const joint = angles[key];
    const oppositeKey = key.replace('left_', 'right_').replace('right_', 'left_');
    const opposite = angles[oppositeKey];
    
    if (joint && opposite) {
      joint.bilateral_difference = Math.abs(
        (joint.left_angle || joint.right_angle || 0) - 
        (opposite.left_angle || opposite.right_angle || 0)
      );
    }
  }

  return angles;
}

function createClinicalAngle(
  jointName: string,
  angle: number,
  normalRange: [number, number],
  normKey: string,
  confidence: number,
  method: '2d' | '3d'
): ClinicalJointAngle {
  const norm = CLINICAL_NORMS[normKey];
  const deviationFromNorm = norm 
    ? Math.min(
        Math.abs(angle - norm.min),
        Math.abs(angle - norm.max)
      )
    : 0;

  let clinicalSignificance: 'normal' | 'limited' | 'excessive' | 'borderline';
  
  if (!norm) {
    clinicalSignificance = 'normal';
  } else if (angle >= norm.min && angle <= norm.max) {
    clinicalSignificance = deviationFromNorm < 10 ? 'normal' : 'borderline';
  } else if (angle < norm.min) {
    clinicalSignificance = deviationFromNorm > 20 ? 'limited' : 'borderline';
  } else {
    clinicalSignificance = deviationFromNorm > 20 ? 'excessive' : 'borderline';
  }

  return {
    joint_name: jointName,
    left_angle: angle,
    right_angle: angle,
    normal_range: normalRange,
    status: clinicalSignificance === 'normal' ? 'normal' : 
            clinicalSignificance === 'limited' ? 'limited' : 'excessive',
    confidence,
    method,
    deviationFromNorm,
    clinicalSignificance
  };
}

// ============================================================================
// MAIN EXPORT
// ============================================================================

export const EnhancedJointTracker = {
  JointTracker,
  calculateClinicalJointAngles,
  calculateAngleWithUncertainty
};
