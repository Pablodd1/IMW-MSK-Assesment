export type Keypoint = {
  id: number;
  x: number;
  y: number;
  z?: number;
  confidence?: number;
  visibility?: number;
};

export type GaitPhase = 'heel_strike' | 'midstance' | 'toe_off' | 'swing';

export type GaitMetrics = {
  phase: GaitPhase;
  strideLengthCm: number;
  cadenceSpm: number;
  stepWidthCm: number;
  singleSupportPct: number;
  doubleSupportPct: number;
  pronation: 'neutral' | 'pronation' | 'supination';
  pelvicTiltDeg: number;
  armSwingSymmetryPct: number;
  stanceSide: 'left' | 'right' | 'unknown';
};

export type MuscleGrade = {
  joint: string;
  movement: string;
  muscleGroup: string;
  grade: number;
  forceEstimateN: number;
  side: 'left' | 'right' | 'bilateral';
};

export type ClinicalTestResult = {
  test: string;
  score: number;
  maxScore: number;
  findings: string[];
  measurements: Record<string, number | string>;
};

const KP = {
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
};

export const MUSCLE_GROUPS = [
  { joint: 'Shoulder', movements: ['flexion', 'extension', 'abduction', 'adduction', 'internal rotation', 'external rotation'], groups: ['deltoid', 'latissimus dorsi', 'rotator cuff', 'pectoralis major'] },
  { joint: 'Elbow', movements: ['flexion', 'extension', 'supination', 'pronation'], groups: ['biceps brachii', 'brachialis', 'triceps', 'pronator teres'] },
  { joint: 'Hip', movements: ['flexion', 'extension', 'abduction', 'adduction', 'internal rotation', 'external rotation'], groups: ['iliopsoas', 'gluteus maximus', 'gluteus medius', 'adductors', 'deep rotators'] },
  { joint: 'Knee', movements: ['flexion', 'extension'], groups: ['hamstrings', 'quadriceps'] },
  { joint: 'Ankle', movements: ['dorsiflexion', 'plantarflexion', 'inversion', 'eversion'], groups: ['tibialis anterior', 'gastrocnemius/soleus', 'tibialis posterior', 'peroneals'] },
  { joint: 'Spine', movements: ['flexion', 'extension', 'lateral flexion', 'rotation'], groups: ['deep neck flexors', 'erector spinae', 'multifidi', 'obliques'] },
];

export const EXERCISE_LIBRARY_CLINICAL = [
  { id: 'hip_abductor_strength_01', name: 'Side-Lying Hip Abduction', region: 'hip', diagnosis: ['hip weakness', 'knee valgus', 'gait asymmetry'], difficulty: 'beginner', equipment: ['mat'], sets: 3, reps: '10-12', frequency: '4x/week', cpt: ['97110'], media: 'https://images.unsplash.com/photo-1518611012118-696072aa579a?auto=format&fit=crop&w=900&q=80' },
  { id: 'ankle_control_01', name: 'Short Foot Drill', region: 'ankle', diagnosis: ['pronation', 'balance deficit'], difficulty: 'beginner', equipment: ['none'], sets: 3, reps: '8 holds', frequency: 'daily', cpt: ['97112'], media: 'https://images.unsplash.com/photo-1571019613914-85f342c6a11e?auto=format&fit=crop&w=900&q=80' },
  { id: 'quad_strength_01', name: 'Sit to Stand', region: 'knee', diagnosis: ['quadriceps weakness', 'squat dysfunction'], difficulty: 'beginner', equipment: ['chair'], sets: 3, reps: '8-10', frequency: '4x/week', cpt: ['97110'], media: 'https://images.unsplash.com/photo-1605296867304-46d5465a13f1?auto=format&fit=crop&w=900&q=80' },
  { id: 'thoracic_mobility_01', name: 'Open Book Rotation', region: 'spine', diagnosis: ['thoracic mobility deficit', 'overhead squat dysfunction'], difficulty: 'beginner', equipment: ['mat'], sets: 2, reps: '10/side', frequency: 'daily', cpt: ['97110', '97112'], media: 'https://images.unsplash.com/photo-1599901860904-17e6ed7083a0?auto=format&fit=crop&w=900&q=80' },
  { id: 'scapular_control_01', name: 'Wall Slide with Lift-Off', region: 'shoulder', diagnosis: ['shoulder mobility deficit', 'scapular dyskinesis'], difficulty: 'intermediate', equipment: ['wall'], sets: 3, reps: '8-10', frequency: '4x/week', cpt: ['97110'], media: 'https://images.unsplash.com/photo-1594381898411-846e7d193883?auto=format&fit=crop&w=900&q=80' },
  { id: 'balance_01', name: 'Single-Leg Balance Reach', region: 'balance', diagnosis: ['single leg stance deficit', 'y-balance asymmetry'], difficulty: 'intermediate', equipment: ['none'], sets: 3, reps: '5 reaches/side', frequency: '4x/week', cpt: ['97112'], media: 'https://images.unsplash.com/photo-1594737625785-a6cbdabd333c?auto=format&fit=crop&w=900&q=80' },
];

function byId(keypoints: Keypoint[]) {
  return Object.fromEntries(keypoints.map((kp) => [kp.id, kp])) as Record<number, Keypoint>;
}

function dist(a?: Keypoint, b?: Keypoint) {
  if (!a || !b) return 0;
  const dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + dz ** 2);
}

function angle(a?: Keypoint, b?: Keypoint, c?: Keypoint) {
  if (!a || !b || !c) return 0;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const denom = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!denom) return 0;
  const cos = Math.max(-1, Math.min(1, (ab.x * cb.x + ab.y * cb.y) / denom));
  return Math.round(Math.acos(cos) * 180 / Math.PI);
}

function tiltDeg(a?: Keypoint, b?: Keypoint) {
  if (!a || !b) return 0;
  return Math.round(Math.atan2(a.y - b.y, a.x - b.x) * 180 / Math.PI * 10) / 10;
}

export function analyzeGait(keypoints: Keypoint[], frameRate = 30, history: Keypoint[][] = []): GaitMetrics {
  const k = byId(keypoints);
  const leftAnkle = k[KP.leftAnkle];
  const rightAnkle = k[KP.rightAnkle];
  const leftHip = k[KP.leftHip];
  const rightHip = k[KP.rightHip];
  const leftWrist = k[KP.leftWrist];
  const rightWrist = k[KP.rightWrist];
  const leftShoulder = k[KP.leftShoulder];
  const rightShoulder = k[KP.rightShoulder];
  const hipWidth = Math.max(0.08, dist(leftHip, rightHip));
  const stride = dist(leftAnkle, rightAnkle);
  const stepWidth = Math.abs((leftAnkle?.x || 0) - (rightAnkle?.x || 0));
  const phaseSeed = history.length ? history.length % 40 : Math.round(((leftAnkle?.y || 0.5) + (rightAnkle?.y || 0.5)) * 100) % 40;
  const phase: GaitPhase = phaseSeed < 5 ? 'heel_strike' : phaseSeed < 18 ? 'midstance' : phaseSeed < 25 ? 'toe_off' : 'swing';
  const stanceSide = (leftAnkle?.y || 0) > (rightAnkle?.y || 0) ? 'left' : 'right';
  const ankleRoll = ((leftAnkle?.x || 0) - (leftHip?.x || 0)) - ((rightAnkle?.x || 0) - (rightHip?.x || 0));
  const pronation = ankleRoll > 0.08 ? 'pronation' : ankleRoll < -0.08 ? 'supination' : 'neutral';
  const armSwing = Math.abs(dist(leftWrist, rightShoulder) - dist(rightWrist, leftShoulder));
  const cadenceSpm = Math.round(Math.max(70, Math.min(145, (frameRate * 60) / Math.max(12, 22 + Math.abs(phaseSeed - 20)))));

  return {
    phase,
    strideLengthCm: Math.round((stride / hipWidth) * 42),
    cadenceSpm,
    stepWidthCm: Math.round((stepWidth / hipWidth) * 28),
    singleSupportPct: phase === 'swing' ? 38 : 32,
    doubleSupportPct: phase === 'heel_strike' || phase === 'toe_off' ? 22 : 14,
    pronation,
    pelvicTiltDeg: tiltDeg(leftHip, rightHip),
    armSwingSymmetryPct: Math.max(0, Math.round(100 - (armSwing / hipWidth) * 55)),
    stanceSide,
  };
}

export function assessMuscles(keypoints: Keypoint[], previousKeypoints: Keypoint[] = []): MuscleGrade[] {
  const current = byId(keypoints);
  const previous = byId(previousKeypoints);
  const velocity = (id: number) => dist(current[id], previous[id]);
  const gradeFromVelocity = (v: number, angleValue: number) => {
    if (angleValue < 8 && v < 0.002) return 0;
    if (angleValue < 18) return 1;
    if (angleValue < 40) return 2;
    if (v < 0.012) return 3;
    if (v < 0.028) return 4;
    return 5;
  };

  const specs = [
    ['Shoulder', 'flexion', 'deltoid/anterior shoulder', 'left', KP.leftShoulder, KP.leftElbow, KP.leftHip],
    ['Shoulder', 'flexion', 'deltoid/anterior shoulder', 'right', KP.rightShoulder, KP.rightElbow, KP.rightHip],
    ['Elbow', 'flexion', 'biceps/brachialis', 'left', KP.leftShoulder, KP.leftElbow, KP.leftWrist],
    ['Elbow', 'flexion', 'biceps/brachialis', 'right', KP.rightShoulder, KP.rightElbow, KP.rightWrist],
    ['Hip', 'flexion', 'iliopsoas/rectus femoris', 'left', KP.leftShoulder, KP.leftHip, KP.leftKnee],
    ['Hip', 'flexion', 'iliopsoas/rectus femoris', 'right', KP.rightShoulder, KP.rightHip, KP.rightKnee],
    ['Knee', 'extension', 'quadriceps', 'left', KP.leftHip, KP.leftKnee, KP.leftAnkle],
    ['Knee', 'extension', 'quadriceps', 'right', KP.rightHip, KP.rightKnee, KP.rightAnkle],
    ['Ankle', 'dorsiflexion', 'tibialis anterior', 'left', KP.leftKnee, KP.leftAnkle, KP.leftHip],
    ['Ankle', 'dorsiflexion', 'tibialis anterior', 'right', KP.rightKnee, KP.rightAnkle, KP.rightHip],
  ] as const;

  return specs.map(([joint, movement, muscleGroup, side, a, b, c]) => {
    const angleValue = angle(current[a], current[b], current[c]);
    const v = velocity(b);
    const grade = gradeFromVelocity(v, angleValue);
    return {
      joint,
      movement,
      muscleGroup,
      grade,
      forceEstimateN: Math.round((grade * 18 + v * 1600) * 10) / 10,
      side,
    };
  });
}

export function calculateFuglMeyer(grades: MuscleGrade[]) {
  const upper = grades.filter((g) => ['Shoulder', 'Elbow'].includes(g.joint)).reduce((sum, g) => sum + Math.min(2, Math.round(g.grade / 2.5)), 0);
  const lower = grades.filter((g) => ['Hip', 'Knee', 'Ankle'].includes(g.joint)).reduce((sum, g) => sum + Math.min(2, Math.round(g.grade / 2.5)), 0);
  return {
    upperExtremity: Math.min(66, upper * 5),
    lowerExtremity: Math.min(34, lower * 4),
    total: Math.min(100, upper * 5 + lower * 4),
  };
}

export function runClinicalTests(keypoints: Keypoint[]): ClinicalTestResult[] {
  const k = byId(keypoints);
  const shoulderMobility = Math.abs((k[KP.leftWrist]?.y || 0.5) - (k[KP.rightWrist]?.y || 0.5));
  const kneeValgus = Math.abs(((k[KP.leftKnee]?.x || 0) - (k[KP.leftAnkle]?.x || 0)) - ((k[KP.rightKnee]?.x || 0) - (k[KP.rightAnkle]?.x || 0)));
  const sway = Math.round(Math.abs((k[KP.leftHip]?.x || 0.5) - (k[KP.rightHip]?.x || 0.5)) * 100);
  const spineRom = Math.abs(tiltDeg(k[KP.leftShoulder], k[KP.leftHip])) + Math.abs(tiltDeg(k[KP.rightShoulder], k[KP.rightHip]));
  const yBalance = Math.max(60, Math.min(100, Math.round(100 - kneeValgus * 180)));

  return [
    {
      test: 'Y-Balance Test',
      score: yBalance,
      maxScore: 100,
      findings: yBalance < 90 ? ['Reach asymmetry exceeds screening threshold', 'Add neuromuscular control work'] : ['Reach symmetry acceptable'],
      measurements: { anteriorReachPct: yBalance, posteromedialReachPct: yBalance - 2, posterolateralReachPct: yBalance - 4 },
    },
    {
      test: 'Overhead Squat Assessment',
      score: kneeValgus > 0.08 ? 1 : 2,
      maxScore: 3,
      findings: kneeValgus > 0.08 ? ['Dynamic valgus or foot collapse observed', 'Screen ankle mobility and hip abductor strength'] : ['No major squat dysfunction in current frame'],
      measurements: { kneeValgusIndex: Math.round(kneeValgus * 100) / 100 },
    },
    {
      test: 'Single Leg Stance',
      score: sway < 9 ? 3 : sway < 15 ? 2 : 1,
      maxScore: 3,
      findings: sway >= 15 ? ['Elevated frontal-plane sway'] : ['Sway within screening range'],
      measurements: { swayCmEstimate: sway },
    },
    {
      test: 'Spine ROM',
      score: spineRom > 40 ? 3 : spineRom > 22 ? 2 : 1,
      maxScore: 3,
      findings: spineRom < 22 ? ['Cervical/thoracic/lumbar ROM appears limited'] : ['Spinal ROM screen acceptable'],
      measurements: { combinedRomDeg: Math.round(spineRom) },
    },
    {
      test: 'Apley Scratch Shoulder Mobility',
      score: shoulderMobility < 0.15 ? 3 : shoulderMobility < 0.28 ? 2 : 1,
      maxScore: 3,
      findings: shoulderMobility >= 0.28 ? ['Shoulder mobility asymmetry detected'] : ['Shoulder mobility symmetry acceptable'],
      measurements: { wristHeightAsymmetry: Math.round(shoulderMobility * 100) },
    },
  ];
}

export function prescribeExercises(findings: string[] = [], filters: Record<string, string> = {}) {
  const haystack = findings.join(' ').toLowerCase();
  return EXERCISE_LIBRARY_CLINICAL.filter((exercise) => {
    const diagnosisMatch = !filters.diagnosis || exercise.diagnosis.some((d) => d.includes(filters.diagnosis.toLowerCase()));
    const regionMatch = !filters.region || exercise.region === filters.region;
    const difficultyMatch = !filters.difficulty || exercise.difficulty === filters.difficulty;
    const equipmentMatch = !filters.equipment || exercise.equipment.includes(filters.equipment);
    const findingMatch = !haystack || exercise.diagnosis.some((d) => haystack.includes(d.split(' ')[0]));
    return diagnosisMatch && regionMatch && difficultyMatch && equipmentMatch && (findingMatch || Object.keys(filters).length > 0);
  });
}

export function calculateProgress(current: number, baseline: number) {
  if (!baseline) return 0;
  return Math.round(((current - baseline) / Math.abs(baseline)) * 100);
}

