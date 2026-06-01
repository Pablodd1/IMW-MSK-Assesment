/**
 * RAG Knowledge Layer — Embedded Clinical KB
 * 403 chunks from Innovate Medical Wellness unified RAG
 * BGE-M3 embeddings, clinical calculators, ACR criteria, SOAP templates
 * 
 * Used as fallback when Supabase pgvector is unavailable.
 * Production path: Supabase → knowledge_chunks table → vector search
 */

// ── ROM Normative Data (degrees, adult) ──
export const ROM_NORMATIVE: Record<string, { min: number; max: number; unit: string }> = {
  shoulder_flexion:       { min: 160, max: 180, unit: '°' },
  shoulder_extension:     { min: 40,  max: 60,  unit: '°' },
  shoulder_abduction:     { min: 160, max: 180, unit: '°' },
  shoulder_er:            { min: 70,  max: 90,  unit: '°' },
  shoulder_ir:            { min: 60,  max: 80,  unit: '°' },
  elbow_flexion:          { min: 135, max: 150, unit: '°' },
  elbow_extension:        { min: 0,   max: 10,  unit: '°' },
  wrist_flexion:          { min: 60,  max: 80,  unit: '°' },
  wrist_extension:        { min: 55,  max: 75,  unit: '°' },
  hip_flexion:            { min: 110, max: 130, unit: '°' },
  hip_extension:          { min: 20,  max: 30,  unit: '°' },
  hip_abduction:          { min: 35,  max: 50,  unit: '°' },
  hip_er:                 { min: 35,  max: 45,  unit: '°' },
  hip_ir:                 { min: 30,  max: 40,  unit: '°' },
  knee_flexion:           { min: 130, max: 150, unit: '°' },
  knee_extension:         { min: -5,  max: 0,   unit: '°' },
  ankle_df:               { min: 15,  max: 20,  unit: '°' },
  ankle_pf:               { min: 40,  max: 50,  unit: '°' },
  cervical_flexion:       { min: 45,  max: 60,  unit: '°' },
  cervical_extension:     { min: 45,  max: 60,  unit: '°' },
  cervical_rotation:      { min: 60,  max: 80,  unit: '°' },
  lumbar_flexion:         { min: 40,  max: 60,  unit: '°' },
  lumbar_extension:       { min: 20,  max: 30,  unit: '°' },
};

// ── Clinical Calculators ──
export interface ClinicalCalculator {
  name: string;
  formula: string;
  interpretation: Record<string, string>;
}

export const CLINICAL_CALCULATORS: Record<string, ClinicalCalculator> = {
  DASH: {
    name: 'Disabilities of the Arm, Shoulder and Hand',
    formula: '30 items scored 1-5 each → transformed to 0-100 (0 = no disability)',
    interpretation: { '0-20': 'Minimal', '21-40': 'Mild', '41-60': 'Moderate', '61-80': 'Severe', '81-100': 'Very Severe' },
  },
  QuickDASH: {
    name: 'QuickDASH',
    formula: '11 items scored 1-5 each → transformed to 0-100',
    interpretation: { '0-20': 'Minimal', '21-40': 'Mild', '41-60': 'Moderate', '61-80': 'Severe', '81-100': 'Very Severe' },
  },
  KOOS: {
    name: 'Knee Injury and Osteoarthritis Outcome Score',
    formula: '42 items across 5 subscales (Pain, Symptoms, ADL, Sport/Rec, QOL). Each 0-100.',
    interpretation: { '0-25': 'Poor', '26-50': 'Fair', '51-75': 'Good', '76-100': 'Excellent' },
  },
  IKDC: {
    name: 'International Knee Documentation Committee',
    formula: 'Subjective knee evaluation form scored 0-100 (100 = normal function)',
    interpretation: { '0-25': 'Poor', '26-50': 'Fair', '51-75': 'Good', '76-100': 'Excellent' },
  },
  FMS: {
    name: 'Functional Movement Screen',
    formula: '7 tests × 0-3 = 0-21 total',
    interpretation: { '0-14': 'High Injury Risk', '15-17': 'Moderate Risk', '18-21': 'Low Risk' },
  },
  NDI: {
    name: 'Neck Disability Index',
    formula: '10 items × 0-5 = 0-50 total (or 0-100%)',
    interpretation: { '0-4': 'None', '5-14': 'Mild', '15-24': 'Moderate', '25-34': 'Severe', '35-50': 'Complete' },
  },
  SPPB: {
    name: 'Short Physical Performance Battery',
    formula: 'Balance + Gait + Chair stand = 0-12',
    interpretation: { '0-3': 'Severe Limitation', '4-6': 'Moderate', '7-9': 'Mild', '10-12': 'Normal' },
  },
  OttawaAnkle: {
    name: 'Ottawa Ankle Rules',
    formula: 'Ankle X-ray if: malleolar zone pain + (lateral malleolus tenderness OR medial malleolus tenderness OR unable to bear weight 4 steps). Foot X-ray if: midfoot pain + (navicular tenderness OR 5th MT base tenderness OR unable to bear weight)',
    interpretation: { positive: 'X-ray indicated', negative: 'No X-ray needed' },
  },
  OttawaKnee: {
    name: 'Ottawa Knee Rules',
    formula: 'Knee X-ray if ANY: age ≥55 OR isolated patellar tenderness OR fibular head tenderness OR unable to flex to 90° OR unable to bear weight 4 steps',
    interpretation: { positive: 'X-ray indicated', negative: 'No X-ray needed' },
  },
};

// ── Medical Conditions (condensed from 78 conditions) ──
export interface MedicalCondition {
  name: string;
  icd10: string;
  category: string;
  symptoms: string[];
  specialTests: { name: string; sensitivity?: string; specificity?: string }[];
  redFlags: string[];
  treatment: string[];
}

export const MEDICAL_CONDITIONS: MedicalCondition[] = [
  // Shoulder
  { name: 'Rotator Cuff Tear', icd10: 'M75.101', category: 'shoulder',
    symptoms: ['Pain with overhead activity', 'Weakness with abduction', 'Night pain', 'Decreased ROM'],
    specialTests: [
      { name: "Jobe's (Empty Can)", sensitivity: '88%', specificity: '62%' },
      { name: 'Full Can', sensitivity: '70%', specificity: '81%' },
      { name: 'Drop Arm', sensitivity: '24%', specificity: '96%' },
    ],
    redFlags: ['Acute traumatic onset with significant weakness', 'Inability to actively elevate arm', 'Atrophy of supraspinatus/infraspinatus fossa'],
    treatment: ['Physical therapy', 'Corticosteroid injection', 'Surgical repair (full-thickness)', 'Activity modification'] },
  { name: 'Subacromial Impingement', icd10: 'M75.4-', category: 'shoulder',
    symptoms: ['Pain with flexion >90°', 'Pain with overhead activity', 'Night pain', 'Pain with reaching'],
    specialTests: [
      { name: "Neer's Sign", sensitivity: '60%', specificity: '58%' },
      { name: 'Hawkins-Kennedy', sensitivity: '64%', specificity: '48%' },
    ],
    redFlags: ['Severe pain with minimal movement', 'Systemic symptoms', 'History of malignancy'],
    treatment: ['Physical therapy', 'Subacromial injection', 'Activity modification', 'Acromioplasty (refractory)'] },
  { name: 'Adhesive Capsulitis', icd10: 'M75.0', category: 'shoulder',
    symptoms: ['Progressive loss of passive and active ROM', 'Pain at end range', 'Difficulty with overhead activities'],
    specialTests: [],
    redFlags: ['Rapid onset after trauma', 'Systemic symptoms', 'History of malignancy'],
    treatment: ['Physical therapy for ROM', 'Corticosteroid injection', 'Hydrodilatation', 'MUA or arthroscopic release'] },
  { name: 'Anterior Shoulder Dislocation', icd10: 'S43.014A', category: 'shoulder',
    symptoms: ['Severe shoulder pain', 'Visible deformity', 'Inability to move arm', 'Numbness/tingling'],
    specialTests: [{ name: 'Apprehension Test' }],
    redFlags: ['Axillary nerve injury', 'Vascular injury', 'Fracture-dislocation'],
    treatment: ['Closed reduction', 'Immobilization', 'Physical therapy', 'Surgical stabilization (recurrent)'] },

  // Knee
  { name: 'ACL Tear', icd10: 'S83.511A', category: 'knee',
    symptoms: ['Pivoting injury', 'Pop sensation', 'Rapid swelling (hemarthrosis)', 'Instability/giving way'],
    specialTests: [
      { name: 'Lachman (OPD)', sensitivity: '78.6%', specificity: '100%' },
      { name: 'Anterior Drawer (OPD)', sensitivity: '89.3%', specificity: '100%' },
      { name: 'Pivot Shift (OPD)', sensitivity: '75%', specificity: '100%' },
    ],
    redFlags: ['Multi-ligament injury', 'Locked knee', 'Neurovascular compromise', 'Gross instability'],
    treatment: ['Physical therapy (partial)', 'ACL reconstruction (young/active)', 'Criterion-based rehab protocol'] },
  { name: 'Meniscus Tear', icd10: 'S83.2-', category: 'knee',
    symptoms: ['Twisting injury', 'Joint line tenderness', 'Clicking/locking', 'Swelling'],
    specialTests: [
      { name: 'McMurray (Medial)', sensitivity: '35.7%', specificity: '85.7%' },
      { name: 'McMurray (Lateral)', sensitivity: '22.2%', specificity: '100%' },
    ],
    redFlags: ['Locked knee', 'Severe mechanical symptoms', 'Acute ACL + meniscus combination'],
    treatment: ['Physical therapy', 'Arthroscopic meniscectomy', 'Meniscal repair', 'Activity modification'] },
  { name: 'Patellofemoral Pain Syndrome', icd10: 'M22.2-', category: 'knee',
    symptoms: ['Anterior knee pain', 'Pain with stairs/squatting', 'Pain with prolonged sitting', 'Crepitus'],
    specialTests: [{ name: "Clarke's Test" }, { name: 'Patellar Grind' }],
    redFlags: ['Recurrent dislocations', 'Significant swelling', 'Locking'],
    treatment: ['Quadriceps strengthening', 'Hip abductor strengthening', 'Patellar taping', 'Physical therapy'] },
  { name: 'Knee Osteoarthritis', icd10: 'M17.-', category: 'knee',
    symptoms: ['Stiffness', 'Crepitus', 'Pain with weight-bearing', 'Decreased walking distance'],
    specialTests: [],
    redFlags: ['Rapidly progressive deformity', 'Inflammatory features', 'Warmth/erythema'],
    treatment: ['Physical therapy', 'Weight loss', 'NSAIDs', 'Injection', 'Total knee replacement'] },

  // Hip
  { name: 'Femoroacetabular Impingement', icd10: 'M24.6-', category: 'hip',
    symptoms: ['Groin pain', 'Pain with hip flexion', 'Pain with sitting'],
    specialTests: [{ name: 'FADIR', sensitivity: '43-100%', specificity: '10-100%' }],
    redFlags: ['Severe mechanical symptoms', 'Rapidly progressive symptoms'],
    treatment: ['Activity modification', 'Physical therapy', 'Injection', 'Arthroscopic osteoplasty'] },
  { name: 'Hip Osteoarthritis', icd10: 'M16.-', category: 'hip',
    symptoms: ['Stiffness', 'Pain with weight-bearing', 'Groin pain', 'Decreased walking distance'],
    specialTests: [],
    redFlags: ['Rest pain/night pain (possible AVN)', 'Rapidly progressive', 'Corticosteroid use history'],
    treatment: ['Activity modification', 'Physical therapy', 'Weight loss', 'NSAIDs', 'Total hip replacement'] },

  // Spine
  { name: 'Lumbar Disc Herniation', icd10: 'M51.-', category: 'spine',
    symptoms: ['Low back pain', 'Sciatica', 'Numbness/tingling in leg', 'Weakness in LE'],
    specialTests: [
      { name: 'Straight Leg Raise', sensitivity: '80%', specificity: '40%' },
      { name: 'Crossed SLR', sensitivity: '25%', specificity: '90%' },
    ],
    redFlags: ['Cauda equina syndrome', 'Progressive motor deficit', 'Systemic symptoms'],
    treatment: ['Physical therapy', 'NSAIDs', 'Epidural injection', 'Surgical discectomy'] },
  { name: 'Lumbar Spinal Stenosis', icd10: 'M48.0-', category: 'spine',
    symptoms: ['Neurogenic claudication', 'Relief with flexion (shopping cart sign)', 'Bilateral leg symptoms'],
    specialTests: [],
    redFlags: ['Rapidly progressive', 'Bowel/bladder dysfunction', 'Severe functional limitation'],
    treatment: ['Physical therapy (flexion-biased)', 'Epidural injection', 'NSAIDs', 'Laminectomy'] },
  { name: 'Cervical Disc Herniation', icd10: 'M50.-', category: 'spine',
    symptoms: ['Neck pain', 'Radicular arm pain', 'Numbness/tingling in arm', 'Weakness in UE'],
    specialTests: [{ name: "Spurling's Test", sensitivity: '30-50%', specificity: '90-100%' }],
    redFlags: ['Myelopathy signs', 'Progressive weakness', 'Cauda equina-like symptoms'],
    treatment: ['Physical therapy', 'Epidural injection', 'NSAIDs', 'Surgical discectomy/fusion'] },

  // Ankle
  { name: 'Ankle Sprain (Lateral)', icd10: 'S93.4-', category: 'ankle',
    symptoms: ['Inversion injury', 'Lateral ankle pain', 'Swelling', 'Difficulty weight-bearing'],
    specialTests: [{ name: 'Anterior Drawer' }, { name: 'Talar Tilt' }],
    redFlags: ['Unable to bear weight >4 steps', 'Bony tenderness at malleoli', 'Severe deformity'],
    treatment: ['RICE', 'Early mobilization', 'Physical therapy/proprioception', 'Ankle brace'] },
  { name: 'Achilles Tendinopathy', icd10: 'M76.6-', category: 'ankle',
    symptoms: ['Pain 2-6 cm above insertion', 'Morning stiffness', 'Pain with push-off'],
    specialTests: [{ name: 'Thompson Test', sensitivity: '96%', specificity: '93%' }],
    redFlags: ['Sudden onset with pop (possible rupture)', 'Significant weakness'],
    treatment: ['Eccentric loading', 'Physical therapy', 'Heel lifts', 'Avoid corticosteroid injection'] },

  // Wrist/Elbow
  { name: 'Carpal Tunnel Syndrome', icd10: 'G56.0-', category: 'wrist',
    symptoms: ['Numbness/tingling in radial 3.5 digits', 'Nighttime symptoms', 'Weakness of grip'],
    specialTests: [
      { name: "Phalen's", sensitivity: '70-80%', specificity: '80%' },
      { name: "Tinel's", sensitivity: '50-60%', specificity: '70-80%' },
    ],
    redFlags: ['Severe thenar atrophy', 'Constant numbness', 'Failed conservative management'],
    treatment: ['Night splinting', 'Activity modification', 'Corticosteroid injection', 'Surgical release'] },
  { name: 'Lateral Epicondylitis', icd10: 'M77.1-', category: 'elbow',
    symptoms: ['Pain over lateral epicondyle', 'Pain with gripping/lifting', 'Pain with wrist extension'],
    specialTests: [{ name: "Cozen's Test" }, { name: "Maudsley's Test" }],
    redFlags: ['Systemic symptoms', 'Neurological deficits', 'History of malignancy'],
    treatment: ['Physical therapy', 'Eccentric exercises', 'Corticosteroid injection', 'PRP/Shockwave'] },
];

// ── SOAP Note Templates ──
export const SOAP_TEMPLATES: Record<string, string> = {
  physical_therapy: `SOAP NOTE — Physical Therapy
SUBJECTIVE: {chief_complaint} at {pain_scale}/10. {symptom_description}. Aggravating: {aggravating}. Easing: {easing}. Goals: {goals}.
OBJECTIVE: AROM: {arom}. PROM: {prom}. MMT: {mmt}/5. Special tests: {special_tests}. Gait: {gait}. Posture: {posture}.
ASSESSMENT: {diagnosis} (ICD-10: {icd10}). Deficits: {deficits}. Progress: {progress}. Medical necessity: {necessity}.
PLAN: Continue PT {frequency}x/week × {duration}w. Interventions: {interventions}. HEP: {hep}. Next visit: {next_visit}.`,
  orthopedic: `POST-OP NOTE — Orthopedic Surgery
SUBJECTIVE: POD#{pod}. Pain {pain}/10. Wound: {wound}. Concerns: {concerns}.
OBJECTIVE: Vitals: {vitals}. Incision: {incision}. ROM: {rom}. Neurovascular: {neurovascular}.
ASSESSMENT: s/p {procedure} (CPT: {cpt}) for {diagnosis}. Course: {course}. No infection signs.
PLAN: Wound care: {wound_care}. Meds: {meds}. Activity: {activity}. PT: {pt_orders}. F/U: {follow_up}.`,
  sports_medicine: `SPORTS MEDICINE NOTE
SUBJECTIVE: {athlete} with {complaint}. MOI: {mechanism}. Sport: {sport}. Training: {training_load}.
OBJECTIVE: Inspection: {inspection}. Palpation: {palpation}. ROM: {rom}. Strength: {strength}. Special tests: {tests}. Functional: {functional}.
ASSESSMENT: {diagnosis} (ICD-10: {icd10}). Grade: {severity}. RTP status: {rtp}.
PLAN: Immediate: {immediate}. Rehab: {rehab}. RTP criteria: {rtp_criteria}. F/U: {follow_up}.`,
};

// ── Rehabilitation Protocols (condensed) ──
export const REHAB_PROTOCOLS: Record<string, { phases: { name: string; goals: string[]; exercises: string[] }[] }> = {
  acl_reconstruction: {
    phases: [
      { name: 'Phase 1 — Protection (0-2 wks)', goals: ['Protect graft', 'Full extension', '90° flexion', 'Independent ambulation'],
        exercises: ['Heel slides (3×20, 3×/day)', 'Quad sets (5×10, q2h)', 'SLR (3×15, 3×/day)', 'Ankle pumps (3×30, hourly)'] },
      { name: 'Phase 2 — ROM/Gait (2-6 wks)', goals: ['Full ROM 0-135°', 'Normal gait', 'Closed-chain strengthening'],
        exercises: ['Heel raises (3×15)', 'Mini-squats 0-45° (3×15)', 'Terminal knee extension w/ band (3×15)', 'Clamshells (3×15)', 'Step-ups 2-4" (3×10)'] },
      { name: 'Phase 3 — Strength (6-12 wks)', goals: ['Quad 80% contralateral', 'Single-leg squat', 'Jogging progression'],
        exercises: ['Leg press (3×12)', 'Lunges (3×12)', 'Step-ups 8-12" (3×12)', 'Single-leg RDL (3×10)', 'Single-leg balance (3×30s)'] },
      { name: 'Phase 4 — RTP (12-24 wks)', goals: ['Quad 90% contralateral', 'Pass RTP tests', 'Sport-specific'],
        exercises: ['Jogging progression (10-20 min)', 'Agility ladder', 'Lateral shuffles', 'Box jumps (3×8)', 'Cutting/pivoting drills'] },
    ]
  },
  rotator_cuff_repair: {
    phases: [
      { name: 'Phase 1 — Protection (0-6 wks)', goals: ['Protect repair', 'Maintain elbow/wrist ROM', 'Passive ROM only'],
        exercises: ['Pendulum exercises (3×30s, 3×/day)', 'Passive forward flexion (3×10)', 'Passive ER 0-30° (3×10)', 'Isometric shoulder (3×10s)', 'Scapular setting (3×10)'] },
      { name: 'Phase 2 — AAROM (6-12 wks)', goals: ['Full passive ROM', 'Begin active-assisted', 'Light strengthening'],
        exercises: ['Wand exercises (3×10)', 'Wall slides (3×10)', 'Table slides (3×10)', 'Light band IR/ER (3×15)', 'Scapular rows (3×15)'] },
      { name: 'Phase 3 — Strength (12-20 wks)', goals: ['Full active ROM', 'Full strengthening', '80% contralateral strength'],
        exercises: ['Prone extension (3×15)', 'Prone horizontal abduction (3×15)', 'Side-lying ER (3×15)', 'Full can (3×15)', 'Push-up progression (3×10)'] },
      { name: 'Phase 4 — RTP (20-26 wks)', goals: ['Full functional strength', 'Return to sport/work'],
        exercises: ['Sport-specific drills', 'Medicine ball throws (3×10)', 'High-speed strengthening (3×10)', 'Endurance training (3×20-30)'] },
    ]
  },
  ankle_sprain: {
    phases: [
      { name: 'Phase 1 — Acute (0-1 wk)', goals: ['Control pain/swelling', 'Protect from further injury', 'Maintain ROM'],
        exercises: ['Ankle alphabet (2×A-Z, 3×/day)', 'Ankle circles (2×20, 3×/day)', 'Towel scrunches (3×20, 3×/day)', 'Calf raises bilateral (3×15, 3×/day)'] },
      { name: 'Phase 2 — Subacute (1-3 wks)', goals: ['Full ROM', 'Proprioception', 'Normal gait'],
        exercises: ['Towel stretch DF (3×30s, 3×/day)', 'Standing calf stretch (3×30s)', 'Single-leg balance (3×30s)', 'Heel-toe walking (3×10 steps)', 'Theraband 4 directions (3×15)'] },
      { name: 'Phase 3 — Functional (3-6 wks)', goals: ['Full strength', 'Return to activity', 'Prevent recurrence'],
        exercises: ['Single-leg calf raises (3×15)', 'Balance board/BOSU (3×30s)', 'Lateral hopping (3×10)', 'Forward hopping (3×10)', 'Sport-specific drills'] },
    ]
  },
};

// ── CPT Code Reference (MSK-focused) ──
export const CPT_CODES: Record<string, { code: string; description: string; reimbursement: string }> = {
  pt_eval_low:    { code: '97161', description: 'PT evaluation — low complexity', reimbursement: '~$75' },
  pt_eval_mod:    { code: '97162', description: 'PT evaluation — moderate complexity', reimbursement: '~$95' },
  pt_eval_high:   { code: '97163', description: 'PT evaluation — high complexity', reimbursement: '~$115' },
  ther_ex:        { code: '97110', description: 'Therapeutic exercise (15 min)', reimbursement: '~$30/unit' },
  neuro_reed:     { code: '97112', description: 'Neuromuscular re-education (15 min)', reimbursement: '~$32/unit' },
  manual_therapy: { code: '97140', description: 'Manual therapy (15 min)', reimbursement: '~$28/unit' },
  gait_training:  { code: '97116', description: 'Gait training (15 min)', reimbursement: '~$28/unit' },
  ther_act:       { code: '97530', description: 'Therapeutic activities (15 min)', reimbursement: '~$32/unit' },
  ultrasound:     { code: '97035', description: 'Ultrasound (15 min)', reimbursement: '~$15/unit' },
  estim:          { code: '97014', description: 'Electrical stimulation (15 min)', reimbursement: '~$15/unit' },
};

// ── RAG Query (local keyword search fallback) ──
export function searchKnowledge(query: string): string[] {
  const q = query.toLowerCase();
  const results: string[] = [];

  // Search conditions
  for (const c of MEDICAL_CONDITIONS) {
    if (q.includes(c.name.toLowerCase()) || q.includes(c.category) || q.includes(c.icd10.toLowerCase())) {
      results.push(`CONDITION: ${c.name} (ICD-10: ${c.icd10})\nTests: ${c.specialTests.map(t => `${t.name} [Sn:${t.sensitivity || 'N/A'}, Sp:${t.specificity || 'N/A'}]`).join(', ') || 'None'}\nRed Flags: ${c.redFlags.join('; ')}\nTreatment: ${c.treatment.join('; ')}`);
      if (results.length >= 5) break;
    }
  }

  // Search calculators
  for (const [key, calc] of Object.entries(CLINICAL_CALCULATORS)) {
    if (q.includes(key.toLowerCase()) || q.includes(calc.name.toLowerCase())) {
      results.push(`CALCULATOR: ${calc.name}\nFormula: ${calc.formula}\nInterpretation: ${JSON.stringify(calc.interpretation)}`);
    }
  }

  // Search ROM data
  for (const [joint, range] of Object.entries(ROM_NORMATIVE)) {
    if (q.includes(joint.replace('_', ' '))) {
      results.push(`ROM: ${joint.replace(/_/g, ' ')} → Normal ${range.min}-${range.max}${range.unit}`);
    }
  }

  return results.slice(0, 10);
}

// ── Get rehab protocol for a condition ──
export function getRehabProtocol(condition: string): typeof REHAB_PROTOCOLS[string] | null {
  const key = condition.toLowerCase().replace(/[^a-z]/g, '_');
  for (const [protoKey, proto] of Object.entries(REHAB_PROTOCOLS)) {
    if (key.includes(protoKey) || protoKey.includes(key)) return proto;
  }
  return null;
}
