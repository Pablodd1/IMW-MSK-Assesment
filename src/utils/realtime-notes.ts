// Real-Time Medical Notes Generator
// Generates SOAP notes from live movement tracking data

import type { BiomechanicalAnalysis, ClinicalJointAngle } from '../types';

interface LiveTrackingSession {
  sessionId: string;
  patientId: number;
  startTime: Date;
  endTime?: Date;
  cameraType: 'phone' | 'laptop' | 'femto_mega';
  movementsRecorded: MovementRecording[];
}

interface MovementRecording {
  timestamp: number;
  movementName: string;
  duration: number;
  jointAngles: Record<string, ClinicalJointAngle>;
  compensations: string[];
  qualityScore: number;
  repetitions?: number;
}

interface MedicalNoteOutput {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  cptCodes: string[];
  icd10Codes: string[];
  additionalNotes: string;
}

// Generate real-time medical note from live tracking
export function generateLiveMedicalNote(
  session: LiveTrackingSession,
  analysis: BiomechanicalAnalysis,
  patientData?: {
    name?: string;
    age?: number;
    chiefComplaint?: string;
  }
): MedicalNoteOutput {
  const duration = session.endTime 
    ? Math.round((session.endTime.getTime() - session.startTime.getTime()) / 60000)
    : 0;

  const movements = session.movementsRecorded;
  const avgQualityScore = movements.length > 0
    ? Math.round(movements.reduce((sum, m) => sum + m.qualityScore, 0) / movements.length)
    : 0;

  // Build Subjective
  const subjective = buildSubjective(patientData, avgQualityScore, movements);

  // Build Objective  
  const objective = buildObjective(movements, analysis);

  // Build Assessment
  const assessment = buildAssessment(analysis, avgQualityScore, movements);

  // Build Plan
  const plan = buildPlan(analysis, avgQualityScore);

  // Get CPT codes
  const cptCodes = suggestCPTFromTracking(session, analysis);

  // Get ICD-10 codes
  const icd10Codes = suggestICD10FromAnalysis(analysis);

  return {
    subjective,
    objective,
    assessment,
    plan,
    cptCodes,
    icd10Codes,
    additionalNotes: generateAdditionalNotes(session, analysis)
  };
}

function buildSubjective(
  patientData: any,
  qualityScore: number,
  movements: MovementRecording[]
): string {
  const parts: string[] = [];

  if (patientData?.name) {
    parts.push(`Patient: ${patientData.name}`);
  }
  if (patientData?.chiefComplaint) {
    parts.push(`Chief Complaint: ${patientData.chiefComplaint}`);
  }

  parts.push(`Assessment Duration: ${movements.length} movement tests completed`);
  
  if (qualityScore >= 80) {
    parts.push('Patient demonstrated good movement quality throughout the assessment.');
  } else if (qualityScore >= 60) {
    parts.push('Patient demonstrated fair movement quality with some compensations observed.');
  } else {
    parts.push('Patient demonstrated limited movement quality with significant compensations noted.');
  }

  return parts.join('. ');
}

function buildObjective(
  movements: MovementRecording[],
  analysis: BiomechanicalAnalysis
): string {
  const lines: string[] = [];

  // Movement Quality
  lines.push(`**Movement Quality Score: ${analysis.movement_quality_score}/100**`);
  lines.push('');

  // Recorded Movements
  lines.push('**Movements Assessed:**');
  for (const movement of movements) {
    const status = movement.qualityScore >= 70 ? '✓' : '⚠';
    lines.push(`  ${status} ${movement.movementName}: ${movement.qualityScore}/100 (${movement.duration}s)`);
  }
  lines.push('');

  // Joint Angle Findings
  lines.push('**Range of Motion Findings:**');
  const jointAngles = Object.values(analysis.deficiencies || []);
  
  if (jointAngles.length > 0) {
    for (const def of jointAngles.slice(0, 5)) {
      lines.push(`  • ${def.area}: ${def.severity} limitation - ${def.description}`);
    }
  } else {
    lines.push('  All measured joint angles within normal limits.');
  }
  lines.push('');

  // Compensations
  if (analysis.detected_compensations?.length > 0) {
    lines.push('**Compensatory Patterns Observed:**');
    for (const comp of analysis.detected_compensations.slice(0, 3)) {
      lines.push(`  • ${comp}`);
    }
  }

  return lines.join('\n');
}

function buildAssessment(
  analysis: BiomechanicalAnalysis,
  qualityScore: number,
  movements: MovementRecording[]
): string {
  const lines: string[] = [];

  // Overall assessment
  if (qualityScore >= 80) {
    lines.push('Patient demonstrates GOOD functional movement quality.');
  } else if (qualityScore >= 60) {
    lines.push('Patient demonstrates FAIR functional movement quality with modifiable compensations.');
  } else {
    lines.push('Patient demonstrates POOR functional movement quality requiring intervention.');
  }

  // Deficiencies summary
  if (analysis.deficiencies?.length > 0) {
    lines.push('');
    lines.push('**Identified Deficiencies:**');
    for (const def of analysis.deficiencies) {
      lines.push(`• ${def.area}: ${def.severity} - ${def.description}`);
    }
  }

  // Recommendations summary
  if (analysis.recommendations?.length > 0) {
    lines.push('');
    lines.push('**Clinical Recommendations:**');
    for (const rec of analysis.recommendations.slice(0, 3)) {
      lines.push(`• ${rec}`);
    }
  }

  return lines.join('\n');
}

function buildPlan(analysis: BiomechanicalAnalysis, qualityScore: number): string {
  const lines: string[] = [];

  lines.push('**Treatment Plan:**');
  lines.push('');

  // Frequency based on quality
  const frequency = qualityScore < 60 ? '2-3 times per week' : '1-2 times per week';
  const duration = qualityScore < 60 ? '8-12 weeks' : '4-8 weeks';

  lines.push(`1. **Therapeutic Exercise Program**: ${frequency} for approximately ${duration}`);
  
  if (analysis.deficiencies?.some(d => d.area.includes('Hip') || d.area.includes('Ankle'))) {
    lines.push('   - Focus on lower extremity mobility and stability');
  }
  
  if (analysis.deficiencies?.some(d => d.area.includes('Shoulder') || d.area.includes('Core'))) {
    lines.push('   - Include upper body and core strengthening');
  }

  lines.push('');
  lines.push('2. **Home Exercise Program**: Daily compliance required');

  if (analysis.detected_compensations?.length > 0) {
    lines.push('');
    lines.push('3. **Movement Pattern Retraining**: Address compensatory patterns identified');
  }

  lines.push('');
  lines.push('4. **Re-assessment**: Schedule follow-up evaluation in 4 weeks to assess progress');

  if (analysis.deficiencies?.some(d => d.severity === 'severe')) {
    lines.push('');
    lines.push('5. **Additional Notes**: Consider additional evaluation or referral for severe limitations');
  }

  return lines.join('\n');
}

function suggestCPTFromTracking(
  session: LiveTrackingSession,
  analysis: BiomechanicalAnalysis
): string[] {
  const codes: string[] = [];

  // Evaluation code
  codes.push('97163'); // PT Eval High Complexity (typically used for comprehensive assessment)

  // Therapeutic exercise
  if (analysis.movement_quality_score < 80) {
    codes.push('97110'); // Therapeutic Exercise
  }

  // Neuromuscular if compensations found
  if (analysis.detected_compensations?.length > 0) {
    codes.push('97112'); // Neuromuscular Re-education
  }

  // Manual therapy if mobility issues
  if (analysis.deficiencies?.some(d => 
    d.area.includes('Mobility') || d.area.includes('Flexion')
  )) {
    codes.push('97140'); // Manual Therapy
  }

  // Gait if lower extremity issues
  if (analysis.deficiencies?.some(d => 
    d.area.includes('Ankle') || d.area.includes('Hip') || d.area.includes('Knee')
  )) {
    codes.push('97116'); // Gait Training
  }

  // Remote monitoring
  codes.push('98977'); // RTM Treatment Management

  return [...new Set(codes)]; // Remove duplicates
}

function suggestICD10FromAnalysis(analysis: BiomechanicalAnalysis): string[] {
  const codes: string[] = [];
  const areas = analysis.deficiencies?.map(d => d.area.toLowerCase()) || [];

  // Based on identified deficiencies
  if (areas.some(a => a.includes('shoulder'))) {
    codes.push('M75.51'); // Impingement syndrome
  }

  if (areas.some(a => a.includes('hip') || a.includes('lumbar'))) {
    codes.push('M54.5'); // Low back pain
    codes.push('M54.16'); // Radiculopathy
  }

  if (areas.some(a => a.includes('knee'))) {
    codes.push('M25.561'); // Pain in right knee
    codes.push('M25.562'); // Pain in left knee
  }

  if (areas.some(a => a.includes('ankle') || a.includes('foot'))) {
    codes.push('M25.571'); // Pain in right ankle
    codes.push('M25.572'); // Pain in left ankle
  }

  if (areas.some(a => a.includes('core') || a.includes('stability'))) {
    codes.push('R26.2'); // Difficulty walking
  }

  if (codes.length === 0) {
    codes.push('M54.5'); // Default: Low back pain
  }

  return [...new Set(codes)];
}

function generateAdditionalNotes(
  session: LiveTrackingSession,
  analysis: BiomechanicalAnalysis
): string {
  const notes: string[] = [];

  // Camera used
  notes.push(`Assessment conducted using ${session.cameraType} camera.`);

  // Technology note
  if (session.cameraType === 'femto_mega') {
    notes.push('3D depth tracking utilized for enhanced accuracy.');
  } else {
    notes.push('2D video analysis performed. Consider 3D assessment for complex cases.');
  }

  // Recommendations
  if (analysis.movement_quality_score < 60) {
    notes.push('Recommend comprehensive evaluation and possible imaging referral.');
  }

  return notes.join(' ');
}

// Format note for printing/export
export function formatMedicalNoteForExport(note: MedicalNoteOutput): string {
  return `
================================================================================
                         PHYSICAL THERAPY EVALUATION
================================================================================

SUBJECTIVE:
${note.subjective}

--------------------------------------------------------------------------------
OBJECTIVE:
${note.objective}

--------------------------------------------------------------------------------
ASSESSMENT:
${note.assessment}

--------------------------------------------------------------------------------
PLAN:
${note.plan}

--------------------------------------------------------------------------------
BILLING CODES:
CPT: ${note.cptCodes.join(', ')}
ICD-10: ${note.icd10Codes.join(', ')}

--------------------------------------------------------------------------------
ADDITIONAL NOTES:
${note.additionalNotes}

================================================================================
Generated: ${new Date().toLocaleString()}
================================================================================
`;
}
