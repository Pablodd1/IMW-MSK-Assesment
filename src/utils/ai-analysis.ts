/**
 * AI Analysis Module — Gemini 2.0 Flash
 * 
 * Provider choice: Gemini 2.0 Flash
 * - Free tier: 1,500 requests/day
 * - $0.10/M input tokens after free tier
 * - Excellent structured JSON output
 * - Medical knowledge built into training
 * - Key already in Vercel (GEMINI_API_KEY)
 * 
 * Cheaper alternative available: DeepSeek ($0.27/M input)
 * Set DEEPSEEK_API_KEY and change provider to 'deepseek'
 */

// ── Types ──

export interface PoseLandmark {
  x: number; y: number; z: number; visibility: number;
}

export interface MovementAssessment {
  patientId: number;
  movementTest: string;       // 'deep_squat', 'shoulder_mobility', 'gait', etc.
  landmarks: PoseLandmark[];  // MediaPipe 33-point landmarks
  bodyRegion: string;         // 'shoulder', 'knee', 'hip', 'spine', 'ankle'
  frameCount?: number;
  depthData?: boolean;        // Femto Mega depth available
}

export interface AIAnalysisResult {
  // Joint angles
  jointAngles: Record<string, { measured: number; normative: string; deviation: string }>;
  // Asymmetries
  asymmetries: { joint: string; left: number; right: number; difference: number; significance: string }[];
  // Movement quality
  movementQualityScore: number;  // 0-100
  compensationDetected: boolean;
  compensations: string[];
  // Clinical
  suspectedConditions: { name: string; icd10: string; confidence: 'high' | 'moderate' | 'low'; reasoning: string }[];
  recommendedTests: { name: string; rationale: string }[];
  // Scores (if applicable)
  clinicalScores?: Record<string, number>;
  // SOAP
  soapNote: {
    subjective: string;
    objective: string;
    assessment: string;
    plan: string;
  };
  // Billing
  suggestedCptCodes: { code: string; description: string; units: number }[];
  // Recommendations
  rehabProtocol?: { name: string; currentPhase: string; exercises: string[] };
  redFlags: string[];
  nextSteps: string[];
}

// ── Provider Selection ──

type AIProvider = 'gemini' | 'deepseek';

function getProvider(): AIProvider {
  if (process.env.DEEPSEEK_API_KEY) return 'deepseek';
  if (process.env.GEMINI_API_KEY) return 'gemini';
  return 'gemini'; // default
}

// ── Gemini Client ──

async function callGemini(prompt: string, systemInstruction: string): Promise<string> {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error('GEMINI_API_KEY not set');

  const model = 'gemini-2.0-flash'; // cheapest capable model
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;

  const body = {
    system_instruction: { parts: [{ text: systemInstruction }] },
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: {
      temperature: 0.3,
      maxOutputTokens: 2048,
      responseMimeType: 'application/json',
    },
  };

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini API error ${res.status}: ${err.slice(0, 300)}`);
  }

  const data = await res.json() as any;
  return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

// ── DeepSeek Client (fallback/alternative) ──

async function callDeepSeek(prompt: string, systemInstruction: string): Promise<string> {
  const key = process.env.DEEPSEEK_API_KEY;
  if (!key) throw new Error('DEEPSEEK_API_KEY not set');

  const res = await fetch('https://api.deepseek.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${key}`,
    },
    body: JSON.stringify({
      model: 'deepseek-chat',
      messages: [
        { role: 'system', content: systemInstruction },
        { role: 'user', content: prompt },
      ],
      temperature: 0.3,
      max_tokens: 2048,
      response_format: { type: 'json_object' },
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`DeepSeek API error ${res.status}: ${err.slice(0, 300)}`);
  }

  const data = await res.json() as any;
  return data.choices?.[0]?.message?.content || '';
}

// ── RAG Context Builder ──

import { searchKnowledge, ROM_NORMATIVE, MEDICAL_CONDITIONS, CLINICAL_CALCULATORS } from './rag-knowledge.js';

function buildRAGContext(assessment: MovementAssessment): string {
  const context: string[] = [];

  // Search knowledge base for relevant conditions
  const knowledge = searchKnowledge(assessment.bodyRegion + ' ' + assessment.movementTest);
  if (knowledge.length > 0) {
    context.push('### Relevant Clinical Knowledge');
    context.push(...knowledge);
  }

  // Add ROM norms for the body region
  context.push('\n### Normative ROM Data');
  for (const [joint, range] of Object.entries(ROM_NORMATIVE)) {
    if (joint.includes(assessment.bodyRegion) || joint.includes(assessment.movementTest.split('_')[0])) {
      context.push(`${joint.replace(/_/g, ' ')}: ${range.min}-${range.max}${range.unit}`);
    }
  }

  // Add relevant calculators
  context.push('\n### Available Clinical Calculators');
  for (const [key, calc] of Object.entries(CLINICAL_CALCULATORS)) {
    if (key.toLowerCase().includes(assessment.bodyRegion) || key === 'FMS') {
      context.push(`${calc.name}: ${calc.formula}`);
    }
  }

  return context.join('\n');
}

// ── Pose Landmark Processing ──

function extractJointAngles(landmarks: PoseLandmark[]): Record<string, { left?: number; right?: number }> {
  // MediaPipe pose indices
  const L_SHOULDER = 11, R_SHOULDER = 12;
  const L_ELBOW = 13, R_ELBOW = 14;
  const L_WRIST = 15, R_WRIST = 16;
  const L_HIP = 23, R_HIP = 24;
  const L_KNEE = 25, R_KNEE = 26;
  const L_ANKLE = 27, R_ANKLE = 28;

  const angle = (a: PoseLandmark, b: PoseLandmark, c: PoseLandmark): number => {
    const ba = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
    const bc = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
    const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
    const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
    const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);
    if (magBA === 0 || magBC === 0) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / (magBA * magBC)))) * (180 / Math.PI);
  };

  const has = (i: number) => landmarks[i] && landmarks[i].visibility > 0.5;

  return {
    shoulder_flexion: has(L_SHOULDER) && has(L_ELBOW) && has(L_HIP)
      ? { left: angle(landmarks[L_HIP], landmarks[L_SHOULDER], landmarks[L_ELBOW]), right: has(R_SHOULDER) && has(R_ELBOW) && has(R_HIP) ? angle(landmarks[R_HIP], landmarks[R_SHOULDER], landmarks[R_ELBOW]) : undefined } : {},
    elbow_flexion: has(L_SHOULDER) && has(L_ELBOW) && has(L_WRIST)
      ? { left: angle(landmarks[L_SHOULDER], landmarks[L_ELBOW], landmarks[L_WRIST]), right: has(R_SHOULDER) && has(R_ELBOW) && has(R_WRIST) ? angle(landmarks[R_SHOULDER], landmarks[R_ELBOW], landmarks[R_WRIST]) : undefined } : {},
    hip_flexion: has(L_SHOULDER) && has(L_HIP) && has(L_KNEE)
      ? { left: angle(landmarks[L_SHOULDER], landmarks[L_HIP], landmarks[L_KNEE]), right: has(R_SHOULDER) && has(R_HIP) && has(R_KNEE) ? angle(landmarks[R_SHOULDER], landmarks[R_HIP], landmarks[R_KNEE]) : undefined } : {},
    knee_flexion: has(L_HIP) && has(L_KNEE) && has(L_ANKLE)
      ? { left: angle(landmarks[L_HIP], landmarks[L_KNEE], landmarks[L_ANKLE]), right: has(R_HIP) && has(R_KNEE) && has(R_ANKLE) ? angle(landmarks[R_HIP], landmarks[R_KNEE], landmarks[R_ANKLE]) : undefined } : {},
  };
}

// ── Main Analysis Pipeline ──

const SYSTEM_INSTRUCTION = `You are a board-certified physical medicine and rehabilitation (PM&R) physician 
specializing in musculoskeletal assessment. You analyze 3D motion capture data and produce 
structured clinical assessments.

CRITICAL RULES:
1. Always output valid JSON exactly matching the requested schema
2. Base findings on the provided data — do not fabricate measurements
3. When data is insufficient for a finding, note it as "Insufficient data" rather than guessing
4. Include ICD-10 codes for all suspected conditions
5. Flag RED FLAGS prominently when identified
6. Suggest CPT codes based on the 8-minute rule (1 unit = 15 min)
7. Use sensitivity/specificity data when available
8. Be conservative — when in doubt, recommend in-person evaluation

MEDICAL ACCURACY REQUIREMENTS:
- Joint angle measurements: cite the measured value, normative range, and deviation
- Asymmetry: flag any left-right difference >10° or >15% for major joints
- Movement quality: score based on ROM achievement, symmetry, compensation patterns
- Conditions: list by confidence (high > moderate > low) with clinical reasoning
- SOAP note: use proper medical terminology, include measurable objective data`;

export async function analyzeMovement(assessment: MovementAssessment): Promise<AIAnalysisResult> {
  const provider = getProvider();
  const ragContext = buildRAGContext(assessment);
  const jointAngles = extractJointAngles(assessment.landmarks);

  const prompt = `Analyze the following movement assessment data and produce a structured clinical report.

PATIENT CONTEXT:
- Patient ID: ${assessment.patientId}
- Movement Test: ${assessment.movementTest}
- Body Region: ${assessment.bodyRegion}
- Depth Data Available: ${assessment.depthData ? 'Yes (Femto Mega 3D)' : 'No (2D only)'}
- Frame Count: ${assessment.frameCount || 'Not specified'}

MEASURED JOINT ANGLES (from pose landmarks):
${JSON.stringify(jointAngles, null, 2)}

CLINICAL KNOWLEDGE BASE:
${ragContext}

Produce a JSON response with this EXACT structure:
{
  "jointAngles": { "joint_name": { "measured": number, "normative": "min-max°", "deviation": "WNL | X° deficit | X° excess" } },
  "asymmetries": [{ "joint": "string", "left": number, "right": number, "difference": number, "significance": "none | mild | moderate | significant" }],
  "movementQualityScore": number (0-100),
  "compensationDetected": boolean,
  "compensations": ["string"],
  "suspectedConditions": [{ "name": "string", "icd10": "string", "confidence": "high|moderate|low", "reasoning": "string" }],
  "recommendedTests": [{ "name": "string", "rationale": "string" }],
  "soapNote": { "subjective": "string", "objective": "string", "assessment": "string", "plan": "string" },
  "suggestedCptCodes": [{ "code": "string", "description": "string", "units": number }],
  "redFlags": ["string"],
  "nextSteps": ["string"]
}`;

  try {
    let rawResponse: string;
    if (provider === 'deepseek') {
      rawResponse = await callDeepSeek(prompt, SYSTEM_INSTRUCTION);
    } else {
      rawResponse = await callGemini(prompt, SYSTEM_INSTRUCTION);
    }

    // Clean and parse
    const cleaned = rawResponse.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    const result: AIAnalysisResult = JSON.parse(cleaned);

    // Enrich with knowledge base data
    if (!result.rehabProtocol) {
      // Try to match a rehab protocol
      for (const condition of result.suspectedConditions || []) {
        const { getRehabProtocol } = await import('./rag-knowledge.js');
        const proto = getRehabProtocol(condition.name);
        if (proto) {
          result.rehabProtocol = {
            name: condition.name,
            currentPhase: proto.phases[0].name,
            exercises: proto.phases[0].exercises,
          };
          break;
        }
      }
    }

    return result;
  } catch (err: any) {
    console.error(`[AI] Analysis failed (${provider}):`, err.message);

    // Return a basic fallback analysis
    return {
      jointAngles: {},
      asymmetries: [],
      movementQualityScore: 0,
      compensationDetected: false,
      compensations: [],
      suspectedConditions: [],
      recommendedTests: [],
      soapNote: {
        subjective: 'AI analysis unavailable — manual assessment required.',
        objective: `Provider: ${provider} returned error: ${err.message.slice(0, 200)}`,
        assessment: 'Unable to complete automated analysis.',
        plan: 'Recommend in-person evaluation by licensed PT/MD.',
      },
      suggestedCptCodes: [{ code: '97161', description: 'PT evaluation — low complexity', units: 1 }],
      redFlags: ['AI analysis failed — requires manual review'],
      nextSteps: ['Schedule in-person evaluation', 'Verify camera setup and recalibrate'],
    };
  }
}

// ── Quick Assessment (no landmarks, just clinical data) ──

export async function quickAssessment(params: {
  patientId: number;
  bodyRegion: string;
  chiefComplaint: string;
  painScale: number;
  functionalLimitations: string;
}): Promise<Pick<AIAnalysisResult, 'suspectedConditions' | 'recommendedTests' | 'soapNote' | 'suggestedCptCodes' | 'redFlags'>> {
  const provider = getProvider();
  const knowledge = searchKnowledge(params.bodyRegion + ' ' + params.chiefComplaint);

  const prompt = `A patient presents with the following. Suggest differential diagnoses and a plan.

Patient ID: ${params.patientId}
Body Region: ${params.bodyRegion}
Chief Complaint: ${params.chiefComplaint}
Pain: ${params.painScale}/10
Functional Limitations: ${params.functionalLimitations}

CLINICAL KNOWLEDGE:
${knowledge.join('\n')}

Respond with JSON:
{
  "suspectedConditions": [{ "name": "string", "icd10": "string", "confidence": "high|moderate|low", "reasoning": "string" }],
  "recommendedTests": [{ "name": "string", "rationale": "string" }],
  "soapNote": { "subjective": "string", "objective": "string", "assessment": "string", "plan": "string" },
  "suggestedCptCodes": [{ "code": "string", "description": "string", "units": number }],
  "redFlags": ["string"]
}`;

  try {
    let raw: string;
    if (provider === 'deepseek') {
      raw = await callDeepSeek(prompt, SYSTEM_INSTRUCTION);
    } else {
      raw = await callGemini(prompt, SYSTEM_INSTRUCTION);
    }
    const cleaned = raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    return JSON.parse(cleaned);
  } catch (err: any) {
    return {
      suspectedConditions: [],
      recommendedTests: [],
      soapNote: {
        subjective: params.chiefComplaint,
        objective: `Pain ${params.painScale}/10. ${params.functionalLimitations}`,
        assessment: 'Requires in-person evaluation.',
        plan: 'Schedule PT evaluation.',
      },
      suggestedCptCodes: [{ code: '97161', description: 'PT evaluation', units: 1 }],
      redFlags: [],
    };
  }
}

// ── Provider Info ──

export function getAIInfo() {
  const provider = getProvider();
  return {
    provider,
    model: provider === 'gemini' ? 'gemini-2.0-flash' : 'deepseek-chat',
    cost: provider === 'gemini' ? '$0.10/M input (free tier: 1,500 req/day)' : '$0.27/M input',
    features: ['Structured JSON output', 'ICD-10/CPT coding', 'SOAP note generation', 'Clinical reasoning'],
  };
}
