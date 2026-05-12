import { Hono } from 'hono';
import { cors } from 'hono/cors';

const OLLAMA_BASE = process.env.OLLAMA_HOST || 'http://localhost:11434';

// ─── Multi-Agent Specialist Swarm for Boxer3D ───
// Each specialist analyzes the same pose data from their expertise
const SPECIALISTS = [
  {
    id: 'upper-extremity',
    name: 'Upper Extremity & Spine Specialist',
    persona: 'orthopedic surgeon specializing in shoulder, elbow, wrist, hand, and cervical spine',
    focus: 'Shoulder impingement, rotator cuff, elbow epicondylitis, cervical radiculopathy, postural analysis',
    landmarks: [0,1,2,5,6,7,8,9,10],
  },
  {
    id: 'lower-extremity',
    name: 'Lower Extremity & Gait Specialist',
    persona: 'sports medicine physician specializing in hip, knee, ankle, foot, and gait analysis',
    focus: 'Hip impingement, patellofemoral pain, ACL stability, gait cycle, ankle sprains, foot mechanics',
    landmarks: [11,12,13,14,15,16],
  },
  {
    id: 'spine-core',
    name: 'Spine, Core & Posture Specialist',
    persona: 'chiropractic and rehabilitation physician specializing in spinal mechanics, core stability, and pelvic alignment',
    focus: 'Lumbar/pelvic mechanics, core stability, thoracic mobility, scoliosis screening, movement compensation',
    landmarks: [5,6,11,12,13,14],
  },
  {
    id: 'auditor',
    name: 'Clinical Auditor & Scribe',
    persona: 'senior physiatrist reviewing all specialist findings and producing the final AIMS-style SOAP note',
    focus: 'Reconcile conflicting findings, validate diagnoses, produce structured SOAP note with ICD-10/CPT',
    landmarks: [],
  },
];

async function ollamaChat(model: string, prompt: string): Promise<string> {
  const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, prompt, stream: false, keep_alive: '5m', temperature: 0.2 }),
  });
  if (!res.ok) throw new Error(`Ollama error: ${res.status}`);
  const data = await res.json() as any;
  return data.response || '';
}

// ─── ROM Norms (embedded) ───
const ROM = {
  shoulder: { flexion: [150,180], extension: [45,60], abduction: [150,180], ext_rotation: [80,90], int_rotation: [60,70] },
  elbow: { flexion: [130,150], extension: [0,10] },
  hip: { flexion: [110,130], extension: [10,30], abduction: [30,50] },
  knee: { flexion: [120,150], extension: [0,10] },
  ankle: { dorsiflexion: [10,20], plantarflexion: [30,50] },
  cervical: { flexion: [40,60], extension: [40,75], rotation: [60,80], lateral_flexion: [30,45] },
  lumbar: { flexion: [40,60], extension: [15,30], lateral_flexion: [15,25] },
};

function buildROMTable(regions: string[]): string {
  const map: Record<string, string[]> = {
    upper: ['shoulder', 'elbow'],
    lower: ['hip', 'knee', 'ankle'],
    spine: ['cervical', 'lumbar'],
  };
  let out = '### ROM Normative Data:\n';
  for (const r of regions) {
    const joints = map[r] || [];
    for (const j of joints) {
      const norms = (ROM as any)[j];
      if (!norms) continue;
      out += `\n**${j.toUpperCase()}**\n`;
      for (const [m, [mn, mx]] of Object.entries(norms as Record<string, [number, number]>)) {
        out += `  ${m}: ${mn}°-${mx}°
      }
    }
  }
  return out;
}

function buildSpecialistPrompt(specialist: any, keypoints: any[], regions: string[]): string {
  const kpSummary = keypoints
    .filter(k => specialist.landmarks.includes(k.id))
    .map(k => `  ${k.id}: (${(k.x*100).toFixed(0)}%, ${(k.y*100).toFixed(0)}%) conf=${k.confidence} depth=${k.z.toFixed(2)}m`)
    .join('\n');

  return `You are a ${specialist.persona}. Analyze these 3D pose keypoints from your specialty perspective.

Focus area: ${specialist.focus}

${buildROMTable(regions)}

### 3D Keypoint Data (your landmarks):
${kpSummary || '  (No relevant landmarks in this frame)'}

Provide your assessment in this format:
FINDINGS: (what you observe from your specialty)
DIAGNOSIS: (possible diagnoses based on these findings)
RECOMMENDATIONS: (specific interventions)
CONFIDENCE: (0.0-1.0 based on data available for your specialty)`;
}

const aiRoutes = new Hono();
aiRoutes.use('/*', cors());

// Multi-agent analysis: route to relevant specialists then synthesize
aiRoutes.post('/analyze-swarm', async (c) => {
  try {
    const body = await c.req.json() as any;
    const { keypoints, abnormalities, body_regions } = body;
    if (!keypoints || keypoints.length === 0) {
      return c.json({ success: false, error: 'Keypoint data required' }, 400);
    }

    const regions = body_regions || ['upper', 'lower', 'spine'];
    const activeSpecialists = SPECIALISTS.filter(s => s.id !== 'auditor');

    console.log(`[SWARM] Analyzing with ${activeSpecialists.length} specialists for [${regions.join(', ')}]...`);

    // Run each specialist sequentially (single model, persona-switching)
    const results: any[] = [];
    for (const spec of activeSpecialists) {
      const prompt = buildSpecialistPrompt(spec, keypoints, regions);
      try {
        const resp = await ollamaChat('phi4-mini:3.8b', prompt);
        results.push({
          agent: spec.id,
          name: spec.name,
          raw: resp,
          findings: resp.match(/FINDINGS:?\s*(.+?)(?=\n|$)/i)?.[1]?.trim() || '',
          diagnosis: (resp.match(/DIAGNOSIS:?\s*([\s\S]*?)(?=\nRECOMMENDATIONS|$)/i)?.[1] || '').split('\n').filter((l: string) => l.trim()).map((l: string) => l.replace(/^[-*\s\d.]+/, '').trim()),
          recommendations: (resp.match(/RECOMMENDATIONS:?\s*([\s\S]*?)(?=\nCONFIDENCE|$)/i)?.[1] || '').split('\n').filter((l: string) => l.trim()).map((l: string) => l.replace(/^[-*\s\d.]+/, '').trim()),
          confidence: parseFloat(resp.match(/CONFIDENCE:?\s*([\d.]+)/i)?.[1] || '0.5'),
        });
      } catch (e: any) {
        results.push({ agent: spec.id, name: spec.name, error: e.message, confidence: 0 });
      }
    }

    // Auditor: synthesize all specialist reports
    const summaryData = results.map(r =>
      `[${r.name}] Findings: ${r.findings || 'N/A'} | Diagnosis: ${(r.diagnosis||[]).join('; ')} | Confidence: ${r.confidence}`
    ).join('\n');

    const auditorPrompt = `You are a senior physiatrist and clinical scribe. Synthesize the specialist reports below into an AIMS EHR-style SOAP note.

${buildROMTable(regions)}

### Specialist Reports:
${summaryData}

Output a structured SOAP note in this exact format — this will populate the AIMS EHR medical scribe:

SUBJECTIVE:
Chief Complaint: (patient's reported issue)
History of Present Illness: (onset, duration, characteristics)

OBJECTIVE:
Vitals: (N/A for pose assessment)
Physical Exam:
  Posture: (postural findings from spine specialist)
  Upper Extremities: (range of motion, strength, special tests)
  Lower Extremities: (range of motion, gait, special tests)
  Spine: (flexion, extension, rotation, tenderness)

ASSESSMENT:
Primary Diagnosis: (ICD-10 code + description)
Secondary Diagnoses: (additional codes)
Clinical Summary: (integrated assessment)

PLAN:
Recommendations: (prioritized treatment)
Exercises: (specific prescriptions from knowledge base)
Referrals: (if needed)
Follow-up: (timeline)

ICD-10: (comma-separated codes)
CPT: (comma-separated codes)
CONFIDENCE: (0.0-1.0)`;

    let auditorResp;
    try {
      auditorResp = await ollamaChat('phi4-mini:3.8b', auditorPrompt);
      auditorResp = auditorResp.replace(/\*\*/g, '');
    } catch (e: any) {
      auditorResp = 'Auditor synthesis failed';
    }

    // Parse SOAP sections
    const soapNote = {
      subjective: auditorResp.match(/SUBJECTIVE:?\s*([\s\S]*?)(?=\nOBJECTIVE:|$)/i)?.[1]?.trim() || '',
      objective: auditorResp.match(/OBJECTIVE:?\s*([\s\S]*?)(?=\nASSESSMENT:|$)/i)?.[1]?.trim() || '',
      assessment: auditorResp.match(/ASSESSMENT:?\s*([\s\S]*?)(?=\nPLAN:|$)/i)?.[1]?.trim() || '',
      plan: auditorResp.match(/PLAN:?\s*([\s\S]*?)(?=\nICD-10:|$)/i)?.[1]?.trim() || '',
      icd10: (auditorResp.match(/ICD-10:?\s*(.+?)(?=\n|$)/i)?.[1] || '').split(',').map((s: string) => s.trim()).filter(Boolean),
      cpt: (auditorResp.match(/CPT:?\s*(.+?)(?=\n|$)/i)?.[1] || '').split(',').map((s: string) => s.trim()).filter(Boolean),
      confidence: Math.min(1, Math.max(0, parseFloat(auditorResp.match(/CONFIDENCE:?\s*([\d.]+)/i)?.[1] || '0.5') / 10)),
    };

    return c.json({
      success: true,
      swarm: {
        specialists: results,
        auditor: soapNote,
        model: 'phi4-mini:3.8b (3-specialist swarm → AIMS SOAP)',
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[SWARM] Error:', error);
    return c.json({ success: false, error: 'Swarm analysis failed' }, 500);
  }
});

// Single-agent analysis (keep existing for backward compat)
aiRoutes.post('/analyze', async (c) => {
  try {
    const body = await c.req.json() as any;
    const assessmentData = body.assessmentData || body;
    if (!assessmentData) {
      return c.json({ success: false, error: 'Assessment data is required' }, 400);
    }

    const ragCtx = (() => {
      const abnormal: string[] = assessmentData?.abnormalities || [];
      const affected = new Set<string>();
      for (const a of abnormal) {
        for (const j of Object.keys(ROM)) {
          if (a.toLowerCase().includes(j)) affected.add(j);
        }
      }
      if (affected.size === 0 && assessmentData?.body_region) affected.add(assessmentData.body_region.toLowerCase());
      if (affected.size === 0) return '';
      let ctx = '## MSK Clinical Knowledge Base\n';
      for (const joint of affected) {
        const norms = (ROM as any)[joint];
        if (!norms) continue;
        ctx += `\n### ${joint.toUpperCase()} ROM Norms:\n`;
        for (const [movement, [mn, mx]] of Object.entries(norms as Record<string, [number, number]>)) {
          ctx += `  ${movement}: normal ${mn}°-${mx}°
        }
      }
      return ctx;
    })();

    const prompt = `You are a board-certified physical medicine specialist.

${ragCtx}

### Patient:
${JSON.stringify(assessmentData, null, 2)}

Provide:
SUMMARY: (clinical summary)
FINDINGS: (bullet list)
DIFFERENTIAL: (possible diagnoses)
RECOMMENDATIONS: (treatment)
ICD10: (codes)
CPT: (codes)
CONFIDENCE: (0.0-1.0)`;

    console.log('[MSK-AI] Analyzing...');
    let response;
    try {
      response = await ollamaChat('phi4-mini:3.8b', prompt);
    } catch (e) {
      console.log('[MSK-AI] Fallback to deepseek-r1:7b');
      response = await ollamaChat('deepseek-r1:7b', prompt);
      response = response.replace(/[\s\S]*?<\/?think>/g, '').trim();
    }
    response = response.replace(/\*\*/g, '');

    const extract = (label: string): string[] => {
      const parts = response.split(new RegExp(`(?:###\\n?\\s*)?${label}:?`, 'i'));
      if (parts.length < 2) return [];
      const content = parts[1].split(/\n(?:###\n?\s*)?(?:FINDINGS|DIFFERENTIAL|RECOMMENDATIONS|ICD10|CPT|CONFIDENCE):/i)[0];
      return content.split('\n').map(l => l.replace(/^[-*\s\d.]+/, '').trim()).filter((l: string) => l.length > 3);
    };

    return c.json({
      success: true,
      analysis: {
        summary: response.match(/SUMMARY:?\s*(.+?)(?=\n|$)/i)?.[1]?.trim().replace(/[*#]/g, '') || '',
        findings: extract('FINDINGS'),
        differential: extract('DIFFERENTIAL'),
        recommendations: extract('RECOMMENDATIONS'),
        icd10: extract('ICD10'),
        cpt: extract('CPT'),
        confidence: Math.min(1, Math.max(0, parseFloat(response.match(/CONFIDENCE:?\s*([\d.]+)/i)?.[1] || '0.5'))),
        model: 'phi4-mini:3.8b (RAG enhanced)',
      },
    });
  } catch (error: any) {
    console.error('[MSK-AI] Error:', error);
    return c.json({ success: false, error: 'Analysis failed' }, 500);
  }
});

aiRoutes.get('/info', (c) => {
  return c.json({
    success: true,
    model: {
      name: 'phi4-mini:3.8b / deepseek-r1:7b',
      description: 'Boxer3D multi-agent MSK assessment',
      specialists: SPECIALISTS.map(s => ({ id: s.id, name: s.name, persona: s.persona })),
    },
  });
});

export default aiRoutes;
