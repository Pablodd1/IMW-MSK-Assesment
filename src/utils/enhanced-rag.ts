// Enhanced RAG System with CPT, ICD-10, Clinical Guidelines
// Medical-Grade Knowledge Base

export interface EnhancedRAGResult {
  answer: string;
  sources: string[];
  confidence: number;
  cptCodes?: CPTCode[];
  icd10Codes?: ICD10Code[];
  guidelines?: ClinicalGuideline[];
  recommendations?: string[];
}

export interface CPTCode {
  cpt_code: string;
  code_description: string;
  category: string;
  medicare_2025_rate?: number;
  documentation_requirements?: string;
}

export interface ICD10Code {
  icd10_code: string;
  code_description: string;
  category: string;
  body_region?: string;
  typical_cpt_codes?: string;
}

export interface ClinicalGuideline {
  title: string;
  content: string;
  source: string;
  evidence_level: string;
}

export async function enhancedRAGQuery(
  db: any,
  query: string,
  context: {
    patientId?: number;
    deficiencies?: string[];
    bodyRegions?: string[];
    intent?: 'billing' | 'clinical' | 'diagnostic' | 'treatment';
  }
): Promise<EnhancedRAGResult> {
  const keywords = query.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2);

  const results: EnhancedRAGResult = {
    answer: '',
    sources: [],
    confidence: 0,
    cptCodes: [],
    icd10Codes: [],
    guidelines: [],
    recommendations: []
  };

  // 1. Search CPT codes if intent is billing
  if (context.intent === 'billing' || !context.intent) {
    const cptResults = await searchCPTCodes(db, keywords);
    if (cptResults.length > 0) {
      results.cptCodes = cptResults;
      results.sources.push(...cptResults.map(c => c.cpt_code));
    }
  }

  // 2. Search ICD-10 codes if intent is diagnostic
  if (context.intent === 'diagnostic' || !context.intent) {
    const icdResults = await searchICD10Codes(db, keywords, context.bodyRegions);
    if (icdResults.length > 0) {
      results.icd10Codes = icdResults;
      results.sources.push(...icdResults.map(i => i.icd10_code));
    }
  }

  // 3. Search clinical guidelines
  const guidelineResults = await searchGuidelines(db, keywords, context.bodyRegions);
  if (guidelineResults.length > 0) {
    results.guidelines = guidelineResults;
    results.sources.push(...guidelineResults.map(g => g.title));
  }

  // 4. Generate recommendations based on deficiencies
  if (context.deficiencies && context.deficiencies.length > 0) {
    results.recommendations = await generateRecommendations(db, context.deficiencies);
  }

  // 5. Build answer
  results.answer = buildAnswer(query, results, context);
  results.confidence = calculateConfidence(results);

  return results;
}

async function searchCPTCodes(db: any, keywords: string[]): Promise<CPTCode[]> {
  if (keywords.length === 0) return [];

  const conditions = keywords
    .slice(0, 5)
    .map(() => `(cpt_code LIKE ? OR code_description LIKE ? OR commonly_used_for LIKE ?)`)
    .join(' OR ');
  
  const params = keywords
    .slice(0, 5)
    .flatMap(k => [`%${k}%`, `%${k}%`, `%${k}%`]);

  try {
    const { results } = await db.prepare(`
      SELECT cpt_code, code_description, code_category, medicare_2025_rate, 
             documentation_requirements, commonly_used_for
      FROM cpt_codes 
      WHERE ${conditions}
      ORDER BY medicare_2025_rate DESC
      LIMIT 10
    `).bind(...params).all();

    return results || [];
  } catch (e) {
    console.error('CPT search error:', e);
    return [];
  }
}

async function searchICD10Codes(
  db: any, 
  keywords: string[], 
  bodyRegions?: string[]
): Promise<ICD10Code[]> {
  if (keywords.length === 0) return [];

  let conditions = keywords
    .slice(0, 5)
    .map(() => `(icd10_code LIKE ? OR code_description LIKE ? OR commonly_used_for LIKE ?)`)
    .join(' OR ');
  
  const params = keywords
    .slice(0, 5)
    .flatMap(k => [`%${k}%`, `%${k}%`, `%${k}%`]);

  if (bodyRegions && bodyRegions.length > 0) {
    conditions += ` AND (${bodyRegions.map(() => 'body_region = ?').join(' OR ')})`;
    params.push(...bodyRegions);
  }

  try {
    const { results } = await db.prepare(`
      SELECT icd10_code, code_description, code_category, body_region, 
             typical_cpt_codes, commonly_used_for
      FROM icd10_codes 
      WHERE ${conditions}
      LIMIT 15
    `).bind(...params).all();

    return results || [];
  } catch (e) {
    console.error('ICD10 search error:', e);
    return [];
  }
}

async function searchGuidelines(
  db: any, 
  keywords: string[],
  bodyRegions?: string[]
): Promise<ClinicalGuideline[]> {
  if (keywords.length === 0) return [];

  let conditions = keywords
    .slice(0, 5)
    .map(() => `(guideline_title LIKE ? OR content LIKE ? OR topic LIKE ?)`)
    .join(' OR ');
  
  const params = keywords
    .slice(0, 5)
    .flatMap(k => [`%${k}%`, `%${k}%`, `%${k}%`]);

  if (bodyRegions && bodyRegions.length > 0) {
    conditions += ` AND (${bodyRegions.map(() => 'body_region = ?').join(' OR ')})`;
    params.push(...bodyRegions);
  }

  try {
    const { results } = await db.prepare(`
      SELECT guideline_title, content, source, evidence_level, topic, body_region
      FROM clinical_guidelines 
      WHERE ${conditions}
      LIMIT 5
    `).bind(...params).all();

    return results || [];
  } catch (e) {
    console.error('Guidelines search error:', e);
    return [];
  }
}

async function generateRecommendations(
  db: any, 
  deficiencies: string[]
): Promise<string[]> {
  const recommendations: string[] = [];

  const deficiencyKeywords = {
    'Ankle Dorsiflexion': ['ankle mobility', 'gait', 'squat depth'],
    'Hip Flexion': ['hip mobility', 'lumbar spine', 'squat'],
    'Shoulder Flexion': ['shoulder ROM', 'overhead reach'],
    'Core Stability': ['core', 'lumbar', 'trunk control'],
    'Bilateral Asymmetry': ['balance', 'unilateral', 'stability'],
    'Knee Valgus': ['hip abduction', 'gluteal', 'valgus control']
  };

  for (const deficiency of deficiencies) {
    const keywords = deficiencyKeywords[deficiency] || [deficiency.toLowerCase()];
    
    const { results } = await db.prepare(`
      SELECT guideline_title, recommendations
      FROM clinical_guidelines 
      WHERE topic IN (${keywords.map(() => '?').join(',')})
         OR guideline_title LIKE ?
         OR body_region LIKE ?
      LIMIT 3
    `).bind(...keywords, ...keywords.map(k => `%${k}%`), ...keywords.map(k => `%${k}%`)).all();

    if (results && results.length > 0) {
      for (const r of results) {
        if (r.recommendations) {
          recommendations.push(r.recommendations);
        }
      }
    }
  }

  return recommendations.slice(0, 5);
}

function buildAnswer(
  query: string, 
  results: EnhancedRAGResult,
  context: any
): string {
  const parts: string[] = [];

  // Add CPT codes if available
  if (results.cptCodes && results.cptCodes.length > 0) {
    const cptList = results.cptCodes
      .slice(0, 5)
      .map(c => `${c.cpt_code}: ${c.code_description}`)
      .join('; ');
    parts.push(`**Recommended CPT Codes:** ${cptList}`);
  }

  // Add ICD-10 codes if available  
  if (results.icd10Codes && results.icd10Codes.length > 0) {
    const icdList = results.icd10Codes
      .slice(0, 5)
      .map(i => `${i.icd10_code}: ${i.code_description}`)
      .join('; ');
    parts.push(`**Diagnosis Codes:** ${icdList}`);
  }

  // Add guidelines if available
  if (results.guidelines && results.guidelines.length > 0) {
    const guideline = results.guidelines[0];
    parts.push(`**Clinical Guideline:** ${guideline.title} - ${guideline.content.substring(0, 200)}...`);
  }

  // Add recommendations
  if (results.recommendations && results.recommendations.length > 0) {
    parts.push(`**Recommendations:** ${results.recommendations.join('; ')}`);
  }

  if (parts.length === 0) {
    return `Based on your query about "${query}", I couldn't find specific clinical guidance. Please provide more details about the patient condition or consult clinical resources.`;
  }

  return parts.join('\n\n');
}

function calculateConfidence(results: EnhancedRAGResult): number {
  let score = 0;
  let maxScore = 0;

  if (results.cptCodes) {
    maxScore += 0.3;
    score += Math.min(0.3, results.cptCodes.length * 0.05);
  }

  if (results.icd10Codes) {
    maxScore += 0.3;
    score += Math.min(0.3, results.icd10Codes.length * 0.05);
  }

  if (results.guidelines) {
    maxScore += 0.25;
    score += Math.min(0.25, results.guidelines.length * 0.08);
  }

  if (results.recommendations) {
    maxScore += 0.15;
    score += Math.min(0.15, results.recommendations.length * 0.03);
  }

  return maxScore > 0 ? Math.round((score / maxScore) * 100) : 0;
}

// Get suggested CPT codes based on assessment type
export async function suggestBillingCodes(
  db: any,
  assessmentType: string,
  deficiencies: string[]
): Promise<CPTCode[]> {
  const codes: CPTCode[] = [];

  // Always include evaluation code
  const evalCode = assessmentType === 'initial' ? '97163' : '97164';
  const { results: evalResults } = await db.prepare(`
    SELECT cpt_code, code_description, code_category, medicare_2025_rate
    FROM cpt_codes WHERE cpt_code = ?
  `).bind(evalCode).all();
  
  if (evalResults?.[0]) codes.push(evalResults[0]);

  // Add therapeutic codes based on deficiencies
  const { results: therapeuticResults } = await db.prepare(`
    SELECT cpt_code, code_description, code_category, medicare_2025_rate
    FROM cpt_codes 
    WHERE code_category = 'therapeutic' 
    ORDER BY medicare_2025_rate DESC
    LIMIT 3
  `).all();
  
  if (therapeuticResults) codes.push(...therapeuticResults.slice(0, 3));

  return codes;
}

// Get ICD-10 codes for body region
export async function getICD10ForRegion(
  db: any,
  bodyRegion: string
): Promise<ICD10Code[]> {
  const { results } = await db.prepare(`
    SELECT icd10_code, code_description, code_category, typical_cpt_codes
    FROM icd10_codes 
    WHERE body_region = ?
    ORDER BY code_description
    LIMIT 10
  `).bind(bodyRegion).all();

  return results || [];
}
