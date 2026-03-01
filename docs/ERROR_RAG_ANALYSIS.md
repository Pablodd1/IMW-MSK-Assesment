# API Health & RAG Enhancement Report

---

## 1. ERROR ANALYSIS

### 1.1 Current 404 Errors (Not Found)

| Endpoint | Issue | Fix |
|----------|-------|-----|
| `/api/auth/profile/:id` | Returns raw error message | Return structured JSON |
| `/api/patients/:id` | Patient not found | Already returns 404 with JSON |
| `/api/patients/:id/medical-history` | Medical history not found | Missing endpoint check |
| `/api/assessments/:id` | Assessment not found | Already returns 404 |
| `/api/tests/:id/results` | Test not found | Already returns 404 |
| `/api/assessments/:id/tests` | Empty array when no tests | Not an error, works correctly |
| `/api/patients/:id/prescriptions` | Empty when no prescriptions | Not an error |

### 1.2 Current 500 Errors (Server Errors)

**Root Causes Identified:**

1. **Database query failures** - No graceful handling when tables missing
2. **Missing foreign key references** - JOINs fail if related records don't exist
3. **JSON parsing errors** - `JSON.parse()` on null/undefined
4. **Undefined property access** - Accessing `.meta.last_row_id` when insert fails

**Problematic Code Patterns:**

```typescript
// Problem: Direct error.message exposure (security risk)
} catch (error: any) {
  return c.json({ success: false, error: error.message }, 500)
}

// Problem: No null check on insert result
const result = await c.env.DB.prepare(`INSERT...`).bind(...).run()
return c.json({ success: true, data: { id: result.meta.last_row_id } })

// Problem: JSON.parse without try-catch
const deficiencies = JSON.parse(test.deficiencies)
```

---

## 2. RAG ENHANCEMENT OPPORTUNITIES

### 2.1 Current RAG System (Basic)

The current RAG (`src/utils/rag.ts`) only searches:
- Exercise names
- Exercise descriptions
- Exercise instructions

**Limitations:**
- Keyword-only matching (no semantic search)
- No clinical guidelines
- No CPT code knowledge
- No biomechanical data context
- No patient history integration

### 2.2 RAG Enhancement Proposal

```typescript
// Enhanced RAG with multiple knowledge bases

interface EnhancedRAGConfig {
  exerciseKnowledge: boolean
  clinicalGuidelines: boolean
  cptBilling: boolean
  patientHistory: boolean
  biomechanicalNorms: boolean
  contraindictions: boolean
}

interface RAGContext {
  patientId?: number
  assessmentId?: number
  clinicianId?: number
  currentDeficiencies?: string[]
  previousHistory?: any[]
}

// Expanded knowledge domains
const KNOWLEDGE_BASES = {
  // 1. Exercise Library (existing)
  exercises: {
    table: 'exercises',
    fields: ['name', 'description', 'instructions', 'contraindications']
  },
  
  // 2. Clinical Guidelines (NEW)
  clinical_guidelines: {
    table: 'clinical_guidelines',
    fields: ['guideline_title', 'content', 'source', 'specialty']
  },
  
  // 3. CPT Billing Codes (NEW)
  cpt_codes: {
    table: 'billing_codes',
    fields: ['cpt_code', 'code_description', 'requirements']
  },
  
  // 4. Medical Contraindications (NEW)
  contraindications: {
    table: 'contraindications',
    fields: ['condition', 'exercise_name', 'risk_level', 'reason']
  },
  
  // 5. Biomechanical Norms (NEW)
  normative_data: {
    table: 'normative_joint_angles',
    fields: ['joint', 'min_normal', 'max_normal', 'age_group', 'gender']
  },
  
  // 6. Provider Resources (NEW)
  provider_resources: {
    table: 'provider_resources',
    fields: ['title', 'content', 'category', 'url']
  }
}

// Enhanced query function
export async function enhancedQuery(
  db: any,
  query: string,
  context: RAGContext,
  config: EnhancedRAGConfig
): Promise<EnhancedRAGResult> {
  const results: RAGResult[] = []
  
  // Query each enabled knowledge base
  if (config.exerciseKnowledge) {
    results.push(await queryExercises(db, query))
  }
  if (config.clinicalGuidelines) {
    results.push(await queryGuidelines(db, query, context.specialty))
  }
  if (config.cptBilling) {
    results.push(await queryCPT(db, query))
  }
  if (config.contraindictions) {
    results.push(await queryContraindications(db, context.currentDeficiencies))
  }
  if (config.biomechanicalNorms) {
    results.push(await queryNorms(db, context.currentDeficiencies))
  }
  
  // Fuse results using reciprocal rank fusion
  return fuseResults(results)
}
```

### 2.3 Database Tables for Enhanced RAG

```sql
-- Clinical Guidelines Table
CREATE TABLE IF NOT EXISTS clinical_guidelines (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  guideline_title TEXT NOT NULL,
  specialty TEXT CHECK(specialty IN ('physical_therapy', 'chiropractic', 'sports_medicine')),
  content TEXT NOT NULL,
  source TEXT,
  evidence_level TEXT CHECK(evidence_level IN ('A', 'B', 'C', 'D')),
  last_updated DATE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Contraindications Table
CREATE TABLE IF NOT EXISTS contraindications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  condition TEXT NOT NULL,
  exercise_name TEXT NOT NULL,
  risk_level TEXT CHECK(risk_level IN ('low', 'moderate', 'high', 'contraindicated')),
  reason TEXT,
  alternative_exercise TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Normative Data Table
CREATE TABLE IF NOT EXISTS normative_joint_angles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  joint_name TEXT NOT NULL,
  movement TEXT NOT NULL,
  min_normal REAL NOT NULL,
  max_normal REAL NOT NULL,
  age_group TEXT CHECK(age_group IN ('18-30', '31-50', '51-70', '70+')),
  gender TEXT CHECK(gender IN ('male', 'female', 'neutral')),
  population_source TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Provider Resources Table
CREATE TABLE IF NOT EXISTS provider_resources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  category TEXT CHECK(category IN ('protocol', 'education', 'billing', 'compliance')),
  content TEXT NOT NULL,
  url TEXT,
  tags TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Seed data for clinical guidelines
INSERT INTO clinical_guidelines (guideline_title, specialty, content, source, evidence_level) VALUES
('FMS Scoring Guidelines', 'physical_therapy', 'Deep Squat: 3 - heels flat, knees past toes, torso parallel to tibia, dowel over feet. 2 - heels flat, knees past toes, torso above parallel. 1 - unable to perform with compensation.', 'Gray Cook FMS', 'A'),
('Hip Mobility Protocol', 'physical_therapy', 'Hip flexion deficit >15 degrees correlates with increased injury risk. Prescribe hip flexor stretching 2x daily, 30 second holds, 3 sets.', 'Journal of Orthopedic Sports Physical Therapy', 'A'),
('Knee Valgus Criteria', 'sports_medicine', 'Knee medial collapse >15 degrees during squatting tasks indicates elevated ACL injury risk. Progress to hip abduction strengthening.', 'American Journal of Sports Medicine', 'A');

-- Seed normative data
INSERT INTO normative_joint_angles (joint_name, movement, min_normal, max_normal, age_group, gender) VALUES
('knee', 'flexion', 130 '18-30, 150,', 'neutral'),
('hip', 'flexion', 110, 125, '18-30', 'neutral'),
('shoulder', 'flexion', 160, 180, '18-30', 'neutral'),
('ankle', 'dorsiflexion', 10, 25, '18-30', 'neutral'),
('knee', 'flexion', 120, 145, '51-70', 'neutral'),
('hip', 'flexion', 100, 120, '51-70', 'neutral');
```

---

## 3. PROVIDER ASSISTANCE IMPROVEMENTS

### 3.1 Current Provider Experience Gaps

| Area | Current State | Gap |
|------|--------------|-----|
| **Diagnosis Suggestions** | Basic deficiency list | No AI-powered differential diagnosis |
| **Treatment Recommendations** | Static exercise list | No personalized to patient history |
| **Billing Guidance** | CPT code lookup | No procedure-specific coding |
| **Clinical Decision Support** | None | No evidence-based recommendations |
| **Risk Alerts** | Basic compensation flags | No critical safety alerts |

### 3.2 Enhanced Provider Assistance System

```typescript
// Provider Assistance Types
interface ProviderAssistance {
  differentialDiagnosis: DiagnosisSuggestion[]
  treatmentRecommendations: TreatmentPlan
  billingGuidance: BillingSuggestion[]
  riskAlerts: RiskAlert[]
  clinicalAlerts: ClinicalAlert[]
  evidenceSupport: EvidenceItem[]
}

interface DiagnosisSuggestion {
  condition: string
  likelihood: number // 0-100
  supportingFindings: string[]
  recommendedTests: string[]
}

interface TreatmentPlan {
  exercises: PrescribedExercise[]
  frequency: string
  duration: string
  progressionCriteria: string
  warnings: string[]
}

interface BillingSuggestion {
  cptCode: string
  description: string
  requirements: string[]
  documentationTips: string[]
  rvus: number
}

interface RiskAlert {
  severity: 'low' | 'medium' | 'high' | 'critical'
  type: 'contraindication' | 'precaution' | 'fall_risk' | 'flag'
  message: string
  action: string
}

// Enhanced analysis with provider assistance
export async function generateProviderAssistance(
  db: any,
  patientId: number,
  analysisResults: BiomechanicalAnalysis
): Promise<ProviderAssistance> {
  const assistance: ProviderAssistance = {
    differentialDiagnosis: [],
    treatmentRecommendations: {} as TreatmentPlan,
    billingGuidance: [],
    riskAlerts: [],
    clinicalAlerts: [],
    evidenceSupport: []
  }
  
  // 1. Generate differential diagnoses based on deficiencies
  assistance.differentialDiagnosis = await generateDifferentialDiagnosis(
    db, analysisResults.deficiencies
  )
  
  // 2. Generate treatment plan
  assistance.treatmentRecommendations = await generateTreatmentPlan(
    db, analysisResults, patientId
  )
  
  // 3. Suggest billing codes
  assistance.billingGuidance = await suggestBillingCodes(
    db, analysisResults, patientId
  )
  
  // 4. Generate risk alerts
  assistance.riskAlerts = generateRiskAlerts(analysisResults)
  
  // 5. Get clinical alerts
  assistance.clinicalAlerts = await getClinicalAlerts(
    db, patientId, analysisResults
  )
  
  // 6. Get evidence support
  assistance.evidenceSupport = await getEvidenceSupport(
    db, analysisResults.deficiencies
  )
  
  return assistance
}

// Differential diagnosis generator
async function generateDifferentialDiagnosis(
  db: any, 
  deficiencies: Deficiency[]
): Promise<DiagnosisSuggestion[]> {
  const diagnoses: DiagnosisSuggestion[] = []
  
  // Map deficiencies to potential conditions
  const deficiencyConditionMap: Record<string, { condition: string, tests: string[] }> = {
    'Ankle Dorsiflexion': { 
      condition: 'Ankle Mobility Dysfunction',
      tests: ['Weight-bearing lunge test', 'Talar glide test']
    },
    'Hip Flexion': { 
      condition: 'Hip Impingement Risk',
      tests: ['FADIR test', 'Hip quadrant test']
    },
    'Shoulder Flexion': { 
      condition: 'Shoulder ROM Limitation',
      tests: ['AROM/PROM assessment', 'Neer impingement test']
    },
    'Bilateral Asymmetry': { 
      condition: 'Movement Pattern Dysfunction',
      tests: ['Single leg squat', 'Y-balance test']
    },
    'Core Stability': { 
      condition: 'Core Dysfunction',
      tests: ['Birthing ball test', 'Trunk stability test']
    }
  }
  
  for (const deficiency of deficiencies) {
    const mapping = deficiencyConditionMap[deficiency.area]
    if (mapping) {
      diagnoses.push({
        condition: mapping.condition,
        likelihood: deficiency.severity === 'severe' ? 85 : 
                   deficiency.severity === 'moderate' ? 65 : 45,
        supportingFindings: [deficiency.description],
        recommendedTests: mapping.tests
      })
    }
  }
  
  return diagnoses
}

// Billing code suggester
async function suggestBillingCodes(
  db: any,
  analysis: BiomechanicalAnalysis,
  patientId: number
): Promise<BillingSuggestion[]> {
  const suggestions: BillingSuggestion[] = []
  
  // Base evaluation code
  suggestions.push({
    cptCode: '97163',
    description: 'PT Evaluation - High Complexity',
    requirements: ['45 minutes face-to-face', 'Medical necessity documented'],
    documentationTips: ['Include all body regions assessed', 'Document clinical findings'],
    rvus: 3.5
  })
  
  // Add codes based on deficiencies
  if (analysis.deficiencies.some(d => d.area.includes('Mobility'))) {
    suggestions.push({
      cptCode: '97110',
      description: 'Therapeutic Exercise',
      requirements: ['15 minutes', 'Must be therapeutic'],
      documentationTips: ['Specify exercises performed', 'Note sets/reps'],
      rvus: 1.5
    })
  }
  
  if (analysis.deficiencies.some(d => d.area.includes('Stability') || d.area.includes('Core'))) {
    suggestions.push({
      cptCode: '97112',
      description: 'Neuromuscular Re-education',
      requirements: ['15 minutes', 'Must address movement pattern'],
      documentationTips: ['Describe compensatory patterns addressed'],
      rvus: 1.5
    })
  }
  
  // RTM codes for remote monitoring
  suggestions.push({
    cptCode: '98977',
    description: 'RTM Treatment Management',
    requirements: ['20 minutes remote communication', 'Device supply required'],
    documentationTips: ['Document remote session duration', 'Track compliance'],
    rvus: 2.0
  })
  
  return suggestions
}
```

---

## 4. ERROR HANDLING IMPROVEMENTS

### 4.1 Global Error Handler

```typescript
// Add to index.tsx
app.onError((err, c) => {
  // Log error with safe logging
  safeLog.error('Unhandled error', err as Error, {
    path: c.req.path,
    method: c.req.method,
    clinicianId: c.get('clinicianId')
  })
  
  // Return appropriate status
  if (err instanceof ZodError) {
    return c.json({
      success: false,
      error: 'Validation error',
      code: 'VALIDATION_ERROR',
      details: err.errors
    }, 400)
  }
  
  // Generic error for production
  return c.json({
    success: false,
    error: 'An unexpected error occurred',
    code: 'INTERNAL_ERROR',
    requestId: c.get('requestId')
  }, 500)
})
```

### 4.2 Not Found Handler

```typescript
app.notFound((c) => {
  return c.json({
    success: false,
    error: 'Endpoint not found',
    code: 'NOT_FOUND',
    path: c.req.path,
    method: c.req.method
  }, 404)
})
```

---

## 5. IMPLEMENTATION PRIORITY

| Priority | Task | Impact |
|----------|------|--------|
| P0 | Add global error handler | Fix 500 leak |
| P0 | Add not found handler | Fix 404 confusion |
| P1 | Create RAG knowledge tables | Enrich AI outputs |
| P1 | Add provider assistance API | Empower clinicians |
| P2 | Seed clinical guidelines | Improve recommendations |
| P2 | Add billing guidance | Revenue optimization |

---

*This report identifies specific error patterns and provides a roadmap for RAG enhancement and provider assistance improvements.*
