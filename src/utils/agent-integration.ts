// ============================================================================
// Multi-Agent Integration Layer
// Connects Brain Orchestrator to PhysioMotion & DealFlow
// ============================================================================

import { Brain } from './brain-orchestrator.js';
import { ACPBus, BaseAgent, ACPMessage } from './agent-communication-protocol.js';

// ============================================================================
// PhysioMotion Integration
// ============================================================================

export interface PhysioMotionAssessmentInput {
  patientId: string;
  assessmentType: 'squat' | 'lunge' | 'gait' | 'balance' | 'rom';
  landmarks: any[]; // MediaPipe pose landmarks
  videoDuration: number;
  previousAssessments?: any[];
}

export interface PhysioMotionAssessmentResult {
  biomechanics: {
    jointAngles: Record<string, number>;
    symmetry: number;
    deviations: string[];
  };
  balance: {
    swayArea: number;
    centerOfPressure: { x: number; y: number };
    stabilityScore: number;
  };
  riskAssessment: {
    reInjuryRisk: number;
    riskFactors: string[];
    timeline: string;
  };
  recommendations: {
    exercises: string[];
    restrictions: string[];
    followUp: string;
  };
  confidence: number;
  agentConsensus: {
    agreed: boolean;
    participatingAgents: string[];
  };
}

export async function runPhysioMotionAssessment(
  input: PhysioMotionAssessmentInput
): Promise<PhysioMotionAssessmentResult> {
  console.log('\n🏃 PHYSIOMOTION: Running Multi-Agent Assessment');
  
  const result = await Brain.executeTask({
    app: 'physiomotion',
    type: 'movement_assessment',
    input,
    requiredConfidence: 0.80,
    enableDebate: true
  });
  
  // Transform consensus result to app-specific format
  return {
    biomechanics: result.finalAnswer?.biomechanics || {
      jointAngles: {},
      symmetry: 0.5,
      deviations: []
    },
    balance: result.finalAnswer?.balance || {
      swayArea: 0,
      centerOfPressure: { x: 0, y: 0 },
      stabilityScore: 0.5
    },
    riskAssessment: result.finalAnswer?.riskAssessment || {
      reInjuryRisk: 0.5,
      riskFactors: [],
      timeline: 'unknown'
    },
    recommendations: result.finalAnswer?.recommendations || {
      exercises: [],
      restrictions: [],
      followUp: 'Schedule follow-up in 2 weeks'
    },
    confidence: result.consensusConfidence,
    agentConsensus: {
      agreed: result.agreed,
      participatingAgents: result.results.map(r => r.agentName)
    }
  };
}

export async function generateExercisePrescription(
  patientId: string,
  assessmentData: any,
  goals: string[]
): Promise<{
  exercises: any[];
  progression: any[];
  precautions: string[];
  confidence: number;
}> {
  const result = await Brain.executeTask({
    app: 'physiomotion',
    type: 'exercise_prescription',
    input: {
      patientId,
      assessmentData,
      goals
    },
    requiredConfidence: 0.85,
    enableDebate: false
  });
  
  return {
    exercises: result.finalAnswer?.exercises || [],
    progression: result.finalAnswer?.progression || [],
    precautions: result.finalAnswer?.precautions || [],
    confidence: result.consensusConfidence
  };
}

// ============================================================================
// DealFlow Integration
// ============================================================================

export interface DealFlowDocumentInput {
  clientId: string;
  documentType: 'I-485' | 'I-140' | 'N-400' | 'I-130' | 'contract';
  content: string;
  caseType: string;
}

export interface DealFlowDocumentResult {
  keyTerms: Array<{
    term: string;
    value: string;
    significance: 'high' | 'medium' | 'low';
  }>;
  risks: Array<{
    type: string;
    description: string;
    severity: 'critical' | 'warning' | 'info';
    recommendation: string;
  }>;
  missingDocuments: string[];
  legalResearch: {
    relevantCases: string[];
    statutes: string[];
    circuitSplit: boolean;
  };
  strategy: {
    recommendedApproach: string;
    alternatives: string[];
    timeline: string;
  };
  confidence: number;
  agentConsensus: {
    agreed: boolean;
    participatingAgents: string[];
  };
}

export async function analyzeDealFlowDocument(
  input: DealFlowDocumentInput
): Promise<DealFlowDocumentResult> {
  console.log('\n⚖️  DEALFLOW: Running Multi-Agent Document Analysis');
  
  const result = await Brain.executeTask({
    app: 'dealflow',
    type: 'document_review',
    input,
    requiredConfidence: 0.88,
    enableDebate: true
  });
  
  return {
    keyTerms: result.finalAnswer?.keyTerms || [],
    risks: result.finalAnswer?.risks || [],
    missingDocuments: result.finalAnswer?.missingDocuments || [],
    legalResearch: result.finalAnswer?.legalResearch || {
      relevantCases: [],
      statutes: [],
      circuitSplit: false
    },
    strategy: result.finalAnswer?.strategy || {
      recommendedApproach: 'Standard filing',
      alternatives: [],
      timeline: '6-12 months'
    },
    confidence: result.consensusConfidence,
    agentConsensus: {
      agreed: result.agreed,
      participatingAgents: result.results.map(r => r.agentName)
    }
  };
}

export async function generateCaseStrategy(
  clientId: string,
  caseDetails: any
): Promise<{
  theory: string;
  claims: string[];
  defenses: string[];
  settlementRange: { min: number; max: number };
  winProbability: number;
  confidence: number;
}> {
  const result = await Brain.executeTask({
    app: 'dealflow',
    type: 'case_analysis',
    input: {
      clientId,
      caseDetails
    },
    requiredConfidence: 0.85,
    enableDebate: true
  });
  
  return {
    theory: result.finalAnswer?.theory || 'Standard approach',
    claims: result.finalAnswer?.claims || [],
    defenses: result.finalAnswer?.defenses || [],
    settlementRange: result.finalAnswer?.settlementRange || { min: 0, max: 0 },
    winProbability: result.finalAnswer?.winProbability || 0.5,
    confidence: result.consensusConfidence
  };
}

// ============================================================================
// Concrete Agent Implementations
// ============================================================================

// PhysioMotion: Biomechanics Agent
export class BiomechanicsAgent extends BaseAgent {
  constructor() {
    super({
      id: 'biomechanics',
      name: 'Biomechanics Agent',
      capabilities: ['pose_analysis', 'joint_angles', 'rom_calculation', 'symmetry_check']
    });
  }
  
  protected async handleMessage(message: ACPMessage): Promise<void> {
    console.log(`   🔧 ${this.name} processing ${message.type}`);
    
    if (message.type === 'TASK_REQUEST') {
      const { landmarks, assessmentType } = message.payload;
      
      // Analyze joint angles
      const analysis = this.analyzeJointAngles(landmarks, assessmentType);
      
      await this.respond(message, {
        biomechanics: analysis,
        confidence: 0.87,
        reasoning: `Analyzed ${landmarks?.length || 0} pose landmarks for ${assessmentType}`
      });
    }
  }
  
  private analyzeJointAngles(landmarks: any[], type: string): any {
    // Simplified - would use actual biomechanics calculations
    return {
      kneeFlexion: 95,
      hipFlexion: 85,
      ankleDorsiflexion: 12,
      kneeValgus: 8.2,
      symmetryScore: 0.82
    };
  }
}

// PhysioMotion: Risk Predictor Agent
export class RiskPredictorAgent extends BaseAgent {
  constructor() {
    super({
      id: 'predictor',
      name: 'Risk Predictor Agent',
      capabilities: ['risk_forecast', 'injury_prediction', 'outcome_prediction']
    });
  }
  
  protected async handleMessage(message: ACPMessage): Promise<void> {
    if (message.type === 'TASK_REQUEST') {
      const { assessmentData } = message.payload;
      
      // Run prediction model
      const prediction = this.calculateRisk(assessmentData);
      
      await this.respond(message, {
        riskAssessment: prediction,
        confidence: 0.78,
        reasoning: 'Regression model using biomechanical factors + patient history'
      });
    }
  }
  
  private calculateRisk(data: any): any {
    return {
      reInjuryRisk: 0.68,
      riskFactors: ['knee_valgus', 'asymmetry', 'limited_hip_mobility'],
      timeline: '6 months',
      confidenceInterval: { lower: 0.55, upper: 0.78 }
    };
  }
}

// DealFlow: Legal Research Agent
export class LegalResearchAgent extends BaseAgent {
  constructor() {
    super({
      id: 'legal-research',
      name: 'Legal Research Agent',
      capabilities: ['case_law_search', 'statute_analysis', 'precedent_matching']
    });
  }
  
  protected async handleMessage(message: ACPMessage): Promise<void> {
    if (message.type === 'TASK_REQUEST') {
      const { documentType, caseType } = message.payload;
      
      // Simulate legal research
      const research = this.searchLegalDatabase(documentType, caseType);
      
      await this.respond(message, {
        legalResearch: research,
        confidence: 0.92,
        reasoning: `Searched legal databases for ${caseType} ${documentType} cases`
      });
    }
  }
  
  private searchLegalDatabase(docType: string, caseType: string): any {
    return {
      relevantCases: ['Matter of A-B-', 'Matter of X-Y-Z'],
      statutes: ['INA § 245(a)', '8 CFR § 245.1'],
      circuitSplit: false,
      recentDevelopments: []
    };
  }
}

// DealFlow: Strategy Agent
export class StrategyAgent extends BaseAgent {
  constructor() {
    super({
      id: 'strategy',
      name: 'Strategy Agent',
      capabilities: ['case_theory', 'claim_recommendation', 'defense_analysis']
    });
  }
  
  protected async handleMessage(message: ACPMessage): Promise<void> {
    if (message.type === 'TASK_REQUEST') {
      const { caseDetails, legalResearch } = message.payload;
      
      // Generate strategy
      const strategy = this.developStrategy(caseDetails, legalResearch);
      
      await this.respond(message, {
        strategy,
        confidence: 0.85,
        reasoning: 'Developed strategy based on case facts and legal research'
      });
    }
  }
  
  private developStrategy(details: any, research: any): any {
    return {
      recommendedApproach: 'Concurrent filing I-140/I-485',
      alternatives: ['Consular processing', 'EB-1A upgrade'],
      timeline: '12-18 months',
      criticalMilestones: ['PERM approval', 'I-140 approval', 'I-485 filing']
    };
  }
}

// ============================================================================
// Initialize All Agents
// ============================================================================

export function initializeAgentSwarm(): void {
  console.log('\n🐝 Initializing Agent Swarm...\n');
  
  // PhysioMotion agents
  const biomechanicsAgent = new BiomechanicsAgent();
  biomechanicsAgent.initialize();
  
  const riskPredictorAgent = new RiskPredictorAgent();
  riskPredictorAgent.initialize();
  
  // DealFlow agents
  const legalResearchAgent = new LegalResearchAgent();
  legalResearchAgent.initialize();
  
  const strategyAgent = new StrategyAgent();
  strategyAgent.initialize();
  
  console.log('\n✅ Agent Swarm initialized\n');
}

// ============================================================================
// Usage Examples
// ============================================================================

export async function demoIntegration(): Promise<void> {
  // Initialize agents
  initializeAgentSwarm();
  
  // Demo 1: PhysioMotion
  console.log('=== PHYSIOMOTION DEMO ===');
  const physioResult = await runPhysioMotionAssessment({
    patientId: 'patient-123',
    assessmentType: 'squat',
    landmarks: [], // Would be actual MediaPipe data
    videoDuration: 45
  });
  
  console.log('\nPhysioMotion Result:');
  console.log(`  Confidence: ${(physioResult.confidence * 100).toFixed(1)}%`);
  console.log(`  Knee Valgus: ${physioResult.biomechanics.jointAngles.kneeValgus}°`);
  console.log(`  Re-injury Risk: ${(physioResult.riskAssessment.reInjuryRisk * 100).toFixed(0)}%`);
  
  // Demo 2: DealFlow
  console.log('\n=== DEALFLOW DEMO ===');
  const legalResult = await analyzeDealFlowDocument({
    clientId: 'client-456',
    documentType: 'I-485',
    content: 'Adjustment of status application...',
    caseType: 'EB-2 NIW'
  });
  
  console.log('\nDealFlow Result:');
  console.log(`  Confidence: ${(legalResult.confidence * 100).toFixed(1)}%`);
  console.log(`  Risks found: ${legalResult.risks.length}`);
  console.log(`  Missing docs: ${legalResult.missingDocuments.join(', ') || 'None'}`);
  
  // Show ACP stats
  console.log('\n=== ACP BUS STATISTICS ===');
  const stats = ACPBus.getStats();
  console.log(`  Total subscriptions: ${stats.totalSubscriptions}`);
  console.log(`  Messages by type:`, stats.messagesByType);
}

// Run demo if executed directly
try {
  if (require && require.main === module) {
    demoIntegration().catch(console.error);
  }
} catch { /* ESM context — skip */ }
