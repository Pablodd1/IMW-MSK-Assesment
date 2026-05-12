// ============================================================================
// MSK Analysis API Route
// Body-region agent swarm + vector RAG + crypto integrity
// POST /api/msk-analysis — Full pipeline entry point
// ============================================================================

import { Hono } from 'hono';
import { Brain, MOVEMENT_REGION_MAP } from '../utils/brain-orchestrator.js';
import { vectorRAGQuery, embedMovementData } from '../utils/vector-rag.js';
import {
  signMovementSession,
  verifyMovementSession,
  verifyDataIntegrity
} from '../utils/crypto-vault.js';

const mskRouter = new Hono();

// Apply integrity verification middleware to all MSK routes
mskRouter.use('*', verifyDataIntegrity());

// ============================================================================
// POST /api/msk-analysis — Run full multi-agent assessment pipeline
// ============================================================================

mskRouter.post('/', async (c) => {
  const body = await c.req.json();

  const {
    patientId,
    sessionId,
    movementTest,       // e.g. 'deep_squat', 'shoulder_mobility'
    landmarks,          // MediaPipe pose landmarks (33 points)
    signedFrames,       // Optional: pre-signed frames for integrity check
    sessionSignature,   // Optional: session-level chain signature
    frameCount
  } = body;

  // Basic validation
  if (!patientId || !movementTest) {
    return c.json({ success: false, error: 'patientId and movementTest are required' }, 400);
  }

  const activeSessionId = sessionId || `session-${Date.now()}`;

  try {
    // -- Step 1: Verify data integrity if signed frames provided --
    if (signedFrames && sessionSignature) {
      const integrityResult = verifyMovementSession({
        signedFrames,
        sessionSignature,
        frameCount: frameCount || signedFrames.length
      });

      if (!integrityResult.valid) {
        return c.json({
          success: false,
          error: 'Movement data integrity check failed',
          detail: integrityResult.reason,
          invalidFrames: integrityResult.invalidFrames
        }, 403);
      }
    }

    // -- Step 2: Run body-region specialist agent swarm --
    console.log(`\n🤖 MSK Analysis: ${movementTest} | Patient: ${patientId}`);

    const agentResult = await Brain.executeTask({
      app: 'physiomotion',
      type: 'movement_assessment',
      input: {
        patientId,
        sessionId: activeSessionId,
        assessmentType: movementTest, // Body-region routing key
        landmarks: landmarks || [],
        videoDuration: body.videoDuration || 0
      },
      requiredConfidence: 0.82,
      enableDebate: true
    });

    // -- Step 3: Aggregate findings from all region agents --
    const agentFindings: Record<string, any> = {};
    const allCompensations: string[] = [];
    const allRomStatus: Record<string, string> = {};
    const bodyRegions: string[] = [];

    for (const result of agentResult.results) {
      agentFindings[result.agentId] = result.result;

      if (result.result?.compensations) {
        allCompensations.push(...result.result.compensations);
      }
      if (result.result?.rom_status) {
        Object.assign(allRomStatus, result.result.rom_status);
      }
      if (result.result?.region && result.result.region !== 'none') {
        bodyRegions.push(result.result.region);
      }
    }

    // Identify deficiencies from agent findings
    const deficiencies: string[] = [];
    for (const [agentId, findings] of Object.entries(agentFindings)) {
      if (!findings || typeof findings !== 'object') continue;
      for (const [metric, status] of Object.entries(findings.rom_status || {})) {
        if (status === 'limited') {
          deficiencies.push(`${metric.replace(/_/g, ' ')} (${agentId.replace(/-/g, ' ')})`);
        }
      }
    }

    // -- Step 4: Generate encrypted movement embedding for RAG --
    let encryptedEmbedding = null;
    let textSummary = '';

    try {
      const embeddingResult = await embedMovementData({
        patientId: patientId.toString(),
        sessionId: activeSessionId,
        movementTest,
        bodyRegion: bodyRegions.join(', '),
        agentFindings,
        compensations: allCompensations
      });

      encryptedEmbedding = embeddingResult.encryptedEmbedding;
      textSummary = embeddingResult.textSummary;
    } catch (err) {
      console.warn('Embedding generation failed (non-critical):', err);
    }

    // -- Step 5: Vector RAG query for clinical context --
    const ragResult = await vectorRAGQuery(
      (c.env as any)?.DB || null,
      `Clinical guidelines for ${movementTest.replace(/_/g, ' ')} assessment deficiencies: ${deficiencies.join(', ')}`,
      {
        patientId,
        deficiencies,
        bodyRegions,
        intent: 'clinical',
        agentFindings,
        useEncryptedSearch: false
      }
    );

    // -- Step 6: Compile full clinical report --
    const report = {
      assessment_id: `msk-${Date.now()}`,
      patient_id: patientId,
      session_id: activeSessionId,
      movement_test: movementTest,
      assessed_at: new Date().toISOString(),

      // Agent swarm results
      agent_analysis: {
        consensus_confidence: agentResult.consensusConfidence,
        agents_activated: agentResult.results.length,
        body_regions_analyzed: [...new Set(bodyRegions)],
        debate_rounds: agentResult.debateRounds,
        agreed: agentResult.agreed,
        regional_findings: agentFindings
      },

      // Clinical summary
      clinical_summary: {
        overall_quality_score: Math.round(agentResult.consensusConfidence * 100),
        compensations_detected: [...new Set(allCompensations)],
        rom_status_by_joint: allRomStatus,
        deficiencies_identified: deficiencies,
        risk_level: agentFindings['predictor']?.reInjuryRisk > 0.7 ? 'high' :
                    agentFindings['predictor']?.reInjuryRisk > 0.4 ? 'moderate' : 'low',
        re_injury_risk: agentFindings['predictor']?.reInjuryRisk || null,
        stability_score: agentFindings['balance']?.stabilityScore || null
      },

      // RAG clinical context
      clinical_context: {
        answer: ragResult.answer,
        confidence: ragResult.confidence,
        similarity_score: ragResult.similarityScore,
        cpt_codes: ragResult.cptCodes || [],
        icd10_codes: ragResult.icd10Codes || [],
        guidelines: (ragResult.guidelines || []).map(g => ({
          title: g.title,
          source: g.source,
          evidence_level: g.evidence_level
        })),
        recommendations: ragResult.recommendations || []
      },

      // Crypto integrity
      data_integrity: {
        frames_verified: signedFrames ? signedFrames.length : 0,
        embedding_encrypted: !!encryptedEmbedding,
        embedding_dimension: encryptedEmbedding ? 128 : null,
        movement_text_summary: textSummary.substring(0, 200)
      }
    };

    return c.json({ success: true, data: report }, 200);

  } catch (err: any) {
    console.error('MSK analysis error:', err);
    return c.json({
      success: false,
      error: 'MSK analysis pipeline failed',
      message: err.message
    }, 500);
  }
});

// ============================================================================
// GET /api/msk-analysis/agents — List available body-region agents & status
// ============================================================================

mskRouter.get('/agents', (c) => {
  const status = Brain.getAgentStatus();
  const physioAgents = status.filter(a =>
    ['cervical-spine', 'thoracic-lumbar',
     'upper-extremity-left', 'upper-extremity-right',
     'lower-extremity-left', 'lower-extremity-right',
     'balance', 'predictor', 'comparative', 'exercise-rx'
    ].includes(a.id)
  );

  return c.json({
    success: true,
    data: {
      total_agents: physioAgents.length,
      agents: physioAgents,
      supported_movements: Object.keys(MOVEMENT_REGION_MAP),
      movement_routing: MOVEMENT_REGION_MAP
    }
  });
});

// ============================================================================
// POST /api/msk-analysis/sign-frames — Sign movement frames for integrity
// ============================================================================

mskRouter.post('/sign-frames', async (c) => {
  const body = await c.req.json();
  const { frames, patientId, sessionId } = body;

  if (!frames || !Array.isArray(frames) || !patientId || !sessionId) {
    return c.json({
      success: false,
      error: 'frames (array), patientId and sessionId required'
    }, 400);
  }

  const signed = signMovementSession(frames, patientId.toString(), sessionId);

  return c.json({
    success: true,
    data: {
      frameCount: signed.frameCount,
      sessionSignature: signed.sessionSignature,
      signedFrames: signed.signedFrames,
      message: 'Frames signed. Include signedFrames and sessionSignature in your analysis request.'
    }
  });
});

// ============================================================================
// GET /api/msk-analysis/movements/:test — Get agents for a specific test
// ============================================================================

mskRouter.get('/movements/:test', (c) => {
  const test = c.req.param('test');
  const regions = MOVEMENT_REGION_MAP[test];

  if (!regions) {
    return c.json({
      success: false,
      error: `Unknown movement test: ${test}`,
      available: Object.keys(MOVEMENT_REGION_MAP)
    }, 404);
  }

  const agents = Brain.getAgentsForMovement(test);

  return c.json({
    success: true,
    data: {
      movement_test: test,
      activated_regions: regions,
      agents_activated: agents.length,
      agents: agents.map(a => ({
        id: a.id,
        name: a.name,
        body_region: a.bodyRegion,
        confidence_threshold: a.confidenceThreshold,
        landmark_keys: a.landmarkKeys || []
      }))
    }
  });
});

export default mskRouter;
