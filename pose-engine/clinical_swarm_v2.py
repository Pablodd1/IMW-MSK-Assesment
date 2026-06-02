"""
Clinical Swarm v2 — 6-agent validated clinical reasoning pipeline.
Agents: Pose Validator, PT Specialist, Chiropractic Specialist,
        Red Flag Sentinel, Evidence Librarian, SOAP Synthesizer.
"""
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any


@dataclass
class AgentOutput:
    agent_name: str
    finding: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)


class ClinicalSwarmV2:
    """
    Multi-agent clinical reasoning with validation gates.
    Requires injectable LLM clients (gemini, local, rag).
    """

    def __init__(
        self,
        gemini_client: Any,
        rag_retriever: Any,
        local_llm_client: Any
    ):
        self.gemini = gemini_client
        self.rag = rag_retriever
        self.local = local_llm_client

    async def analyze(
        self,
        pose_data: Dict,
        fms_results: Dict[str, Any],
        chiro_data: Dict,
        patient_ctx: Dict
    ) -> Dict:
        # STEP 1: Pose Validator (fast, local)
        validator_prompt = (
            "You are a Pose Data Validator. Given these detected landmark IDs, "
            "identify which body regions have sufficient data for clinical inference. "
            'Output ONLY JSON: {"valid_regions":["shoulder","knee"],'
            '"missing":["wrist"],"quality_score":0.8}\n\nLandmark IDs: '
            + json.dumps([k.get("id") for k in pose_data.get("keypoints", [])])
        )
        val_raw = await self._complete_local(validator_prompt)
        try:
            validation = json.loads(val_raw)
        except Exception:
            validation = {"valid_regions": [], "missing": [], "quality_score": 0.0}

        # STEP 2: Retrieve evidence
        def _get_score(v):
            if hasattr(v, 'score'):
                s = v.score
                if hasattr(s, 'value'):
                    return s.value
                return int(s)
            if isinstance(v, dict):
                return v.get('score', 'unknown')
            return str(v)
        evidence_ctx = await self._rag_search(
            f"{patient_ctx.get('chief_complaint', 'pain')} "
            f"{json.dumps({k: _get_score(v) for k, v in fms_results.items()})}"
        )

        # STEP 3: Parallel specialists
        pt_task = self._pt_specialist(pose_data, fms_results, validation, evidence_ctx)
        chiro_task = self._chiro_specialist(pose_data, chiro_data, validation, evidence_ctx)
        red_task = self._red_flag_sentinel(pose_data, patient_ctx)

        pt_out, chiro_out, redflag_out = await asyncio.gather(
            pt_task, chiro_task, red_task
        )

        # STEP 4: Conflict detection
        conflicts = self._detect_conflicts(pt_out, chiro_out)

        # STEP 5: SOAP synthesis
        soap = await self._soap_synthesizer(
            pt_out, chiro_out, redflag_out, conflicts, evidence_ctx
        )

        return {
            "validation": validation,
            "pt_assessment": {
                "finding": pt_out.finding,
                "confidence": pt_out.confidence,
                "evidence": pt_out.evidence,
            },
            "chiropractic_assessment": {
                "finding": chiro_out.finding,
                "confidence": chiro_out.confidence,
                "evidence": chiro_out.evidence,
            },
            "red_flags": {
                "finding": redflag_out.finding,
                "flags": redflag_out.red_flags,
            },
            "conflicts": conflicts,
            "soap_note": soap,
            "human_review_required": (
                len(redflag_out.red_flags) > 0 or len(conflicts) > 0
            ),
        }

    async def _complete_local(self, prompt: str) -> str:
        return await self.local.complete(prompt)

    async def _complete_gemini(self, prompt: str) -> str:
        return await self.gemini.complete(prompt)

    async def _rag_search(self, query: str) -> List[Dict]:
        return await self.rag.search(query)

    async def _pt_specialist(
        self, pose_data, fms, validation, evidence
    ) -> AgentOutput:
        def _score(test_key):
            v = fms.get(test_key, {})
            if hasattr(v, 'score'):
                s = v.score
                if hasattr(s, 'value'):
                    return s.value
                return int(s)
            if isinstance(v, dict):
                return v.get('score', 'unknown')
            return 'unknown'
        prompt = (
            "You are a Board-Certified Physical Therapist.\n"
            "Analyze this movement data and output JSON only:\n"
            '{"finding":"...","confidence":0.85,"evidence":["..."],"red_flags":[]}\n\n'
            f"FMS Deep Squat score: {_score('deep_squat')}\n"
            f"FMS ASLR L: {_score('aslr_left')}\n"
            f"FMS ASLR R: {_score('aslr_right')}\n"
            f"Valid regions: {validation.get('valid_regions')}\n"
            f"Evidence: {json.dumps(evidence[:3])}"
        )
        raw = await self._complete_gemini(prompt)
        data = json.loads(raw)
        return AgentOutput(
            "pt_specialist",
            data.get("finding", ""),
            data.get("confidence", 0.5),
            data.get("evidence", []),
            data.get("red_flags", []),
        )

    async def _chiro_specialist(
        self, pose_data, chiro_data, validation, evidence
    ) -> AgentOutput:
        prompt = (
            "You are a Doctor of Chiropractic specializing in biomechanics.\n"
            "Analyze postural data and output JSON only:\n"
            '{"finding":"...","confidence":0.85,"evidence":["..."],"red_flags":[]}\n\n'
            f"Plumb line: {json.dumps(chiro_data.get('plumb_line', {}))}\n"
            f"Leg length: {json.dumps(chiro_data.get('leg_length', {}))}\n"
            f"Pelvic tilt: {json.dumps(chiro_data.get('pelvic_tilt', {}))}"
        )
        raw = await self._complete_gemini(prompt)
        data = json.loads(raw)
        return AgentOutput(
            "chiropractic",
            data.get("finding", ""),
            data.get("confidence", 0.5),
            data.get("evidence", []),
            data.get("red_flags", []),
        )

    async def _red_flag_sentinel(
        self, pose_data, patient_ctx
    ) -> AgentOutput:
        prompt = (
            "You are a clinical safety screener. Check for RED FLAGS:\n"
            "- Cauda equina, fracture, infection, cancer, vascular\n"
            "Output JSON: {\"finding\":\"...\",\"confidence\":1.0,\"red_flags\":[]}\n\n"
            f"Age: {patient_ctx.get('age', 'unknown')}\n"
            f"Chief complaint: {patient_ctx.get('chief_complaint', 'unknown')}\n"
            f"Trauma: {patient_ctx.get('trauma', 'none')}\n"
            f"Night pain: {patient_ctx.get('night_pain', 'unknown')}\n"
            f"Weight loss: {patient_ctx.get('weight_loss', 'unknown')}"
        )
        raw = await self._complete_local(prompt)
        data = json.loads(raw)
        return AgentOutput(
            "red_flag",
            data.get("finding", ""),
            data.get("confidence", 0.9),
            [],
            data.get("red_flags", []),
        )

    def _detect_conflicts(self, pt: AgentOutput, chiro: AgentOutput) -> List[Dict]:
        conflicts = []
        if (
            "manipulation" in chiro.finding.lower()
            and "contraindicated" in pt.finding.lower()
        ):
            conflicts.append({
                "type": "tx_conflict",
                "detail": "Chiro recommends manipulation; PT contraindicates",
            })
        return conflicts

    async def _soap_synthesizer(
        self, pt, chiro, redflag, conflicts, evidence
    ) -> Dict:
        prompt = (
            "Synthesize into structured SOAP note. Output JSON only:\n"
            "{\"subjective\":\"...\",\"objective\":\"...\","
            "\"assessment\":\"...\",\"plan\":\"...\","
            "\"icd10\":[],\"cpt\":[],\"confidence\":0.85,"
            "\"human_review_required\":true}\n\n"
            f"PT: {pt.finding}\n"
            f"Chiro: {chiro.finding}\n"
            f"Red flags: {json.dumps(redflag.red_flags)}\n"
            f"Conflicts: {json.dumps(conflicts)}"
        )
        raw = await self._complete_gemini(prompt)
        return json.loads(raw)
