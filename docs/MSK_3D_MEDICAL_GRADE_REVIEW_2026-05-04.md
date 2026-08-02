# MSK 3D Medical-Grade Capability Review (Laptop + Phone + Orbbec/Femto)

Date: 2026-05-04  
Reviewer: GPT-5.3-Codex

## 1) Bottom Line

You asked for a practical review of what is already implemented vs missing for a **medical-grade, real-time 3D MSK platform** (patient/doctor/admin + telehealth + voice + recording + AI notes + treatment plans + HIPAA + encryption/RAG).

**Current status:** strong prototype with many core blocks present, but **not yet medical-grade production ready**.

---

## 2) Feature-by-Feature Status (What is in / not in / working risk)

### A. Real-time joint tracking + 3D skeleton + blue-line overlay

**In repo now**
- Live assessment page and real-time pose workflow exist (`public/assessment.html`, `public/static/assessment-workflow-v2.js`).
- Pose engine and skeleton rendering hooks are wired (`PoseEngine`, `SkeletonRenderer` usage).
- Joint angle computation and live dashboard updates are present.

**Gaps / risks**
- True **validated 3D medical-grade** kinematic accuracy not proven by calibration/validation tests in runtime path.
- Filtering implementation has a critical design issue (axis state contamination) in enhanced tracker, which can reduce reliability if used in production analysis.

**Verdict**: **Partially implemented, not medical-grade validated yet**.

### B. Multi-device capture (laptop webcam, cellphone, Orbbec/Femto)

**In repo now**
- Camera type selection workflow exists in assessment v2.
- Mobile/phone handling appears present (flip camera UI logic).
- Femto/Orbbec-oriented path exists conceptually (`type === 'femto'`, setup branch).

**Gaps / risks**
- Some Femto integration points still look like integration placeholders in codebase notes.
- No clear end-to-end hardware validation evidence for each device path under clinical constraints.

**Verdict**: **Implemented at integration level, needs hardware-complete verification and hardening**.

### C. Graphics + dynamic + interactive side-by-side realtime UI

**In repo now**
- Interactive tabs, real-time metrics panel, quality overlays, and live canvas rendering are present.

**Gaps / risks**
- Performance and deterministic behavior under low-end devices/network conditions need measured QA.

**Verdict**: **Mostly present in frontend UX**.

### D. Recording video + microphone (patient complaints + clinician dictation)

**In repo now**
- Video routes and upload/listing backend exist (`src/routes/videos.ts`).
- Browser permissions policy includes camera and microphone.
- Voice command module exists.

**Gaps / risks**
- Need explicit confirmation of synchronized **audio+video+pose+timestamps** pipeline with reliable storage and replay integrity.
- Need explicit workflow for clinician/patient voice -> structured note fields with provenance.

**Verdict**: **Foundation exists; full clinical-grade capture pipeline not fully evidenced**.

### E. Automated medical note + treatment recommendations

**In repo now**
- Biomechanical analysis routes exist.
- RAG + agent orchestration + Gemini integration hooks exist (`msk-analysis`, `vector-rag`, agent orchestrator).
- Exercise/prescription routes and schemas are present.

**Gaps / risks**
- Clinical governance not fully enforced (quality gates, safety checks, model confidence thresholds before recommendations).
- Need explicit evidence hierarchy and traceability in note generation for compliance.

**Verdict**: **Core AI pipeline present; clinical safety controls incomplete for medical-grade claim**.

### F. Patient app + doctor app + admin app + telehealth reassessment

**In repo now**
- Role-aware backend and admin routes exist.
- Patient/portal/admin-like pages and APIs are present.
- Reassessment-related data structures exist via assessments/tests workflow.

**Gaps / risks**
- Direct production-grade telehealth video-call implementation is not clearly complete in reviewed paths.

**Verdict**: **Multi-role architecture present; telehealth completeness requires verification**.

### G. Exercise library / prescribed videos

**In repo now**
- Exercise and prescription modules are implemented.
- Video upload/list endpoints exist and can support exercise media.

**Gaps / risks**
- Need explicit content QA workflow, adherence analytics, and outcome-loop feedback.

**Verdict**: **Present at API level; needs product-level completion**.

### H. Security, encryption, HIPAA, audit, RAG embeddings

**In repo now**
- HIPAA-oriented middleware and audit logging utilities are present.
- Crypto/integrity and encrypted embedding concepts are integrated in MSK analysis flow.
- RAG pipeline is present.

**Gaps / risks**
- Typing/policy rigor gaps exist in audit action handling.
- Need formal HIPAA controls validation (access controls, retention policies, BAA/vendor posture, incident response, immutable audit posture, etc.).

**Verdict**: **Strong security architecture direction; formal compliance readiness not yet proven**.

### I. Voice command for workflow/buttons/links

**In repo now**
- Voice command controller includes commands for recording, analysis, camera control, navigation, notes, save, help.

**Gaps / risks**
- Browser speech support variability and medical-environment noise robustness not validated.
- Need deterministic command confirmation UX for safety-critical actions.

**Verdict**: **Implemented with useful coverage; requires reliability hardening for clinical use**.

---

## 3) What Appears to Work vs What Is Not Yet Ready

### Appears to work (prototype level)
- Real-time pose tracking UI loop.
- Joint metrics rendering.
- Assessment/test creation flow.
- Video upload/listing backend.
- Role-based route structure.
- AI analysis + RAG + integrity flow wiring.
- Voice command action map.

### Not yet ready for medical-grade claim
- Verified and documented accuracy/certification process.
- Robust per-device clinical validation suite.
- Fail-closed quality gates before clinical output.
- End-to-end telehealth clinical workflow proof.
- Compliance evidence package (policy + technical control + audit trace validation).

---

## 4) Priority Fix Plan (Execution Order)

1. **Tracking correctness first**
   - Fix filter state isolation by axis.
   - Add kinematic unit/integration tests.

2. **Clinical quality gates**
   - Block report generation when confidence/quality minima are not met.
   - Persist quality indicators per session/report.

3. **Capture pipeline hardening**
   - Validate synchronized audio/video/skeleton capture and replay.
   - Add checksum/signature chain for recorded artifacts.

4. **Telehealth completion**
   - Finalize secure call + reassessment workflow and audit events.

5. **Compliance hardening**
   - Formalize PHI audit action vocabulary and enforcement.
   - Complete HIPAA control checklist and evidence mapping.

6. **Voice safety layer**
   - Add confirmation prompts for destructive/sensitive actions.
   - Add fallback UI and command confidence thresholds.

---


## 5) Direct Answers to Your Latest Questions

- **Should we use two agents (builder + reviewer)?** Yes. This is recommended for safety.
  - Agent 1: generates tracking output, draft note, draft recommendations.
  - Agent 2: independently verifies data quality, contradictions, safety constraints, and can block release until resolved.

- **Is the Femto/Orbbec camera properly connected right now?** Not proven by repository review alone.
  - The code includes Femto/Orbbec integration branches, but a real hardware session log is still required to confirm device connection, depth stream health, calibration, and timestamp stability on your exact machine.

- **Is code optimized and tested for production?** Partially optimized, not fully production-proven.
  - Frontend performance work exists, but production grade needs automated load/performance tests, device matrix tests, security regression tests, and clinical accuracy validation reports.

- **Can this run on GitHub + Vercel + Supabase?** Yes, this is a viable deployment architecture.
  - GitHub for source control/CI policy gates.
  - Vercel for frontend/backend hosting and preview deployments.
  - Supabase for Postgres/Auth/Storage with strict RLS and audit controls.
  - For medical-grade posture, add HIPAA-oriented control evidence, encryption at rest/in transit, signed URL policies, and strict role-scoped access.

## 6) Final Verdict

This codebase is a **good advanced foundation** for your vision, but it is currently best described as a **clinical prototype / pre-production platform**, not yet a fully validated medical-grade system.

If you want, next step should be a **hard implementation sprint** focused on the top 6 priorities above, starting with real-time tracking correctness and report quality gates.
