# IMW PhysioMotion — Accuracy Upgrade Plan
**Date:** 2026-06-02  
**Goal:** Medical-grade 3D MSK assessment for PT & Chiropractic  
**Current Stack:** YOLO11-pose ONNX + 1D Kalman + DEIMv2 + 3-specialist swarm

---

## Files Added

| File | Purpose |
|------|---------|
| `pose-engine/dwpose_backend.py` | DWPose 133-keypoint wholebody inference (hands + feet) |
| `pose-engine/one_euro_filter.py` | 1-Euro adaptive filter — replaces Kalman for zero-lag smoothing |
| `pose-engine/bone_constraints.py` | Bone-length constancy enforcement — anatomically invalid poses rejected |
| `pose-engine/fms_engine.py` | Automated FMS scoring (Deep Squat, ASLR, Shoulder Mobility) |
| `pose-engine/chiropractic_analyzer.py` | Postural screen, leg length, pelvic tilt, forward head posture |
| `pose-engine/clinical_swarm_v2.py` | 6-agent validated clinical reasoning (validator + PT + chiro + red flag + evidence + SOAP) |
| `pose-engine/camera_calibration.py` | Femto Mega depth calibration: median neighborhood + intrinsics + temporal IIR |
| `public/static/calibration-wizard.js` | Patient positioning wizard: T-pose, distance, side-view guidance |

---

## Upgrade Priority

### P0 — Immediate (Highest Impact, Lowest Effort)
1. **1-Euro Filter** → Replace Kalman in `server.py`
2. **Bone-Length Constraints** → Add `AnatomicalSkeletonFilter` after pose output

### P1 — This Week
3. **DWPose Integration** → Clinical-precision capture on "Assess" button
4. **FMS Deep Squat + ASLR** → Automated scoring from anterior camera
5. **Chiropractic Analyzer** → Postural screen module

### P2 — Next Sprint
6. **6-Agent Swarm v2** → Replace current 3-agent chain with validation gates
7. **Femto Mega Calibration** → Median depth + intrinsic projection
8. **Positioning Wizard** → Patient self-calibration UI

---

## Key Accuracy Improvements

| Layer | Change | Expected Gain |
|-------|--------|---------------|
| Temporal smoothing | Kalman → 1-Euro | 20–30% jitter reduction, no lag |
| Anatomical constraints | Add bone-length filter | Eliminates impossible poses |
| Pose model | YOLO11 (17kp) → DWPose (133kp) | Hands/feet for FMS & balance |
| Depth processing | Single-pixel → median 11×11 + intrinsics | 20–40% depth accuracy |
| Clinical reasoning | 3-agent chain → 6-agent validated swarm | Safety + evidence citation |

---

## FMS Automation Reality

From **single anterior camera**:
- ✅ Deep Squat — knee valgus, depth, torso angle
- ✅ ASLR — leg straightness, height
- ⚠️ Shoulder Mobility — requires DWPose hand keypoints
- ❌ Hurdle Step / Inline Lunge / Trunk Push-Up / Rotary Stability — need sagittal view

**Solution:** Camera-positioning wizard prompts patient to turn side-on for tests requiring sagittal plane.

---

## Next Step
Wire P0 filters into `server.py` and test with live camera.
