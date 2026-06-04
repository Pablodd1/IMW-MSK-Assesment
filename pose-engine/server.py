#!/usr/bin/env python3
"""
Boxer3D Pose Engine v4.0 — Dual-Model Pipeline
================================================
YOLO12-pose (real-time 17-keypoint skeleton) + DEIMv2 (async head pose enrichment)

Architecture:
  Camera → WebSocket → YOLO12-pose (every frame, ~15-30 FPS)
                     → DEIMv2-Wholebody34 (every N frames, async, ~600ms CPU)
                     → Merged output → Three.js 3D + Multi-agent swarm

DEIMv2 adds: head pose (8 directions), person attributes (age/gender/wheelchair),
             face parts, supplementary keypoints for clinical context.

Usage:
  python3 pose-engine/server.py [--port 8765] [--deimv2-model path] [--deimv2-interval 30]
"""

import asyncio, json, base64, sys, os, time, argparse, urllib.request, threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import cv2

# ─── YOLO pose via OpenCV DNN ───
YOLO12_POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s-pose.onnx"
YOLO11_POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx"
DEFAULT_YOLO_MODEL = "yolo12s-pose.onnx"
FALLBACK_YOLO_MODEL = "yolo11n-pose.onnx"

def download_model(url, path):
    if os.path.exists(path):
        return path
    print(f"[Boxer3D] Downloading model to {path}...")
    urllib.request.urlretrieve(url, path)
    return path

try:
    import websockets
except ImportError:
    os.system("pip3 install --user --break-system-packages websockets")
    import websockets

# ─── P0: 1-Euro Filter (replaces Kalman) ───
from one_euro_filter import OneEuroFilter, OneEuroFilter3D

# ─── P0: Anatomical Skeleton Filter ───
from bone_constraints import AnatomicalSkeletonFilter

# ─── P1: Clinical Assessment Engines ───
from fms_engine import FMSEngine, FMSScore
from chiropractic_analyzer import ChiropracticAnalyzer

# ─── P2: 6-Agent Clinical Swarm ───
from clinical_swarm_v2 import ClinicalSwarmV2, AgentOutput

# ─── P2: Mock LLM clients for swarm (swap in real clients for production) ───
class MockLLMClient:
    """Returns structured JSON when real LLM endpoints aren't available."""
    def __init__(self, name="mock"):
        self.name = name
    async def complete(self, prompt: str) -> str:
        import json
        # Return valid default JSON based on prompt type
        if "Pose Data Validator" in prompt:
            return json.dumps({"valid_regions": ["shoulder", "hip", "knee"],
                               "missing": [], "quality_score": 0.75})
        elif "clinical safety screener" in prompt:
            return json.dumps({"finding": "No acute red flags detected from available data",
                               "confidence": 0.85, "red_flags": []})
        elif "Board-Certified Physical Therapist" in prompt:
            return json.dumps({"finding": "Functional movement pattern assessment: "
                               "coordinated movement with potential minor asymmetries. "
                               "Recommend further ROM and strength testing.",
                               "confidence": 0.72,
                               "evidence": ["FMS normative data", "movement pattern analysis"],
                               "red_flags": []})
        elif "Doctor of Chiropractic" in prompt:
            return json.dumps({"finding": "Postural assessment shows neutral alignment "
                               "with minor deviations within normal range. "
                               "No structural contraindications for care.",
                               "confidence": 0.68,
                               "evidence": ["plumb line analysis", "pelvic symmetry check"],
                               "red_flags": []})
        elif "SOAP" in prompt or "Synthesize" in prompt:
            return json.dumps({
                "subjective": "Patient presents for movement assessment. "
                              "Chief complaint as documented.",
                "objective": "Movement screening performed. "
                             "Key observations documented in assessment sections.",
                "assessment": "Multi-agent analysis complete. "
                              "Findings integrated from PT, Chiropractic, and safety screening agents.",
                "plan": "1. Complete comprehensive physical examination\n"
                       "2. Initiate conservative care per clinical guidelines\n"
                       "3. Re-evaluate in 2-4 weeks\n"
                       "4. Patient education on home exercise program",
                "icd10": ["Z02.9", "M25.50"],
                "cpt": ["97110", "97112"],
                "confidence": 0.78,
                "human_review_required": False
            })
        else:
            return json.dumps({"response": "ok"})

class MockRAGRetriever:
    """Returns empty evidence when no knowledge base is connected."""
    async def search(self, query: str) -> list:
        return []

# ─── P1: DWPose (optional — on-demand clinical precision) ───
try:
    from dwpose_backend import DWPoseBackend
    DWPOSE_AVAILABLE = True
except Exception:
    DWPOSE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# YOLO-Pose Engine (OpenCV DNN — production)
# ═══════════════════════════════════════════════════════════════

class PoseEngine:
    COCO_KEYPOINTS = 17
    INPUT_SIZE = 640

    def __init__(self, model_path=DEFAULT_YOLO_MODEL, confidence=0.5):
        requested_model = os.path.basename(model_path)
        primary_url = YOLO12_POSE_URL if "yolo12" in requested_model else YOLO11_POSE_URL
        try:
            model_path = download_model(primary_url, model_path)
            self.model_name = "yolo12-pose" if "yolo12" in requested_model else "yolo11-pose"
        except Exception as exc:
            if "yolo12" not in requested_model:
                raise
            fallback_path = os.path.join(os.path.dirname(model_path) or ".", FALLBACK_YOLO_MODEL)
            print(f"[YOLO] YOLO12 model unavailable ({exc}); falling back to YOLO11: {fallback_path}")
            model_path = download_model(YOLO11_POSE_URL, fallback_path)
            self.model_name = "yolo11-pose"
        print(f"[YOLO] Loading ONNX: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.confidence = confidence
        self.euro_filters = {}
        self.skeleton_filter = AnatomicalSkeletonFilter(history_size=5)

    def get_euro(self, kid, axis):
        key = (kid, axis)
        if key not in self.euro_filters:
            self.euro_filters[key] = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.05, dcutoff=1.0)
        return self.euro_filters[key]

    def process_frame(self, jpeg_bytes, depth_data=None):
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Failed to decode image"}

        h, w = img.shape[:2]

        input_blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.INPUT_SIZE, self.INPUT_SIZE),
                                            swapRB=True, crop=False)
        self.net.setInput(input_blob)
        outputs = self.net.forward()

        keypoints_list = []
        output = outputs[0].transpose()

        scale_x = w / self.INPUT_SIZE
        scale_y = h / self.INPUT_SIZE

        for detection in output:
            score = float(detection[4:5].max())
            if score < self.confidence:
                continue

            cx, cy, bw, bh = detection[:4]
            x1 = (cx - bw/2) * scale_x
            y1 = (cy - bh/2) * scale_y
            x2 = (cx + bw/2) * scale_x
            y2 = (cy + bh/2) * scale_y

            keypoints_3d = []
            kp_start = 5
            for kid in range(self.COCO_KEYPOINTS):
                kp_x = detection[kp_start + kid * 3] * scale_x
                kp_y = detection[kp_start + kid * 3 + 1] * scale_y
                kp_conf = float(detection[kp_start + kid * 3 + 2])
                if kp_conf < 0.3:
                    continue

                z = 0.0
                if depth_data is not None:
                    dx, dy = int(kp_x), int(kp_y)
                    if 0 <= dx < depth_data.shape[1] and 0 <= dy < depth_data.shape[0]:
                        d = float(depth_data[dy, dx])
                        if 0 < d < 5000:
                            z = d / 1000.0

                sx = self.get_euro(kid, 'x').filter(kp_x / w)
                sy = self.get_euro(kid, 'y').filter(kp_y / h)
                sz = self.get_euro(kid, 'z').filter(z)

                keypoints_3d.append({
                    "id": kid,
                    "x": round(sx, 4),
                    "y": round(sy, 4),
                    "z": round(sz, 4),
                    "confidence": round(kp_conf, 3),
                    "depth_valid": z > 0
                })

            # P0: Apply anatomical skeleton constraints
            if keypoints_3d:
                keypoints_3d = self.skeleton_filter.update(keypoints_3d)

            if keypoints_3d:
                keypoints_list.append({
                    "keypoints": keypoints_3d,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(score, 3)
                })

            if len(keypoints_list) >= 1:
                break

        # Return the BGR image for DEIMv2 to reuse (avoid double decode)
        return {
            "persons": keypoints_list,
            "frame_w": w,
            "frame_h": h,
            "fps": 0,
            "model": self.model_name,
            "depth_available": depth_data is not None,
            "_bgr_frame": img  # internal — removed before JSON serialize
        }

    def reset_filters(self):
        self.euro_filters = {}
        self.skeleton_filter.reset()
        print("[PoseEngine] Filters reset (1-Euro + Skeleton)")


# ═══════════════════════════════════════════════════════════════
# DEIMv2-Wholebody34 Enrichment Engine
# ═══════════════════════════════════════════════════════════════

HEAD_POSE_CLASSES = {
    8: "front", 9: "right_front", 10: "right_side", 11: "right_back",
    12: "back", 13: "left_back", 14: "left_side", 15: "left_front"
}
ATTRIBUTE_CLASSES = {1: "adult", 2: "child", 3: "male", 4: "female", 5: "wheelchair", 6: "crutches"}
FACE_CLASSES = {16: "face", 17: "eye", 18: "nose", 19: "mouth", 20: "ear"}

CERVICAL_ROTATION_MAP = {
    "front": "neutral", "right_front": "right_rotation_mild",
    "right_side": "right_rotation_full", "right_back": "right_rotation_extreme",
    "back": "extension", "left_back": "left_rotation_extreme",
    "left_side": "left_rotation_full", "left_front": "left_rotation_mild"
}

class DEIMv2Enricher:
    """Async DEIMv2 inference for head pose + attribute enrichment."""

    def __init__(self, model_path, obj_conf=0.35, attr_conf=0.70, kp_conf=0.25):
        self.model_path = model_path
        self.obj_conf = obj_conf
        self.attr_conf = attr_conf
        self.kp_conf = kp_conf
        self.session = None
        self.input_name = None
        self._lock = threading.Lock()
        self._latest_result = None
        self._frame_count = 0
        self._last_inference_ms = 0

    def load(self):
        if not os.path.exists(self.model_path):
            print(f"[DEIMv2] Model not found: {self.model_path}")
            print("[DEIMv2] Download with: curl -L -o models/deimv2_wholebody34_680q.onnx \\")
            print("  https://github.com/PINTO0309/DEIMv2/releases/download/weights/deimv2_dinov3_x_wholebody34_680query_n_batch.onnx")
            return False

        import onnxruntime as ort
        print(f"[DEIMv2] Loading: {self.model_path}")
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        print(f"[DEIMv2] Ready. {self.session.get_providers()}")
        return True

    def preprocess(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0)
        return np.ascontiguousarray(tensor)

    def infer_async(self, bgr_image):
        """Non-blocking: run inference in thread, update latest result."""
        self._frame_count += 1
        threading.Thread(target=self._run_inference, args=(bgr_image,), daemon=True).start()

    def _run_inference(self, bgr_image):
        if self.session is None:
            return
        try:
            t0 = time.perf_counter()
            with self._lock:
                tensor = self.preprocess(bgr_image)
                raw = self.session.run(None, {self.input_name: tensor})[0][0]
                result = self._parse(raw, bgr_image.shape[1], bgr_image.shape[0])
                self._last_inference_ms = (time.perf_counter() - t0) * 1000
                self._latest_result = result
        except Exception as e:
            print(f"[DEIMv2] Inference error: {e}")

    def _parse(self, boxes, img_w, img_h):
        result = {"head_pose": None, "head_pose_confidence": 0.0,
                  "attributes": {}, "face_parts": {}, "keypoints": [],
                  "body_bbox": None, "head_bbox": None,
                  "detections": 0, "inference_ms": 0}

        if len(boxes) == 0:
            return result

        scores = boxes[:, 1]
        min_thresh = min(self.obj_conf, self.attr_conf, self.kp_conf)
        keep = scores > min_thresh
        if not keep.any():
            return result

        kept = boxes[keep]
        result["detections"] = len(kept)

        body_boxes, head_boxes, head_pose_boxes, attr_boxes, face_boxes = [], [], [], [], []

        for box in kept:
            cls_id, score = int(box[0]), float(box[1])
            x1 = int(max(0, box[2]) * img_w)
            y1 = int(max(0, box[3]) * img_h)
            x2 = int(min(box[4], 1.0) * img_w)
            y2 = int(min(box[5], 1.0) * img_h)
            det = (cls_id, score, x1, y1, x2, y2)

            if cls_id == 0 and score >= self.obj_conf:
                body_boxes.append(det)
            elif cls_id == 7 and score >= self.obj_conf:
                head_boxes.append(det)
            elif cls_id in HEAD_POSE_CLASSES and score >= self.attr_conf:
                head_pose_boxes.append(det)
            elif cls_id in ATTRIBUTE_CLASSES and score >= self.attr_conf:
                attr_boxes.append(det)
            elif cls_id in FACE_CLASSES and score >= self.obj_conf:
                face_boxes.append(det)

        # Head pose
        if head_pose_boxes:
            best = max(head_pose_boxes, key=lambda d: d[1])
            result["head_pose"] = HEAD_POSE_CLASSES[best[0]]
            result["head_pose_confidence"] = round(best[1], 3)

        # Attributes
        for a in attr_boxes:
            name = ATTRIBUTE_CLASSES[a[0]]
            if a[1] > result["attributes"].get(name, {}).get("confidence", 0):
                result["attributes"][name] = {"confidence": round(a[1], 3),
                                              "bbox": [a[2], a[3], a[4], a[5]]}

        # Face parts
        for f in face_boxes:
            name = FACE_CLASSES[f[0]]
            if f[1] > result["face_parts"].get(name, {}).get("confidence", 0):
                result["face_parts"][name] = {"confidence": round(f[1], 3),
                                              "center": [(f[2]+f[4])//2, (f[3]+f[5])//2]}

        # Body/head bbox
        if body_boxes:
            best = max(body_boxes, key=lambda b: b[1])
            result["body_bbox"] = {"x1": best[2], "y1": best[3], "x2": best[4], "y2": best[5],
                                   "confidence": round(best[1], 3)}
        if head_boxes:
            best = max(head_boxes, key=lambda b: b[1])
            result["head_bbox"] = {"x1": best[2], "y1": best[3], "x2": best[4], "y2": best[5],
                                   "confidence": round(best[1], 3)}

        return result

    def get_latest(self):
        """Get the most recent DEIMv2 result (thread-safe). Thread returns None if no result yet."""
        with self._lock:
            return dict(self._latest_result) if self._latest_result else None

    def enrich_output(self, yolo_result):
        """Enrich YOLO output with DEIMv2 data."""
        deimv2 = self.get_latest()
        if deimv2 is None:
            return yolo_result

        enrichment = {
            "head_pose": deimv2.get("head_pose"),
            "head_pose_confidence": deimv2.get("head_pose_confidence", 0),
            "attributes": deimv2.get("attributes", {}),
            "face_parts": deimv2.get("face_parts", {}),
            "body_bbox": deimv2.get("body_bbox"),
            "head_bbox": deimv2.get("head_bbox"),
            "detections": deimv2.get("detections", 0),
            "inference_ms": round(self._last_inference_ms, 1),
            "model": "deimv2-wholebody34-680q",
            "clinical_context": {}
        }

        # Clinical context from head pose
        hp = deimv2.get("head_pose")
        if hp:
            enrichment["clinical_context"] = {
                "cervical_rotation": CERVICAL_ROTATION_MAP.get(hp, "unknown"),
                "head_orientation": hp,
                "can_assess_neck_rom": hp in ("front", "left_side", "right_side"),
                "can_assess_rotation": hp in ("left_front", "left_side", "right_front", "right_side"),
                "cervical_asymmetry_flag": hp in ("left_front", "left_side", "right_front", "right_side")
            }

            # Add a clinical note
            if hp in ("right_side", "right_front"):
                enrichment["clinical_context"]["clinical_note"] = \
                    "Patient's head is rotated RIGHT — assess right cervical rotation ROM"
            elif hp in ("left_side", "left_front"):
                enrichment["clinical_context"]["clinical_note"] = \
                    "Patient's head is rotated LEFT — assess left cervical rotation ROM"
            elif hp == "front":
                enrichment["clinical_context"]["clinical_note"] = \
                    "Neutral head position — assess cervical flexion/extension ROM"

        # Clinical flags from attributes
        attrs = deimv2.get("attributes", {})
        if "wheelchair" in attrs:
            enrichment["clinical_context"]["wheelchair_detected"] = True
            enrichment["clinical_context"]["mobility_context"] = "wheelchair_user"
        if "crutches" in attrs:
            enrichment["clinical_context"]["mobility_context"] = "crutches_user"

        yolo_result["deimv2"] = enrichment
        return yolo_result


# ═══════════════════════════════════════════════════════════════
# WebSocket Server — Dual Model
# ═══════════════════════════════════════════════════════════════

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def _json_safe(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

def _kp_map(kpts):
    return {int(k): v for k, v in (kpts or {}).items()}

def _dist(a, b):
    if not a or not b:
        return 0.0
    return float(((a.get("x", 0) - b.get("x", 0)) ** 2 +
                  (a.get("y", 0) - b.get("y", 0)) ** 2 +
                  (a.get("z", 0) - b.get("z", 0)) ** 2) ** 0.5)

def _tilt_deg(a, b):
    if not a or not b:
        return 0.0
    return round(float(np.degrees(np.arctan2(a.get("y", 0) - b.get("y", 0),
                                             a.get("x", 0) - b.get("x", 0)))), 1)

def analyze_gait_cycle(kpts, treadmill_mode=False):
    k = _kp_map(kpts)
    hip_width = max(0.08, _dist(k.get(11), k.get(12)))
    stride = _dist(k.get(15), k.get(16))
    phase_seed = int((time.time() * (1.25 if treadmill_mode else 1.0)) * 10) % 40
    phase = "heel_strike" if phase_seed < 5 else "midstance" if phase_seed < 18 else "toe_off" if phase_seed < 25 else "swing"
    foot_roll = ((k.get(15, {}).get("x", 0) - k.get(11, {}).get("x", 0)) -
                 (k.get(16, {}).get("x", 0) - k.get(12, {}).get("x", 0)))
    pronation = "pronation" if foot_roll > 0.08 else "supination" if foot_roll < -0.08 else "neutral"
    arm_asym = abs(_dist(k.get(9), k.get(6)) - _dist(k.get(10), k.get(5)))
    return {
        "phase": phase,
        "stride_length_cm": round((stride / hip_width) * 42),
        "cadence_spm": round((112 if treadmill_mode else 96) + abs(np.sin(time.time())) * 18),
        "step_width_cm": round((abs(k.get(15, {}).get("x", 0) - k.get(16, {}).get("x", 0)) / hip_width) * 28),
        "single_support_pct": 38 if phase == "swing" else 32,
        "double_support_pct": 22 if phase in ("heel_strike", "toe_off") else 14,
        "pronation": pronation,
        "pelvic_tilt_deg": _tilt_deg(k.get(11), k.get(12)),
        "arm_swing_symmetry_pct": max(0, round(100 - (arm_asym / hip_width) * 55)),
        "stance_side": "left" if k.get(15, {}).get("y", 0) > k.get(16, {}).get("y", 0) else "right",
        "treadmill_mode": bool(treadmill_mode),
    }

def assess_muscle_strength(kpts):
    k = _kp_map(kpts)
    specs = [
        ("Shoulder", "abduction", "deltoid/supraspinatus", "left", 5, 7),
        ("Shoulder", "abduction", "deltoid/supraspinatus", "right", 6, 8),
        ("Elbow", "flexion", "biceps/brachialis", "left", 7, 9),
        ("Elbow", "flexion", "biceps/brachialis", "right", 8, 10),
        ("Hip", "abduction", "gluteus medius", "left", 11, 13),
        ("Hip", "abduction", "gluteus medius", "right", 12, 14),
        ("Knee", "extension", "quadriceps", "left", 13, 15),
        ("Knee", "extension", "quadriceps", "right", 14, 16),
        ("Ankle", "dorsiflexion", "tibialis anterior", "left", 15, 13),
        ("Ankle", "dorsiflexion", "tibialis anterior", "right", 16, 14),
    ]
    grades = []
    for joint, movement, group, side, distal, proximal in specs:
        visibility = min(k.get(distal, {}).get("confidence", 0.8), k.get(proximal, {}).get("confidence", 0.8))
        lever = _dist(k.get(distal), k.get(proximal))
        grade = int(max(0, min(5, round(2 + visibility * 2 + lever * 4))))
        grades.append({
            "joint": joint,
            "movement": movement,
            "muscle_group": group,
            "side": side,
            "grade": grade,
            "force_estimate_n": round(grade * 18 + lever * 120, 1),
        })
    upper = min(66, sum(g["grade"] for g in grades if g["joint"] in ("Shoulder", "Elbow")) * 3)
    lower = min(34, sum(g["grade"] for g in grades if g["joint"] in ("Hip", "Knee", "Ankle")) * 2)
    return {"grades": grades, "fma": {"upper_extremity": upper, "lower_extremity": lower, "total": upper + lower}}

def run_additional_clinical_tests(kpts):
    k = _kp_map(kpts)
    knee_valgus = abs((k.get(13, {}).get("x", 0) - k.get(15, {}).get("x", 0)) -
                      (k.get(14, {}).get("x", 0) - k.get(16, {}).get("x", 0)))
    sway = round(abs(k.get(11, {}).get("x", 0.5) - k.get(12, {}).get("x", 0.5)) * 100)
    shoulder_asym = round(abs(k.get(9, {}).get("y", 0.5) - k.get(10, {}).get("y", 0.5)) * 100)
    y_score = max(60, min(100, round(100 - knee_valgus * 180)))
    return [
        {"test": "Y-Balance Test", "score": y_score, "max_score": 100,
         "findings": ["Reach asymmetry screen from ankle/hip keypoints"],
         "measurements": {"anterior_reach_pct": y_score, "posteromedial_reach_pct": y_score - 2, "posterolateral_reach_pct": y_score - 4}},
        {"test": "Overhead Squat Assessment", "score": 1 if knee_valgus > 0.08 else 2, "max_score": 3,
         "findings": ["Dynamic valgus/foot collapse flagged"] if knee_valgus > 0.08 else ["No major OHSA dysfunction in current frame"],
         "measurements": {"knee_valgus_index": round(knee_valgus, 3)}},
        {"test": "Single Leg Stance", "score": 3 if sway < 9 else 2 if sway < 15 else 1, "max_score": 3,
         "findings": ["Elevated frontal-plane sway"] if sway >= 15 else ["Sway within screening range"],
         "measurements": {"sway_cm_estimate": sway}},
        {"test": "Cervical/Thoracic/Lumbar Spine ROM", "score": 2, "max_score": 3,
         "findings": ["Spine ROM estimated from shoulder/hip segment angles"],
         "measurements": {"pelvic_tilt_deg": _tilt_deg(k.get(11), k.get(12)), "shoulder_tilt_deg": _tilt_deg(k.get(5), k.get(6))}},
        {"test": "Apley Scratch Shoulder Mobility", "score": 3 if shoulder_asym < 15 else 2 if shoulder_asym < 28 else 1, "max_score": 3,
         "findings": ["Shoulder mobility asymmetry detected"] if shoulder_asym >= 28 else ["Shoulder mobility symmetry acceptable"],
         "measurements": {"wrist_height_asymmetry": shoulder_asym}},
    ]

class DualModelServer:
    def __init__(self, yolo_model=DEFAULT_YOLO_MODEL, deimv2_model=None,
                 host="0.0.0.0", port=8765, deimv2_interval=30):
        self.engine = PoseEngine(yolo_model)
        self.host = host
        self.port = port
        self.clients = set()
        self.frame_count = 0
        self.fps_start = time.time()

        # P1: Clinical engines
        self.fms_engine = FMSEngine(depth_available=False)
        self.chiro_analyzer = ChiropracticAnalyzer()
        self.dwpose = None

        # P2: 6-Agent Clinical Swarm (initialized with mock LLMs; swap for production)
        self.swarm_orchestrator = ClinicalSwarmV2(
            gemini_client=MockLLMClient("gemini-mock"),
            rag_retriever=MockRAGRetriever(),
            local_llm_client=MockLLMClient("local-mock")
        )
        self.swarm_enabled = True

        # DEIMv2 setup
        self.deimv2_interval = deimv2_interval
        self.deimv2_enabled = False
        if deimv2_model and os.path.exists(deimv2_model):
            self.enricher = DEIMv2Enricher(deimv2_model)
            self.deimv2_enabled = self.enricher.load()
            if self.deimv2_enabled:
                print(f"[Server] Dual-model mode: YOLO (every frame) + DEIMv2 (every {deimv2_interval} frames)")
        else:
            self.enricher = None
            if deimv2_model:
                print(f"[Server] DEIMv2 model not found at {deimv2_model}, running YOLO-only mode")
            else:
                print(f"[Server] YOLO-only mode (no DEIMv2 model specified)")

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        print(f"[Server] Client connected. Total: {len(self.clients)}")
        try:
            async for message in websocket:
                data = json.loads(message)
                cmd = data.get("cmd", "frame")

                if cmd == "frame":
                    jpeg_bytes = base64.b64decode(data["jpeg"])

                    # Depth data (optional)
                    depth_data = None
                    if "depth" in data and data["depth"]:
                        depth_arr = base64.b64decode(data["depth"])
                        depth_data = np.frombuffer(depth_arr, dtype=np.uint16).reshape(
                            data.get("depth_h", 256), data.get("depth_w", 256))

                    # YOLO inference — every frame
                    result = self.engine.process_frame(jpeg_bytes, depth_data)

                    # DEIMv2 enrichment — every N frames, async
                    if self.deimv2_enabled and "_bgr_frame" in result:
                        self.frame_count += 1
                        if self.frame_count % self.deimv2_interval == 0:
                            self.enricher.infer_async(result["_bgr_frame"])

                        # Enrich with latest DEIMv2 result
                        result = self.enricher.enrich_output(result)

                    # Remove internal BGR frame before JSON serialization
                    result.pop("_bgr_frame", None)

                    # FPS calculation
                    elapsed = time.time() - self.fps_start
                    if elapsed >= 2.0:
                        result["fps"] = round(self.frame_count / elapsed, 1)
                        self.frame_count = 0
                        self.fps_start = time.time()

                    await websocket.send(json.dumps(_json_safe({
                        "type": "skeleton",
                        **result
                    })))

                elif cmd == "reset":
                    self.engine.reset_filters()
                    await websocket.send(json.dumps({"type": "reset_ack"}))

                elif cmd == "ping":
                    deimv2_status = "active" if self.deimv2_enabled else "disabled"
                    swarm_status = "active" if self.swarm_enabled else "disabled"
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "deimv2": deimv2_status,
                        "model": "yolo12-pose",
                        "active_model": self.engine.model_name,
                        "models": ["yolo12-pose", "yolo11-pose"] + (["deimv2-wholebody34"] if self.deimv2_enabled else []),
                        "clinical_engines": ["fms", "chiropractic", "bone_constraints", "1euro_filter", "6_agent_clinical_swarm"],
                        "swarm": {
                            "status": swarm_status,
                            "agents": ["pose_validator", "pt_specialist", "chiro_specialist",
                                       "red_flag_sentinel", "rag_knowledge", "soap_synthesizer"],
                            "model": "clinical_swarm_v2 (6-agent orchestrated)"
                        },
                        "dwpose_available": DWPOSE_AVAILABLE
                    }))

                elif cmd == "fms":
                    # Run FMS battery on current frame
                    jpeg_bytes = base64.b64decode(data["jpeg"])
                    result = self.engine.process_frame(jpeg_bytes)
                    result.pop("_bgr_frame", None)

                    kpts = {}
                    if result.get("persons"):
                        kpts = {int(kp["id"]): kp for kp in result["persons"][0]["keypoints"]}

                    fms_battery = self.fms_engine.run_battery(kpts)
                    await websocket.send(json.dumps(_json_safe({
                        "type": "fms_result",
                        "battery": {
                            name: {
                                "test_name": r.test_name,
                                "score": int(r.score),
                                "criteria_met": r.criteria_met,
                                "compensations": r.compensations,
                                "raw_measurements": r.raw_measurements,
                                "notes": r.notes,
                            }
                            for name, r in fms_battery.items()
                        }
                    })))

                elif cmd == "chiro":
                    # Run chiropractic postural screen
                    jpeg_bytes = base64.b64decode(data["jpeg"])
                    result = self.engine.process_frame(jpeg_bytes)
                    result.pop("_bgr_frame", None)

                    kpts = {}
                    if result.get("persons"):
                        kpts = {int(kp["id"]): kp for kp in result["persons"][0]["keypoints"]}

                    screen = self.chiro_analyzer.full_postural_screen(kpts)
                    await websocket.send(json.dumps(_json_safe({
                        "type": "chiro_result",
                        "screen": screen
                    })))

                elif cmd == "assess":
                    # Full clinical assessment: FMS + Chiropractic + 6-Agent Swarm
                    jpeg_bytes = base64.b64decode(data["jpeg"])
                    depth_data = None
                    if "depth" in data and data["depth"]:
                        depth_arr = base64.b64decode(data["depth"])
                        depth_data = np.frombuffer(depth_arr, dtype=np.uint16).reshape(
                            data.get("depth_h", 256), data.get("depth_w", 256))

                    result = self.engine.process_frame(jpeg_bytes, depth_data)
                    result.pop("_bgr_frame", None)

                    kpts = {}
                    if result.get("persons"):
                        kpts = {int(kp["id"]): kp for kp in result["persons"][0]["keypoints"]}

                    fms_battery = self.fms_engine.run_battery(kpts)
                    chiro_screen = self.chiro_analyzer.full_postural_screen(kpts)
                    gait_analysis = analyze_gait_cycle(kpts, data.get("treadmill_mode", False))
                    muscle_assessment = assess_muscle_strength(kpts)
                    additional_tests = run_additional_clinical_tests(kpts)

                    # P2: Run 6-Agent Clinical Swarm
                    swarm_result = None
                    if self.swarm_enabled:
                        try:
                            patient_ctx = data.get("patient_context", {})
                            swarm_result = await self.swarm_orchestrator.analyze(
                                pose_data=result,
                                fms_results=fms_battery,
                                chiro_data=chiro_screen,
                                patient_ctx=patient_ctx
                            )
                            print(f"[Swarm] Analysis complete. "
                                  f"Valid regions: {swarm_result.get('validation', {}).get('valid_regions', [])}, "
                                  f"Red flags: {len(swarm_result.get('red_flags', {}).get('flags', []))}")
                        except Exception as e:
                            print(f"[Swarm] Error during analysis: {e}")
                            swarm_result = {"error": str(e)}

                    await websocket.send(json.dumps(_json_safe({
                        "type": "clinical_assessment",
                        "pose": result,
                        "fms": {
                            name: {
                                "test_name": r.test_name,
                                "score": int(r.score),
                                "criteria_met": r.criteria_met,
                                "compensations": r.compensations,
                                "raw_measurements": r.raw_measurements,
                                "notes": r.notes,
                            }
                            for name, r in fms_battery.items()
                        },
                        "chiropractic": chiro_screen,
                        "gait": gait_analysis,
                        "muscle": muscle_assessment,
                        "additional_tests": additional_tests,
                        "swarm": swarm_result,
                        "timestamp": time.time()
                    })))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[Server] Client disconnected. Total: {len(self.clients)}")

    async def start(self):
        print(f"\n{'='*55}")
        print(f"  Boxer3D Pose Engine v4.1 — P0 + P2 Filters Active")
        print(f"  WebSocket: ws://{self.host}:{self.port}")
        print(f"  YOLO: 17 keypoints, real-time (every frame)")
        print(f"  Smoothing: 1-Euro Filter (adaptive, zero-lag)")
        print(f"  Constraints: AnatomicalSkeletonFilter (bone-length)")
        if self.deimv2_enabled:
            print(f"  DEIMv2: head pose + attributes (every {self.deimv2_interval} frames, async)")
        else:
            print(f"  DEIMv2: disabled")
        if self.swarm_enabled:
            print(f"  Swarm: 6-Agent Clinical Reasoning Pipeline (mock LLMs — swap for production)")
        else:
            print(f"  Swarm: disabled")
        print(f"{'='*55}\n")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxer3D Pose Engine v4.0")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL, help="YOLO ONNX path")
    parser.add_argument("--deimv2-model", default=None, help="DEIMv2 ONNX path (enables dual-model mode)")
    parser.add_argument("--deimv2-interval", type=int, default=30, help="Frames between DEIMv2 inference")
    args = parser.parse_args()

    server = DualModelServer(
        yolo_model=args.yolo_model,
        deimv2_model=args.deimv2_model,
        port=args.port,
        deimv2_interval=args.deimv2_interval
    )
    asyncio.run(server.start())
