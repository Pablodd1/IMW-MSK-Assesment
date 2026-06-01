#!/usr/bin/env python3
"""
Boxer3D Pose Engine v4.0 — Dual-Model Pipeline
================================================
YOLO11-pose (real-time 17-keypoint skeleton) + DEIMv2 (async head pose enrichment)

Architecture:
  Camera → WebSocket → YOLO11-pose (every frame, ~15-30 FPS)
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

# ─── YOLO11-pose via OpenCV DNN ───
YOLO11_POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx"

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

# ─── Kalman Filter ───
class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.x = 0.0
        self.p = 1.0
        self.q = process_noise
        self.r = measurement_noise
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x


# ═══════════════════════════════════════════════════════════════
# YOLO11-Pose Engine (unchanged core — production)
# ═══════════════════════════════════════════════════════════════

class PoseEngine:
    COCO_KEYPOINTS = 17
    INPUT_SIZE = 640

    def __init__(self, model_path="yolo11n-pose.onnx", confidence=0.5):
        model_path = download_model(YOLO11_POSE_URL, model_path)
        print(f"[YOLO] Loading ONNX: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.confidence = confidence
        self.kalman_filters = {}

    def get_kalman(self, kid, axis):
        key = (kid, axis)
        if key not in self.kalman_filters:
            self.kalman_filters[key] = KalmanFilter1D()
        return self.kalman_filters[key]

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

                sx = self.get_kalman(kid, 'x').update(kp_x / w)
                sy = self.get_kalman(kid, 'y').update(kp_y / h)
                sz = self.get_kalman(kid, 'z').update(z)

                keypoints_3d.append({
                    "id": kid,
                    "x": round(sx, 4),
                    "y": round(sy, 4),
                    "z": round(sz, 4),
                    "confidence": round(kp_conf, 3),
                    "depth_valid": z > 0
                })

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
            "model": "yolo11-pose",
            "depth_available": depth_data is not None,
            "_bgr_frame": img  # internal — removed before JSON serialize
        }

    def reset_filters(self):
        self.kalman_filters = {}


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

class DualModelServer:
    def __init__(self, yolo_model="yolo11n-pose.onnx", deimv2_model=None,
                 host="0.0.0.0", port=8765, deimv2_interval=30):
        self.engine = PoseEngine(yolo_model)
        self.host = host
        self.port = port
        self.clients = set()
        self.frame_count = 0
        self.fps_start = time.time()

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
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "deimv2": deimv2_status,
                        "models": ["yolo11-pose"] + (["deimv2-wholebody34"] if self.deimv2_enabled else [])
                    }))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[Server] Client disconnected. Total: {len(self.clients)}")

    async def start(self):
        print(f"\n{'='*55}")
        print(f"  Boxer3D Pose Engine v4.0 — Dual Model")
        print(f"  WebSocket: ws://{self.host}:{self.port}")
        print(f"  YOLO: 17 keypoints, real-time (every frame)")
        if self.deimv2_enabled:
            print(f"  DEIMv2: head pose + attributes (every {self.deimv2_interval} frames, async)")
        else:
            print(f"  DEIMv2: disabled")
        print(f"{'='*55}\n")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxer3D Pose Engine v4.0")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--yolo-model", default="yolo11n-pose.onnx", help="YOLO ONNX path")
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
