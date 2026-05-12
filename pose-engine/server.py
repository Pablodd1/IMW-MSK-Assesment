#!/usr/bin/env python3
"""
Boxer3D Pose Engine — YOLO11-pose real-time 3D skeleton tracking
WebSocket server that receives camera frames, runs YOLO11-pose inference,
fuses depth data (LiDAR/Femto Mega), and streams 3D keypoints to the frontend.

Usage:
  python3 pose-engine/server.py [--port 8765] [--model yolov11n-pose.pt]
"""
import asyncio, json, base64, sys, os, time, argparse, urllib.request, zipfile
import numpy as np
import cv2

# ─── YOLO11-pose via OpenCV DNN (ONNX) — no torch dependency ───
YOLO11_POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx"

def download_model(path):
    """Download YOLO11-pose ONNX model if not present."""
    if os.path.exists(path):
        return path
    print(f"[Boxer3D] Downloading YOLO11-pose ONNX model (~5MB)...")
    urllib.request.urlretrieve(YOLO11_POSE_URL, path)
    print(f"[Boxer3D] Model saved: {path}")
    return path

try:
    import websockets
except ImportError:
    os.system("pip3 install --user --break-system-packages websockets")
    import websockets

# ─── Kalman filter for smoothing joint positions ───
class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.x = 0.0      # state
        self.p = 1.0      # estimate covariance
        self.q = process_noise
        self.r = measurement_noise
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        # Predict
        self.p += self.q
        # Update
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

class PoseEngine:
    COCO_KEYPOINTS = 17
    INPUT_SIZE = 640

    def __init__(self, model_path="yolo11n-pose.onnx", confidence=0.5):
        model_path = download_model(model_path)
        print(f"[Boxer3D] Loading ONNX model: {model_path}")
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
        """Process a JPEG frame, return 3D keypoints with confidence."""
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Failed to decode image"}

        h, w = img.shape[:2]

        # Preprocess: letterbox to 640x640
        input_blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.INPUT_SIZE, self.INPUT_SIZE),
                                            swapRB=True, crop=False)

        # Inference
        self.net.setInput(input_blob)
        outputs = self.net.forward()
        # YOLO11-pose output shape: [1, 56, 8400] — 56 = 4(bbox) + 1(conf) + 17*3(kpt_x,kpt_y,kpt_conf)

        keypoints_list = []
        output = outputs[0]  # shape: (56, 8400)
        output = output.transpose()  # (8400, 56)

        # Scale factors (letterbox may have padding)
        scale_x = w / self.INPUT_SIZE
        scale_y = h / self.INPUT_SIZE

        for detection in output:
            scores = detection[4:5]  # single class score (person)
            score = float(scores.max())

            if score < self.confidence:
                continue

            # Bounding box (cx, cy, bw, bh)
            cx, cy, bw, bh = detection[:4]
            x1 = (cx - bw/2) * scale_x
            y1 = (cy - bh/2) * scale_y
            x2 = (cx + bw/2) * scale_x
            y2 = (cy + bh/2) * scale_y

            # Keypoints: 17 keypoints, each with (x, y, conf)
            kp_start = 5  # after bbox(4) + class_score(1)
            keypoints_3d = []

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
                        if d > 0 and d < 5000:
                            z = d / 1000.0

                # Kalman smooth
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

            # Only track the primary person
            if len(keypoints_list) >= 1:
                break

        return {
            "persons": keypoints_list,
            "frame_w": w,
            "frame_h": h,
            "fps": 0,
            "model": "yolo11-pose",
            "depth_available": depth_data is not None
        }

    def reset_filters(self):
        self.kalman_filters = {}

# ─── YOLO11 keypoint names ───
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

CONNECTIONS = [
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 6),            # shoulders
    (5, 11), (6, 12), # torso
    (11, 12),          # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
]

# ─── WebSocket Server ───
class Boxer3DServer:
    def __init__(self, model_path="yolo11n-pose.onnx", host="0.0.0.0", port=8765):
        self.engine = PoseEngine(model_path)
        self.host = host
        self.port = port
        self.clients = set()
        self.frame_count = 0
        self.fps_start = time.time()

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        print(f"[Boxer3D] Client connected. Total: {len(self.clients)}")
        try:
            async for message in websocket:
                data = json.loads(message)
                cmd = data.get("cmd", "frame")

                if cmd == "frame":
                    # Decode base64 JPEG
                    jpeg_bytes = base64.b64decode(data["jpeg"])
                    depth_data = None
                    if "depth" in data and data["depth"]:
                        depth_arr = base64.b64decode(data["depth"])
                        depth_data = np.frombuffer(depth_arr, dtype=np.uint16).reshape(
                            data.get("depth_h", 256), data.get("depth_w", 256)
                        )

                    result = self.engine.process_frame(jpeg_bytes, depth_data)
                    self.frame_count += 1

                    # Calculate FPS
                    elapsed = time.time() - self.fps_start
                    if elapsed >= 2.0:
                        result["fps"] = round(self.frame_count / elapsed, 1)
                        self.frame_count = 0
                        self.fps_start = time.time()

                    await websocket.send(json.dumps({
                        "type": "skeleton",
                        **result
                    }))

                elif cmd == "reset":
                    self.engine.reset_filters()
                    await websocket.send(json.dumps({"type": "reset_ack"}))

                elif cmd == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[Boxer3D] Client disconnected. Total: {len(self.clients)}")

    async def start(self):
        print(f"\n{'='*50}")
        print(f"  Boxer3D Pose Engine")
        print(f"  YOLO11-pose + Kalman Filter + Depth Fusion")
        print(f"  WebSocket: ws://{self.host}:{self.port}")
        print(f"  Keypoints: {len(KEYPOINT_NAMES)}")
        print(f"{'='*50}\n")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxer3D Pose Engine")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="YOLO model path")
    args = parser.parse_args()

    server = Boxer3DServer(model_path=args.model, port=args.port)
    asyncio.run(server.start())
