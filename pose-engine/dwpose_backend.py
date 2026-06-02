"""
DWPose Backend — Wholebody 133-keypoint pose estimation via ONNX.
Run on-demand for clinical-precision captures (not real-time preview).
YOLO11-pose stays for real-time stream; DWPose triggers on "Assess" button.

Model: dw-ll_ucoco_384.onnx (wholebody 133 keypoints)
Download: https://download.openmmlab.com/mmpose/v1/projects/dwpose/
"""
import cv2
import numpy as np
from typing import List, Dict, Optional

class DWPoseBackend:
    """
    Wholebody 133-keypoint pose estimation using DWPose ONNX Runtime.
    Keypoint format includes body (0-16), face (17-90), hands (91-132).
    """

    # COCO body keypoints (indices 0-16)
    BODY_KPS = {
        "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
    }

    # Hand keypoints (indices 91-132) — 21 per hand
    LEFT_HAND_START = 91
    RIGHT_HAND_START = 112

    # Feet keypoints for balance / FMS
    LEFT_HEEL = 121
    RIGHT_HEEL = 122
    LEFT_BIG_TOE = 117
    RIGHT_BIG_TOE = 118

    def __init__(self, model_path: str, input_size=(384, 288), conf_threshold=0.3):
        if not cv2.__version__ >= "4.5":
            raise RuntimeError("OpenCV >= 4.5 required for ONNX DNN")
        self.model_path = model_path
        self.input_size = input_size  # (W, H)
        self.conf_threshold = conf_threshold
        self.net = cv2.dnn.readNetFromONNX(model_path)
        # Prefer CUDA if available, fallback CPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # If CUDA fails, OpenCV auto-falls back; catch with try/except if needed
        print(f"[DWPose] Loaded: {model_path} | Input: {input_size}")

    def preprocess(self, img: np.ndarray) -> tuple:
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, self.input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(
            img_rgb,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )
        return blob, (w, h)

    def postprocess(
        self,
        outputs,
        img_w: int,
        img_h: int
    ) -> List[Dict]:
        """
        DWPose ONNX exports vary. This handles the common
        [1, K, 3] output format: [x, y, confidence] per keypoint.
        """
        kpts_array = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        # Squeeze batch dim if present
        if kpts_array.ndim == 3 and kpts_array.shape[0] == 1:
            kpts_array = kpts_array[0]

        scale_x = img_w / self.input_size[0]
        scale_y = img_h / self.input_size[1]

        keypoints = []
        for i in range(kpts_array.shape[0]):
            x = float(kpts_array[i, 0]) * scale_x
            y = float(kpts_array[i, 1]) * scale_y
            c = float(kpts_array[i, 2])
            if c > self.conf_threshold:
                keypoints.append({
                    "id": i,
                    "x": round(x / img_w, 4),
                    "y": round(y / img_h, 4),
                    "confidence": round(c, 3),
                    "depth_valid": False,
                    "z": 0.0
                })
        return keypoints

    def infer(self, img: np.ndarray) -> List[Dict]:
        """Run inference and return normalized keypoint list."""
        blob, (w, h) = self.preprocess(img)
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self.postprocess(outputs, w, h)

    def get_hand_keypoints(self, keypoints: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract hand keypoints from wholebody output."""
        left_hand = [kp for kp in keypoints if self.LEFT_HAND_START <= kp["id"] < self.RIGHT_HAND_START]
        right_hand = [kp for kp in keypoints if self.RIGHT_HAND_START <= kp["id"] < self.RIGHT_HAND_START + 21]
        return {"left": left_hand, "right": right_hand}

    def get_foot_keypoints(self, keypoints: List[Dict]) -> Dict[str, Optional[Dict]]:
        """Extract heel and toe keypoints for balance analysis."""
        by_id = {kp["id"]: kp for kp in keypoints}
        return {
            "left_heel": by_id.get(self.LEFT_HEEL),
            "right_heel": by_id.get(self.RIGHT_HEEL),
            "left_big_toe": by_id.get(self.LEFT_BIG_TOE),
            "right_big_toe": by_id.get(self.RIGHT_BIG_TOE),
        }
