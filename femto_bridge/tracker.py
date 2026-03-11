"""
Femto Mega Body Tracker
Logic for body tracking using MediaPipe and Orbbec SDK.
"""

import logging
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime

# SDK imports with fallback
SDK_AVAILABLE = False
try:
    from pyorbbecsdk import (
        Pipeline, Config, OBSensorType, OBFormat,
        OBAlignMode, VideoStreamProfile
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

class FemtoMegaBodyTracker:
    """Body tracking implementation for Femto Mega using MediaPipe Tasks + Depth"""

    def __init__(self):
        self.pipeline = None
        self.config = None
        self.is_started = False
        self.landmarker = None

        # Initialize MediaPipe Pose Landmarker
        try:
            logger.info("🧠 Initializing MediaPipe Pose Landmarker...")

            # Path to model file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, 'models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            model_path = os.path.join(models_dir, 'pose_landmarker_full.task')

            if not os.path.exists(model_path):
                logger.info(f"⬇️  Downloading MediaPipe model to {model_path}...")
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
                try:
                    urllib.request.urlretrieve(url, model_path)
                    logger.info("✅ Model downloaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to download model: {e}")
                    raise

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info("✅ MediaPipe Pose Landmarker initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")

    def init_camera(self):
        """Initialize Femto Mega camera with depth and color streams"""
        if not SDK_AVAILABLE:
            logger.error("❌ pyorbbecsdk not available")
            return False

        try:
            logger.info("📷 Initializing Femto Mega camera...")

            self.pipeline = Pipeline()
            self.config = Config()

            # Enable depth stream
            self.config.enable_stream(
                OBSensorType.DEPTH_SENSOR,
                640, 576,  # Resolution
                OBFormat.Y16,
                30  # FPS
            )

            # Enable color stream
            self.config.enable_stream(
                OBSensorType.COLOR_SENSOR,
                1920, 1080,
                OBFormat.RGB,
                30
            )

            # Enable alignment (align depth to color)
            # This ensures depth map matches RGB image pixel-for-pixel
            self.config.set_align_mode(OBAlignMode.ALIGN_D2C_SW_MODE)

            # Start pipeline
            self.pipeline.start(self.config)
            self.is_started = True

            logger.info("✅ Femto Mega initialized successfully")
            logger.info("   - Depth: 640x576 @ 30fps")
            logger.info("   - Color: 1920x1080 @ 30fps")
            logger.info("   - Alignment: Depth-to-Color")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize camera: {e}")
            return False

    def get_frames(self):
        """Get synchronized depth and color frames"""
        if not self.is_started:
            return None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            return frames
        except Exception as e:
            logger.error(f"❌ Error getting frames: {e}")
            return None

    def _deproject_pixel_to_point(self, u, v, depth, width=1920, height=1080):
        """
        Deproject 2D pixel to 3D point using camera intrinsics
        (Approximated for Femto Mega RGB camera if intrinsics unavailable)
        """
        # Approximate intrinsics for Femto Mega RGB (FOV ~90 degrees horizontal)
        # fx = fy = width / (2 * tan(FOV_H / 2))
        # For 90 deg FOV, tan(45) = 1, so fx = width / 2

        fx = width / 2.0
        fy = fx  # Square pixels assumption
        cx = width / 2.0
        cy = height / 2.0

        # X = (u - cx) * depth / fx
        # Y = (v - cy) * depth / fy
        # Z = depth

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return {'x': x, 'y': y, 'z': z}

    def _map_mediapipe_to_k4abt(self, results, depth_image):
        """Map MediaPipe landmarks to Azure Kinect Body Tracking skeleton"""
        if not results.pose_landmarks:
            return None

        h, w = depth_image.shape
        # Take the first detected person
        landmarks = results.pose_landmarks[0]

        # Helper to get 3D point for a MediaPipe landmark
        def get_joint_3d(mp_index):
            lm = landmarks[mp_index]
            px = int(lm.x * w)
            py = int(lm.y * h)

            # Clamp to image bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            # Sample depth (mm)
            d = float(depth_image[py, px])

            # If invalid depth, try simple neighborhood search
            if d == 0:
                # 3x3 kernel check
                neighborhood = depth_image[max(0, py-1):min(h, py+2), max(0, px-1):min(w, px+2)]
                valid = neighborhood[neighborhood > 0]
                if valid.size > 0:
                    d = float(np.median(valid))
                else:
                    # Fallback: estimate from MediaPipe relative Z (not real world scale)
                    # This is tricky, so we might just use last known or neighbor
                    d = 1500.0 # Default guess if absolutely nothing found

            return self._deproject_pixel_to_point(px, py, d, w, h)

        # Joint mapping dictionary
        joints = {}

        # 32 Joints of K4ABT
        # MediaPipe Indices:
        # 11: left_shoulder, 12: right_shoulder, 23: left_hip, 24: right_hip
        # 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist
        # 25: left_knee, 26: right_knee, 27: left_ankle, 28: right_ankle

        # Basic limb joints (direct mapping)
        mapping = {
            'SHOULDER_LEFT': 11, 'SHOULDER_RIGHT': 12,
            'ELBOW_LEFT': 13, 'ELBOW_RIGHT': 14,
            'WRIST_LEFT': 15, 'WRIST_RIGHT': 16,
            'HIP_LEFT': 23, 'HIP_RIGHT': 24,
            'KNEE_LEFT': 25, 'KNEE_RIGHT': 26,
            'ANKLE_LEFT': 27, 'ANKLE_RIGHT': 28,
            'EYE_LEFT': 2, 'EYE_RIGHT': 5,
            'EAR_LEFT': 7, 'EAR_RIGHT': 8,
            'NOSE': 0
        }

        # Extract direct mappings
        for name, idx in mapping.items():
            joints[name] = {
                'position': get_joint_3d(idx),
                'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, # Identity for now
                'confidence': 'HIGH' if landmarks[idx].visibility > 0.5 else 'LOW'
            }

        # Computed joints (Approximations)

        # PELVIS: Midpoint of Hips
        hl = joints['HIP_LEFT']['position']
        hr = joints['HIP_RIGHT']['position']
        pelvis = {
            'x': (hl['x'] + hr['x']) / 2,
            'y': (hl['y'] + hr['y']) / 2,
            'z': (hl['z'] + hr['z']) / 2
        }
        joints['PELVIS'] = {
            'position': pelvis,
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'MEDIUM'
        }

        # NECK: Midpoint of Shoulders
        sl = joints['SHOULDER_LEFT']['position']
        sr = joints['SHOULDER_RIGHT']['position']
        neck = {
            'x': (sl['x'] + sr['x']) / 2,
            'y': (sl['y'] + sr['y']) / 2,
            'z': (sl['z'] + sr['z']) / 2
        }
        joints['NECK'] = {
            'position': neck,
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'MEDIUM'
        }

        # SPINE_CHEST: Interpolate 30% from Neck to Pelvis
        joints['SPINE_CHEST'] = {
            'position': {
                'x': neck['x'] * 0.7 + pelvis['x'] * 0.3,
                'y': neck['y'] * 0.7 + pelvis['y'] * 0.3,
                'z': neck['z'] * 0.7 + pelvis['z'] * 0.3,
            },
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'MEDIUM'
        }

        # SPINE_NAVAL: Interpolate 70% from Neck to Pelvis
        joints['SPINE_NAVAL'] = {
            'position': {
                'x': neck['x'] * 0.3 + pelvis['x'] * 0.7,
                'y': neck['y'] * 0.3 + pelvis['y'] * 0.7,
                'z': neck['z'] * 0.3 + pelvis['z'] * 0.7,
            },
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'MEDIUM'
        }

        # HEAD: Midpoint of Ears
        el = joints['EAR_LEFT']['position']
        er = joints['EAR_RIGHT']['position']
        joints['HEAD'] = {
            'position': {
                'x': (el['x'] + er['x']) / 2,
                'y': (el['y'] + er['y']) / 2,
                'z': (el['z'] + er['z']) / 2,
            },
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'MEDIUM'
        }

        # CLAVICLES (Approximate)
        joints['CLAVICLE_LEFT'] = joints['SHOULDER_LEFT'] # Close enough
        joints['CLAVICLE_RIGHT'] = joints['SHOULDER_RIGHT']

        # HANDS and THUMBS
        # Use MediaPipe specific landmarks
        # 19: index_left, 20: index_right, 21: thumb_left, 22: thumb_right
        # 17: pinky_left, 18: pinky_right

        joints['HAND_LEFT'] = {
            'position': get_joint_3d(19), # Index finger base approx
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'HIGH'
        }
        joints['HAND_RIGHT'] = {
            'position': get_joint_3d(20),
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'HIGH'
        }

        joints['THUMB_LEFT'] = {
            'position': get_joint_3d(21),
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'HIGH'
        }
        joints['THUMB_RIGHT'] = {
            'position': get_joint_3d(22),
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'HIGH'
        }

        joints['HANDTIP_LEFT'] = joints['HAND_LEFT'] # Approx
        joints['HANDTIP_RIGHT'] = joints['HAND_RIGHT'] # Approx

        # FEET
        joints['FOOT_LEFT'] = {
            'position': get_joint_3d(31), # Left foot index
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'HIGH'
        }
        joints['FOOT_RIGHT'] = {
            'position': get_joint_3d(32), # Right foot index
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
            'confidence': 'HIGH'
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'body_id': 0,
            'joints': joints,
            'simulation': False,
            'has_real_depth': True
        }

    def extract_skeleton_from_depth(self, frames):
        """
        Extract skeleton data using MediaPipe Pose + Depth Map
        """
        if frames is None or self.landmarker is None:
            return None

        try:
            # 1. Get color and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                return None

            # 2. Convert to numpy arrays
            # Color is RGB
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_data = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))

            # Depth is Y16 (uint16)
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

            # 3. Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color_data))

            # 4. Process with MediaPipe Pose Landmarker
            timestamp_ms = int(datetime.now().timestamp() * 1000)
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            # 5. Map to skeleton
            if results.pose_landmarks:
                skeleton = self._map_mediapipe_to_k4abt(results, depth_data)
                return skeleton
            else:
                return None

        except Exception as e:
            logger.error(f"❌ Error extracting skeleton: {e}")
            return None

    def generate_simulated_skeleton(self):
        """Generate simulated skeleton for testing"""
        import random
        import math

        time = datetime.now().timestamp()
        squat_phase = (math.sin(time * 0.5) + 1) / 2

        joints = {}
        joint_names = [
            'PELVIS', 'SPINE_NAVAL', 'SPINE_CHEST', 'NECK',
            'CLAVICLE_LEFT', 'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT',
            'HAND_LEFT', 'HANDTIP_LEFT', 'THUMB_LEFT',
            'CLAVICLE_RIGHT', 'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',
            'HAND_RIGHT', 'HANDTIP_RIGHT', 'THUMB_RIGHT',
            'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT', 'FOOT_LEFT',
            'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT', 'FOOT_RIGHT',
            'HEAD', 'NOSE', 'EYE_LEFT', 'EAR_LEFT', 'EYE_RIGHT', 'EAR_RIGHT'
        ]

        for i, name in enumerate(joint_names):
            y_offset = 0
            if 'PELVIS' in name or 'HIP' in name or 'KNEE' in name:
                y_offset = -squat_phase * 300

            joints[name] = {
                'position': {
                    'x': random.uniform(-200, 200) + (i * 10),
                    'y': 500 + y_offset + (i * 20),
                    'z': 1500 + random.uniform(-50, 50)
                },
                'orientation': {
                    'w': 1.0,
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0
                },
                'confidence': 'HIGH' if random.random() > 0.1 else 'MEDIUM'
            }

        return {
            'timestamp': datetime.now().isoformat(),
            'body_id': 0,
            'joints': joints,
            'simulation': not SDK_AVAILABLE,
            'has_real_depth': SDK_AVAILABLE and self.is_started
        }

    def stop(self):
        """Stop camera pipeline"""
        if self.is_started and self.pipeline:
            try:
                self.pipeline.stop()
                logger.info("🔌 Camera pipeline stopped")
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")
        self.is_started = False
