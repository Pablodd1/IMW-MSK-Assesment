"""
Body tracking implementation for Femto Mega
Supports:
1. Azure Kinect Body Tracking SDK (via pyk4a) - Priority
2. MediaPipe Pose + Depth (via pyorbbecsdk) - Fallback
3. Simulation - Final fallback
"""

import logging
import os
import sys
import numpy as np
import math
import random
import time
from datetime import datetime
import urllib.request

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SDK Imports with fallback
SDK_AVAILABLE = False
K4A_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
    SDK_AVAILABLE = True
except ImportError:
    pass

try:
    # Import pyk4a inside try block to avoid failure if not installed
    from pyk4a import PyK4A, Config as K4AConfig, ColorResolution, DepthMode, WiredSyncMode
    from pyk4a import BodyTracker
    K4A_AVAILABLE = True
except ImportError:
    pass

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass


class FemtoMegaTracker:
    """
    Unified body tracker for Femto Mega.
    Prioritizes Azure Kinect Body Tracking, falls back to MediaPipe, then Simulation.
    """

    def __init__(self):
        self.pipeline = None
        self.config = None
        self.k4a = None
        self.tracker = None
        self.mp_landmarker = None

        self.is_running = False
        self.use_k4a = False
        self.use_mediapipe = False
        self.simulation = False

        # Initialize MediaPipe if needed (lazy init is better but we can do it here)
        if MEDIAPIPE_AVAILABLE and not K4A_AVAILABLE:
            self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe Pose Landmarker"""
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
                    return

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            self.mp_landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info("✅ MediaPipe Pose Landmarker initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")

    def start(self):
        """Initialize and start the best available tracking method"""
        if self.is_running:
            return True

        # 1. Try Azure Kinect Body Tracking (pyk4a)
        if K4A_AVAILABLE:
            try:
                logger.info("📷 Initializing Femto Mega in K4A mode...")
                self.k4a = PyK4A(
                    K4AConfig(
                        color_resolution=ColorResolution.RES_720P,
                        depth_mode=DepthMode.NFOV_UNBINNED,
                        camera_fps=30,
                        wired_sync_mode=WiredSyncMode.STANDALONE,
                    )
                )
                self.k4a.start()

                # Initialize Body Tracker
                logger.info("✅ Camera started. Initializing Body Tracker...")
                self.tracker = BodyTracker(self.k4a.calibration)

                self.use_k4a = True
                self.is_running = True
                logger.info("✅ Azure Kinect Body Tracking initialized successfully")
                return True

            except Exception as e:
                logger.error(f"❌ Failed to initialize K4A/Body Tracking: {e}")
                self.k4a = None
                self.tracker = None

        # 2. Try Orbbec SDK + MediaPipe
        if SDK_AVAILABLE:
            try:
                logger.info("📷 Initializing Femto Mega with Orbbec SDK...")
                self.pipeline = Pipeline()
                self.config = Config()

                # Enable depth stream
                self.config.enable_stream(
                    OBSensorType.DEPTH_SENSOR,
                    640, 576, OBFormat.Y16, 30
                )

                # Enable color stream
                self.config.enable_stream(
                    OBSensorType.COLOR_SENSOR,
                    1920, 1080, OBFormat.RGB, 30
                )

                # Align depth to color
                self.config.set_align_mode(OBAlignMode.ALIGN_D2C_SW_MODE)

                self.pipeline.start(self.config)
                self.is_running = True

                if self.mp_landmarker:
                    self.use_mediapipe = True
                    logger.info("✅ Orbbec SDK + MediaPipe initialized successfully")
                else:
                    self._init_mediapipe()
                    if self.mp_landmarker:
                         self.use_mediapipe = True
                         logger.info("✅ Orbbec SDK + MediaPipe initialized successfully")
                    else:
                         logger.warning("⚠️  MediaPipe not available. Camera only (no tracking).")

                return True

            except Exception as e:
                logger.error(f"❌ Failed to initialize Orbbec SDK: {e}")
                self.pipeline = None

        # 3. Fallback to Simulation
        logger.warning("⚠️  No camera/tracking available. Falling back to SIMULATION.")
        self.simulation = True
        self.is_running = True
        return True

    def stop(self):
        """Stop tracking and cleanup"""
        self.is_running = False

        if self.use_k4a:
            if self.k4a:
                try:
                    self.k4a.stop()
                except:
                    pass
                self.k4a = None
            self.tracker = None

        elif self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None

    def is_camera_connected(self):
        """Check if camera is connected"""
        return self.is_running and not self.simulation

    def get_skeleton(self):
        """
        Get current skeleton data.
        Returns a dictionary with skeleton data or None.
        This method blocks until a frame is available or timeout.
        """
        if not self.is_running:
            return None

        if self.simulation:
            return self.generate_simulated_skeleton()

        if self.use_k4a:
            return self._get_k4a_skeleton()

        if self.use_mediapipe:
            return self._get_mediapipe_skeleton()

        return None

    def _get_k4a_skeleton(self):
        """Process K4A frame and return skeleton"""
        try:
            # Capture frame
            capture = self.k4a.get_capture(timeout_ms=100)

            # Update tracker
            body_frame = self.tracker.update(capture)

            if body_frame.num_bodies == 0:
                return None

            # Get first body
            body = body_frame.bodies[0]

            # Map to standard format
            joints = {}
            joint_names = [
                'PELVIS', 'SPINE_NAVAL', 'SPINE_CHEST', 'NECK', 'CLAVICLE_LEFT',
                'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT', 'HAND_LEFT', 'HANDTIP_LEFT',
                'THUMB_LEFT', 'CLAVICLE_RIGHT', 'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',
                'HAND_RIGHT', 'HANDTIP_RIGHT', 'THUMB_RIGHT', 'HIP_LEFT', 'KNEE_LEFT',
                'ANKLE_LEFT', 'FOOT_LEFT', 'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT',
                'FOOT_RIGHT', 'HEAD', 'NOSE', 'EYE_LEFT', 'EAR_LEFT', 'EYE_RIGHT', 'EAR_RIGHT'
            ]

            for i, name in enumerate(joint_names):
                joint = body.joints[i]
                joints[name] = {
                    'position': {
                        'x': float(joint.position.x),
                        'y': float(joint.position.y),
                        'z': float(joint.position.z)
                    },
                    'orientation': {
                        'w': float(joint.orientation.w),
                        'x': float(joint.orientation.x),
                        'y': float(joint.orientation.y),
                        'z': float(joint.orientation.z)
                    },
                    'confidence': self._map_confidence(joint.confidence_level)
                }

            return {
                'timestamp': datetime.now().isoformat(),
                'body_id': int(body.id),
                'joints': joints,
                'simulation': False,
                'source': 'k4a'
            }

        except Exception as e:
            logger.error(f"❌ Error getting K4A skeleton: {e}")
            return None

    def _map_confidence(self, level):
        """Map K4A confidence level to string"""
        if level == 0: return 'NONE'
        if level == 1: return 'LOW'
        if level == 2: return 'MEDIUM'
        if level == 3: return 'HIGH'
        return 'UNKNOWN'

    def _get_mediapipe_skeleton(self):
        """Process Orbbec frame with MediaPipe and return skeleton"""
        try:
            # Get frames
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            if frames is None:
                return None

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                return None

            # Convert to numpy
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_data = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

            # Create MP Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color_data))

            # Detect
            timestamp_ms = int(time.time() * 1000)
            results = self.mp_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks:
                return self._map_mediapipe_to_k4abt(results, depth_data)

            return None

        except Exception as e:
            logger.error(f"❌ Error getting MediaPipe skeleton: {e}")
            return None

    def _deproject_pixel_to_point(self, u, v, depth, width=1920, height=1080):
        """Deproject 2D pixel to 3D point"""
        # Approximate intrinsics for 90 deg FOV
        fx = width / 2.0
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return {'x': x, 'y': y, 'z': z}

    def _map_mediapipe_to_k4abt(self, results, depth_image):
        """Map MediaPipe landmarks to K4ABT skeleton"""
        if not results.pose_landmarks:
            return None

        h, w = depth_image.shape
        landmarks = results.pose_landmarks[0]

        def get_joint_3d(mp_index):
            lm = landmarks[mp_index]
            px = int(lm.x * w)
            py = int(lm.y * h)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            d = float(depth_image[py, px])
            if d == 0:
                # 3x3 neighborhood check
                neighborhood = depth_image[max(0, py-1):min(h, py+2), max(0, px-1):min(w, px+2)]
                valid = neighborhood[neighborhood > 0]
                if valid.size > 0:
                    d = float(np.median(valid))
                else:
                    d = 1500.0 # Default fallback

            return self._deproject_pixel_to_point(px, py, d, w, h)

        joints = {}

        # Direct mappings
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

        for name, idx in mapping.items():
            joints[name] = {
                'position': get_joint_3d(idx),
                'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
                'confidence': 'HIGH' if landmarks[idx].visibility > 0.5 else 'LOW'
            }

        # Computed joints
        hl = joints['HIP_LEFT']['position']
        hr = joints['HIP_RIGHT']['position']
        pelvis = {
            'x': (hl['x'] + hr['x']) / 2,
            'y': (hl['y'] + hr['y']) / 2,
            'z': (hl['z'] + hr['z']) / 2
        }
        joints['PELVIS'] = {'position': pelvis, 'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'MEDIUM'}

        sl = joints['SHOULDER_LEFT']['position']
        sr = joints['SHOULDER_RIGHT']['position']
        neck = {
            'x': (sl['x'] + sr['x']) / 2,
            'y': (sl['y'] + sr['y']) / 2,
            'z': (sl['z'] + sr['z']) / 2
        }
        joints['NECK'] = {'position': neck, 'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'MEDIUM'}

        joints['SPINE_CHEST'] = {
            'position': {
                'x': neck['x'] * 0.7 + pelvis['x'] * 0.3,
                'y': neck['y'] * 0.7 + pelvis['y'] * 0.3,
                'z': neck['z'] * 0.7 + pelvis['z'] * 0.3,
            },
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'MEDIUM'
        }

        joints['SPINE_NAVAL'] = {
            'position': {
                'x': neck['x'] * 0.3 + pelvis['x'] * 0.7,
                'y': neck['y'] * 0.3 + pelvis['y'] * 0.7,
                'z': neck['z'] * 0.3 + pelvis['z'] * 0.7,
            },
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'MEDIUM'
        }

        # Head
        el = joints['EAR_LEFT']['position']
        er = joints['EAR_RIGHT']['position']
        joints['HEAD'] = {
            'position': {
                'x': (el['x'] + er['x']) / 2,
                'y': (el['y'] + er['y']) / 2,
                'z': (el['z'] + er['z']) / 2,
            },
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'MEDIUM'
        }

        joints['CLAVICLE_LEFT'] = joints['SHOULDER_LEFT']
        joints['CLAVICLE_RIGHT'] = joints['SHOULDER_RIGHT']

        joints['HAND_LEFT'] = {'position': get_joint_3d(19), 'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'HIGH'}
        joints['HAND_RIGHT'] = {'position': get_joint_3d(20), 'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'HIGH'}
        joints['THUMB_LEFT'] = {'position': get_joint_3d(21), 'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'HIGH'}
        joints['THUMB_RIGHT'] = {'position': get_joint_3d(22), 'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'HIGH'}
        joints['HANDTIP_LEFT'] = joints['HAND_LEFT']
        joints['HANDTIP_RIGHT'] = joints['HAND_RIGHT']
        joints['FOOT_LEFT'] = {'position': get_joint_3d(31), 'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'HIGH'}
        joints['FOOT_RIGHT'] = {'position': get_joint_3d(32), 'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 'HIGH'}

        return {
            'timestamp': datetime.now().isoformat(),
            'body_id': 0,
            'joints': joints,
            'simulation': False,
            'source': 'mediapipe'
        }

    def generate_simulated_skeleton(self):
        """Generate simulated skeleton for testing"""
        time_now = datetime.now().timestamp()
        squat_phase = (math.sin(time_now * 0.5) + 1) / 2

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
            'simulation': True,
            'source': 'simulation'
        }
