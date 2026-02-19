import logging
import time
import json
import os
import sys
import numpy as np
from datetime import datetime
import urllib.request

# Configure logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dependency Imports
# -----------------------------------------------------------------------------

# 1. Azure Kinect SDK (PyK4A) - Preferred for Body Tracking
try:
    import pyk4a
    from pyk4a import PyK4A, Config as K4AConfig, ColorResolution, DepthMode, WiredSyncMode
    from pyk4a import BodyTracker
    K4A_AVAILABLE = True
except ImportError:
    K4A_AVAILABLE = False

# 2. Orbbec SDK (PyOrbbecSDK) - Fallback for Camera Access
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# 3. MediaPipe - Fallback for Body Tracking (used with Orbbec SDK)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FemtoMegaTracker:
    """
    Unified Body Tracker for Femto Mega.
    Prioritizes Azure Kinect SDK (pyk4a) for native body tracking.
    Falls back to Orbbec SDK + MediaPipe if K4A is unavailable.
    Defaults to Simulation if no camera is found.
    """

    def __init__(self):
        self.k4a = None
        self.pipeline = None  # Orbbec pipeline
        self.tracker = None   # K4A BodyTracker or MediaPipe Landmarker
        self.mode = 'simulation' # 'k4a', 'orbbec_mediapipe', 'simulation'
        self.is_running = False

        # MediaPipe specific
        self.mp_landmarker = None

    def init_camera(self, simulation=False):
        """Initialize camera and tracking (Blocking)"""
        logger.info("Initializing Femto Mega Tracker...")

        if simulation:
            logger.info("📊 Simulation mode requested.")
            self.mode = 'simulation'
            self.is_running = True
            return True

        # 1. Try Azure Kinect SDK (PyK4A)
        if K4A_AVAILABLE:
            try:
                logger.info("📷 Attempting to initialize Femto Mega with PyK4A...")
                self.k4a = PyK4A(
                    K4AConfig(
                        color_resolution=ColorResolution.RES_720P,
                        depth_mode=DepthMode.NFOV_UNBINNED,
                        camera_fps=30,
                        wired_sync_mode=WiredSyncMode.STANDALONE,
                    )
                )
                self.k4a.start()
                logger.info("✅ Camera started (PyK4A). Initializing Body Tracker...")

                self.tracker = BodyTracker(self.k4a.calibration)
                self.mode = 'k4a'
                self.is_running = True
                logger.info("✅ Azure Kinect Body Tracking initialized successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize K4A/Body Tracking: {e}")
                if self.k4a:
                    try:
                        self.k4a.stop()
                    except:
                        pass
                    self.k4a = None

        # 2. Try Orbbec SDK + MediaPipe
        if SDK_AVAILABLE:
            try:
                logger.info("📷 Attempting to initialize Femto Mega with Orbbec SDK...")
                self.pipeline = Pipeline()
                config = Config()

                # Configure streams
                config.enable_stream(OBSensorType.DEPTH_SENSOR, 640, 576, OBFormat.Y16, 30)
                config.enable_stream(OBSensorType.COLOR_SENSOR, 1920, 1080, OBFormat.RGB, 30)
                config.set_align_mode(OBAlignMode.ALIGN_D2C_SW_MODE)

                self.pipeline.start(config)
                logger.info("✅ Camera started (Orbbec SDK).")

                if MEDIAPIPE_AVAILABLE:
                    logger.info("🧠 Initializing MediaPipe Pose Landmarker...")
                    if self._init_mediapipe():
                        self.mode = 'orbbec_mediapipe'
                        self.is_running = True
                        logger.info("✅ MediaPipe Tracking initialized successfully")
                        return True
                    else:
                        logger.warning("⚠️ MediaPipe initialization failed. Falling back to simple streaming (no tracking).")
                        # Technically we could stream just video, but for 'tracker' we want skeleton.
                        # We will treat this as 'orbbec_raw' but get_skeleton will return None.
                        self.mode = 'orbbec_raw'
                        self.is_running = True
                        return True
                else:
                    logger.warning("⚠️ MediaPipe not available. Body tracking disabled.")
                    self.mode = 'orbbec_raw'
                    self.is_running = True
                    return True

            except Exception as e:
                logger.error(f"❌ Failed to initialize Orbbec SDK: {e}")
                if self.pipeline:
                    try:
                        self.pipeline.stop()
                    except:
                        pass
                    self.pipeline = None

        # 3. Fallback to Simulation
        logger.info("⚠️  No camera/SDK available. Falling back to SIMULATION mode.")
        self.mode = 'simulation'
        self.is_running = True
        return True

    def _init_mediapipe(self):
        """Initialize MediaPipe Pose Landmarker"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, 'models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            model_path = os.path.join(models_dir, 'pose_landmarker_full.task')

            if not os.path.exists(model_path):
                logger.info(f"⬇️  Downloading MediaPipe model to {model_path}...")
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
                urllib.request.urlretrieve(url, model_path)
                logger.info("✅ Model downloaded successfully")

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
            return True
        except Exception as e:
            logger.error(f"❌ MediaPipe init error: {e}")
            return False

    def get_skeleton(self):
        """
        Get current skeleton data.
        Returns unified skeleton dictionary or None.
        Blocking call (should be run in executor).
        """
        if not self.is_running:
            return None

        if self.mode == 'k4a':
            return self._get_skeleton_k4a()
        elif self.mode == 'orbbec_mediapipe':
            return self._get_skeleton_mediapipe()
        elif self.mode == 'simulation':
            return self._get_skeleton_simulation()

        return None

    def _get_skeleton_k4a(self):
        """Get skeleton using Azure Kinect SDK"""
        try:
            # Blocking capture
            capture = self.k4a.get_capture(timeout_ms=100)

            # Feed to tracker
            body_frame = self.tracker.update(capture)

            if body_frame.num_bodies == 0:
                return None

            body = body_frame.bodies[0]
            return self._convert_k4a_body_to_dict(body)

        except Exception as e:
            logger.error(f"K4A Tracking Error: {e}")
            return None

    def _get_skeleton_mediapipe(self):
        """Get skeleton using Orbbec SDK + MediaPipe"""
        try:
            # Blocking capture
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            if frames is None:
                return None

            # Process frames
            return self._process_frames_mediapipe(frames)

        except Exception as e:
            logger.error(f"MediaPipe Tracking Error: {e}")
            return None

    def _process_frames_mediapipe(self, frames):
        """Extract skeleton from Orbbec frames using MediaPipe"""
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None

        # Convert to numpy
        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        color_data = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

        # Detect landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color_data))
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        results = self.mp_landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            return self._map_mediapipe_to_k4abt(results, depth_data)

        return None

    def _convert_k4a_body_to_dict(self, body):
        """Convert K4A body object to standard dictionary"""
        joints = {}
        # 32 joints as per K4ABT
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
                'confidence': int(joint.confidence_level) # 0=NONE, 1=LOW, 2=MEDIUM, 3=HIGH
            }

        return {
            'timestamp': datetime.now().isoformat(),
            'body_id': int(body.id),
            'joints': joints,
            'source': 'k4a'
        }

    def _map_mediapipe_to_k4abt(self, results, depth_image):
        """Map MediaPipe landmarks + Depth to K4ABT structure"""
        # (This logic is ported from server_production.py)
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
                neighborhood = depth_image[max(0, py-1):min(h, py+2), max(0, px-1):min(w, px+2)]
                valid = neighborhood[neighborhood > 0]
                if valid.size > 0:
                    d = float(np.median(valid))
                else:
                    d = 1500.0 # Fallback

            return self._deproject_pixel_to_point(px, py, d, w, h)

        # Mapping dictionary (MediaPipe Index -> K4A Name)
        mp_map = {
            11: 'SHOULDER_LEFT', 12: 'SHOULDER_RIGHT',
            13: 'ELBOW_LEFT', 14: 'ELBOW_RIGHT',
            15: 'WRIST_LEFT', 16: 'WRIST_RIGHT',
            23: 'HIP_LEFT', 24: 'HIP_RIGHT',
            25: 'KNEE_LEFT', 26: 'KNEE_RIGHT',
            27: 'ANKLE_LEFT', 28: 'ANKLE_RIGHT',
            2: 'EYE_LEFT', 5: 'EYE_RIGHT',
            7: 'EAR_LEFT', 8: 'EAR_RIGHT',
            0: 'NOSE',
            19: 'HAND_LEFT', 20: 'HAND_RIGHT',
            21: 'THUMB_LEFT', 22: 'THUMB_RIGHT',
            31: 'FOOT_LEFT', 32: 'FOOT_RIGHT'
        }

        joints = {}
        for mp_idx, name in mp_map.items():
            joints[name] = {
                'position': get_joint_3d(mp_idx),
                'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
                'confidence': 2 # MEDIUM
            }

        # Helper to interpolate
        def interpolate(name, parent, child, ratio):
            p = joints[parent]['position']
            c = joints[child]['position']
            joints[name] = {
                'position': {
                    'x': p['x'] * (1-ratio) + c['x'] * ratio,
                    'y': p['y'] * (1-ratio) + c['y'] * ratio,
                    'z': p['z'] * (1-ratio) + c['z'] * ratio,
                },
                'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0},
                'confidence': 2
            }

        # Computed joints
        # PELVIS
        hl = joints['HIP_LEFT']['position']
        hr = joints['HIP_RIGHT']['position']
        joints['PELVIS'] = {
            'position': {'x': (hl['x']+hr['x'])/2, 'y': (hl['y']+hr['y'])/2, 'z': (hl['z']+hr['z'])/2},
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 2
        }
        # NECK
        sl = joints['SHOULDER_LEFT']['position']
        sr = joints['SHOULDER_RIGHT']['position']
        joints['NECK'] = {
            'position': {'x': (sl['x']+sr['x'])/2, 'y': (sl['y']+sr['y'])/2, 'z': (sl['z']+sr['z'])/2},
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 2
        }
        # SPINE_CHEST
        interpolate('SPINE_CHEST', 'NECK', 'PELVIS', 0.3)
        # SPINE_NAVAL
        interpolate('SPINE_NAVAL', 'NECK', 'PELVIS', 0.7)
        # HEAD
        el = joints['EAR_LEFT']['position']
        er = joints['EAR_RIGHT']['position']
        joints['HEAD'] = {
            'position': {'x': (el['x']+er['x'])/2, 'y': (el['y']+er['y'])/2, 'z': (el['z']+er['z'])/2},
            'orientation': {'w': 1, 'x': 0, 'y': 0, 'z': 0}, 'confidence': 2
        }
        # CLAVICLES
        joints['CLAVICLE_LEFT'] = joints['SHOULDER_LEFT']
        joints['CLAVICLE_RIGHT'] = joints['SHOULDER_RIGHT']
        # HANDTIPS
        joints['HANDTIP_LEFT'] = joints['HAND_LEFT']
        joints['HANDTIP_RIGHT'] = joints['HAND_RIGHT']

        return {
            'timestamp': datetime.now().isoformat(),
            'body_id': 0,
            'joints': joints,
            'source': 'mediapipe'
        }

    def _deproject_pixel_to_point(self, u, v, depth, width, height):
        """Deproject 2D pixel to 3D point (Approximation)"""
        fx = width / 2.0
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return {'x': x, 'y': y, 'z': z}

    def _get_skeleton_simulation(self):
        """Generate simulated skeleton"""
        import random
        import math

        time_sec = datetime.now().timestamp()
        squat_phase = (math.sin(time_sec * 0.5) + 1) / 2

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
            y_offset = 0
            if 'PELVIS' in name or 'HIP' in name or 'KNEE' in name:
                y_offset = -squat_phase * 300

            joints[name] = {
                'position': {
                    'x': random.uniform(-200, 200) + (i * 10),
                    'y': 500 + y_offset + (i * 20),
                    'z': 1500 + random.uniform(-50, 50)
                },
                'orientation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                'confidence': 3 # HIGH
            }

        return {
            'timestamp': datetime.now().isoformat(),
            'body_id': 0,
            'joints': joints,
            'source': 'simulation'
        }

    def stop(self):
        """Stop tracking and release resources"""
        logger.info("Stopping Femto Mega Tracker...")
        self.is_running = False

        if self.k4a:
            try:
                self.k4a.stop()
            except:
                pass
            self.k4a = None

        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None

        logger.info("✅ Tracker stopped")
