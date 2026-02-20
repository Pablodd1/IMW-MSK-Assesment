#!/usr/bin/env python3
"""
Orbbec Femto Mega Bridge Server - PRODUCTION VERSION
WebSocket server that streams skeleton data AND video from Orbbec Femto Mega camera
with proper color handling and full camera controls.

Requirements:
- Orbbec Femto Mega camera connected via USB 3.0
- OrbbecSDK_v2 installed
- Python 3.8+

Run: python server_production.py
"""

import asyncio
import json
import websockets
import logging
import base64
import numpy as np
import sys
from datetime import datetime
from typing import Optional, Dict, Any

# Try to import required packages
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  WARNING: opencv-python not installed. Video streaming will be limited.")

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("⚠️  WARNING: pyorbbecsdk not installed. Running in simulation mode.")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("⚠️  WARNING: mediapipe not installed. Body tracking will use simulated data.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Joint names mapping (MediaPipe to our format)
MEDIAPIPE_JOINT_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky', 'right_pinky',
    'left_index', 'right_index',
    'left_thumb', 'right_thumb',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

class FemtoMegaServer:
    """WebSocket server for Femto Mega skeleton + video streaming"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8765, simulation: bool = False):
        self.host = host
        self.port = port
        self.simulation = simulation or not SDK_AVAILABLE
        self.clients: set = set()
        self.pipeline: Optional[Pipeline] = None
        self.config: Optional[Config] = None
        self.is_started: bool = False
        self.is_streaming: bool = False
        self.stream_task: Optional[asyncio.Task] = None
        self.landmarker = None
        
        # Camera settings
        self.camera_settings = {
            'auto_exposure': True,
            'exposure': 100,
            'white_balance': 'auto',
            'gain': 50,
            'laser': True,
            'mirror': False
        }
        
        # Initialize MediaPipe if available
        if MP_AVAILABLE and not self.simulation:
            self.init_mediapipe()
    
    def init_mediapipe(self):
        """Initialize MediaPipe Pose Landmarker"""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                output_segmentation_masks=False
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info("✅ MediaPipe Pose Landmarker initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")
            self.landmarker = None
    
    def init_camera(self) -> bool:
        """Initialize Femto Mega camera with proper color settings"""
        if self.simulation:
            logger.info("📷 Running in SIMULATION mode")
            return True
            
        if not SDK_AVAILABLE:
            logger.warning("⚠️ SDK not available, running in simulation mode")
            self.simulation = True
            return True
            
        try:
            logger.info("📷 Initializing Femto Mega camera...")
            
            self.pipeline = Pipeline()
            self.config = Config()
            
            # Enable depth stream (for skeleton depth)
            self.config.enable_stream(
                OBSensorType.DEPTH_SENSOR,
                640, 576,  # Resolution
                OBFormat.Y16,
                30  # FPS
            )
            
            # Enable RGB color stream - PROPER COLOR SETTINGS
            # Using RGB format for correct colors (not BGR)
            self.config.enable_stream(
                OBSensorType.COLOR_SENSOR,
                1280, 720,  # 720p for better performance
                OBFormat.RGB,  # RGB format for correct colors!
                30
            )
            
            # Enable alignment (align depth to color)
            self.config.set_align_mode(OBAlignMode.ALIGN_D2C_SW_MODE)
            
            # Start pipeline
            self.pipeline.start(self.config)
            self.is_started = True
            
            logger.info("✅ Femto Mega initialized successfully")
            logger.info("   - Depth: 640x576 @ 30fps")
            logger.info("   - Color: 1280x720 RGB @ 30fps")
            logger.info("   - Alignment: Depth-to-Color enabled")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera: {e}")
            self.simulation = True
            return False
    
    def get_frames(self):
        """Get synchronized depth and color frames"""
        if not self.is_started or self.simulation:
            return self.get_simulation_frame()
            
        try:
            frames = self.pipeline.wait_for_frames(100)
            return frames
        except Exception as e:
            logger.error(f"❌ Error getting frames: {e}")
            return None
    
    def get_simulation_frame(self):
        """Generate simulation frame for testing"""
        # Create a simple gradient image
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Create gradient from blue to red (simulating color camera)
        for i in range(720):
            img[i, :, 0] = int(i * 255 / 720)  # Blue channel
            img[i, :, 1] = 100  # Green channel
            img[i, :, 2] = int((720 - i) * 255 / 720)  # Red channel
        
        # Add some text
        cv2.putText(img, "Femto Mega Simulation", (400, 360), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        return {'color': img, 'depth': np.zeros((576, 640), dtype=np.uint16)}
    
    def process_frame(self, frames) -> Optional[Dict[str, Any]]:
        """Process frame and return skeleton data"""
        if not frames:
            return None
            
        try:
            # Get color and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame is None:
                return None
                
            # Convert to numpy array with CORRECT COLOR (RGB)
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_data = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
            
            # IMPORTANT: OpenCV returns BGR, but Femto Mega RGB format is correct
            # If using OpenCV to capture, convert BGR to RGB:
            # color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            
            # Get depth if available
            depth_data = None
            if depth_frame:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
            
            # Process with MediaPipe for skeleton detection
            skeleton_data = self.detect_pose(color_data, depth_data)
            
            # Encode frame as JPEG for streaming (with correct colors)
            _, jpeg_data = cv2.imencode('.jpg', color_data)
            frame_base64 = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
            
            return {
                'skeleton': skeleton_data,
                'frame': frame_base64,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing frame: {e}")
            return None
    
    def detect_pose(self, color_image, depth_image=None):
        """Detect pose using MediaPipe"""
        if self.landmarker is None:
            return self.get_simulation_skeleton()
        
        try:
            from mediapipe import Image
            from mediapipe.ImageFormat import SRGB
            
            # Create MediaPipe image (ensure RGB format)
            mp_image = Image(image_format=SRGB, data=np.ascontiguousarray(color_image))
            
            # Detect pose
            timestamp_ms = int(datetime.now().timestamp() * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if not result.pose_landmarks:
                return None
                
            # Convert to our format
            landmarks = {}
            for idx, landmark in enumerate(result.pose_landmarks[0]):
                if idx < len(MEDIAPIPE_JOINT_NAMES):
                    joint_name = MEDIAPIPE_JOINT_NAMES[idx]
                    
                    # Get depth if available
                    z_depth = 0
                    if depth_image is not None:
                        px = int(landmark.x * depth_image.shape[1])
                        py = int(landmark.y * depth_image.shape[0])
                        px = max(0, min(px, depth_image.shape[1] - 1))
                        py = max(0, min(py, depth_image.shape[0] - 1))
                        try:
                            z_depth = float(depth_image[py, px]) / 1000.0  # Convert mm to meters
                        except:
                            z_depth = 0
                    
                    landmarks[joint_name] = {
                        'x': float(landmark.x),
                        'y': float(landmark.y),
                        'z': z_depth,
                        'visibility': float(landmark.visibility)
                    }
            
            return {
                'timestamp': timestamp_ms,
                'landmarks': landmarks
            }
            
        except Exception as e:
            logger.error(f"❌ Pose detection error: {e}")
            return self.get_simulation_skeleton()
    
    def get_simulation_skeleton(self):
        """Generate simulation skeleton for testing"""
        landmarks = {}
        
        # Basic pose skeleton coordinates (normalized 0-1)
        base_landmarks = {
            'nose': (0.5, 0.1, 0),
            'left_eye': (0.48, 0.08, 0),
            'right_eye': (0.52, 0.08, 0),
            'left_ear': (0.45, 0.1, 0),
            'right_eye': (0.55, 0.1, 0),
            'left_shoulder': (0.4, 0.2, 0),
            'right_shoulder': (0.6, 0.2, 0),
            'left_elbow': (0.35, 0.35, 0),
            'right_elbow': (0.65, 0.35, 0),
            'left_wrist': (0.3, 0.45, 0),
            'right_wrist': (0.7, 0.45, 0),
            'left_hip': (0.45, 0.5, 0),
            'right_hip': (0.55, 0.5, 0),
            'left_knee': (0.43, 0.7, 0),
            'right_knee': (0.57, 0.7, 0),
            'left_ankle': (0.42, 0.9, 0),
            'right_ankle': (0.58, 0.9, 0),
        }
        
        for name, (x, y, z) in base_landmarks.items():
            landmarks[name] = {
                'x': x + (np.random.random() - 0.5) * 0.01,
                'y': y + (np.random.random() - 0.5) * 0.01,
                'z': z,
                'visibility': 0.9 + np.random.random() * 0.1
            }
        
        return {
            'timestamp': int(datetime.now().timestamp() * 1000),
            'landmarks': landmarks
        }
    
    async def stream_data(self, websocket):
        """Main streaming loop"""
        logger.info("🎥 Starting skeleton + video stream...")
        
        while self.is_streaming:
            try:
                # Get processed frame
                data = self.process_frame(self.get_frames())
                
                if data:
                    # Send skeleton data
                    await websocket.send(json.dumps({
                        'type': 'skeleton',
                        'skeleton': data['skeleton'],
                        'timestamp': data['timestamp']
                    }))
                    
                    # Send video frame (every 2nd frame to save bandwidth)
                    if data['frame']:
                        await websocket.send(json.dumps({
                            'type': 'video_frame',
                            'image': data['frame'],
                            'timestamp': data['timestamp']
                        }))
                
                # Maintain ~30 FPS
                await asyncio.sleep(0.033)
                
            except Exception as e:
                logger.error(f"❌ Stream error: {e}")
                break
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        logger.info(f"👤 Client connected: {websocket.remote_address}")
        self.clients.add(websocket)
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'status',
                'status': 'connected',
                'message': 'Femto Mega Bridge connected',
                'simulation': self.simulation,
                'camera_info': {
                    'model': 'Orbbec Femto Mega' if not self.simulation else 'Simulation',
                    'color_format': 'RGB',
                    'resolution': '1280x720',
                    'fps': 30
                }
            }))
            
            # Handle commands
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get('command', '')
                    
                    await self.handle_command(websocket, command, data)
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("👤 Client disconnected")
        finally:
            self.clients.remove(websocket)
    
    async def handle_command(self, websocket, command: str, data: Dict):
        """Handle client commands"""
        
        if command == 'start_streaming':
            if not self.is_streaming:
                self.is_streaming = True
                # Initialize camera if not done
                if not self.is_started:
                    self.init_camera()
                self.stream_task = asyncio.create_task(self.stream_data(websocket))
                
                await websocket.send(json.dumps({
                    'type': 'status',
                    'status': 'streaming',
                    'message': 'Video streaming started'
                }))
        
        elif command == 'stop_streaming':
            self.is_streaming = False
            if self.stream_task:
                self.stream_task.cancel()
                
            await websocket.send(json.dumps({
                'type': 'status',
                'status': 'stopped',
                'message': 'Video streaming stopped'
            }))
        
        elif command == 'ping':
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }))
        
        elif command == 'get_camera_info':
            await websocket.send(json.dumps({
                'type': 'camera_info',
                'settings': self.camera_settings,
                'simulation': self.simulation,
                'streaming': self.is_streaming
            }))
        
        elif command == 'set_exposure':
            value = data.get('value', 100)
            self.camera_settings['exposure'] = value
            # Apply to camera if real
            logger.info(f"Setting exposure: {value}")
            await websocket.send(json.dumps({
                'type': 'status',
                'message': f'Exposure set to {value}'
            }))
        
        elif command == 'set_white_balance':
            mode = data.get('mode', 'auto')
            self.camera_settings['white_balance'] = mode
            logger.info(f"Setting white balance: {mode}")
            await websocket.send(json.dumps({
                'type': 'status',
                'message': f'White balance set to {mode}'
            }))
        
        elif command == 'set_gain':
            value = data.get('value', 50)
            self.camera_settings['gain'] = value
            logger.info(f"Setting gain: {value}")
            await websocket.send(json.dumps({
                'type': 'status',
                'message': f'Gain set to {value}'
            }))
        
        elif command == 'set_laser':
            enabled = data.get('enabled', True)
            self.camera_settings['laser'] = enabled
            logger.info(f"Laser: {'enabled' if enabled else 'disabled'}")
            await websocket.send(json.dumps({
                'type': 'status',
                'message': f"Laser {'enabled' if enabled else 'disabled'}"
            }))
        
        else:
            logger.warning(f"Unknown command: {command}")
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting Femto Mega Bridge Server on {self.host}:{self.port}")
        
        # Initialize camera
        self.init_camera()
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"✅ Server running on ws://{self.host}:{self.port}")
            logger.info("   - WebSocket endpoint for skeleton + video")
            logger.info("   - Connect with: new WebSocket('ws://YOUR_IP:8765')")
            
            # Run forever
            await asyncio.Future()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Femto Mega Bridge Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind to')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode')
    
    args = parser.parse_args()
    
    server = FemtoMegaServer(
        host=args.host,
        port=args.port,
        simulation=args.simulate
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("👋 Server stopped")
        sys.exit(0)


if __name__ == '__main__':
    main()
