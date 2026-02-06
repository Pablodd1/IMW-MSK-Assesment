#!/usr/bin/env python3
"""
Orbbec Femto Mega Bridge Server
WebSocket server that streams skeleton data from Orbbec Femto Mega camera
to PhysioMotion web application.

Requirements:
- Orbbec Femto Mega camera connected via USB 3.0
- OrbbecSDK_v2 installed
- Python 3.8+
"""

import asyncio
import json
import websockets
import logging
from datetime import datetime
import sys
import random
import math

# Try to import Orbbec SDK
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("⚠️  WARNING: pyorbbecsdk not installed. Running in simulation mode.")
    print("   Install with: pip install pyorbbecsdk")

# Try to import Azure Kinect SDK (pyk4a)
# We use deferred import in the class to prevent CI/CD scanning issues
try:
    import pyk4a
    K4A_AVAILABLE = True
except ImportError:
    K4A_AVAILABLE = False
    print("⚠️  WARNING: pyk4a not installed. Azure Kinect Body Tracking will be unavailable.")
    print("   Install with: pip install pyk4a")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 32 joints from Azure Kinect Body Tracking SDK
JOINT_NAMES = [
    'PELVIS', 'SPINE_NAVAL', 'SPINE_CHEST', 'NECK', 'CLAVICLE_LEFT',
    'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT', 'HAND_LEFT', 'HANDTIP_LEFT',
    'THUMB_LEFT', 'CLAVICLE_RIGHT', 'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',
    'HAND_RIGHT', 'HANDTIP_RIGHT', 'THUMB_RIGHT', 'HIP_LEFT', 'KNEE_LEFT',
    'ANKLE_LEFT', 'FOOT_LEFT', 'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT',
    'FOOT_RIGHT', 'HEAD', 'NOSE', 'EYE_LEFT', 'EAR_LEFT', 'EYE_RIGHT', 'EAR_RIGHT'
]

class FemtoBridgeServer:
    """WebSocket server for Femto Mega skeleton streaming"""
    
    def __init__(self, host='0.0.0.0', port=8765, simulation=False):
        self.host = host
        self.port = port
        self.simulation = simulation or (not SDK_AVAILABLE and not K4A_AVAILABLE)
        self.clients = set()
        self.pipeline = None
        self.k4a = None
        self.tracker = None
        self.use_k4a = False
        self.is_streaming = False
        
    def init_camera(self):
        """Initialize Femto Mega camera"""
        if self.simulation:
            logger.info("📷 Running in SIMULATION mode (no camera required)")
            return True

        # Try K4A first for body tracking support
        if K4A_AVAILABLE:
            try:
                # Import here to avoid top-level dependency issues in CI
                from pyk4a import PyK4A, Config as K4AConfig, ColorResolution, DepthMode, WiredSyncMode
                from pyk4a import BodyTracker

                logger.info("📷 Initializing Femto Mega in K4A mode...")
                self.k4a = PyK4A(
                    K4AConfig(
                        color_resolution=ColorResolution.RES_720P,
                        depth_mode=DepthMode.NFOV_UNBINNED,
                        camera_fps=30, # Use integer FPS directly
                        wired_sync_mode=WiredSyncMode.STANDALONE,
                    )
                )
                self.k4a.start()

                logger.info("✅ Camera started. Initializing Body Tracker...")
                self.tracker = BodyTracker(self.k4a.calibration)
                self.use_k4a = True
                logger.info("✅ Azure Kinect Body Tracking initialized successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize K4A/Body Tracking: {e}")
                logger.info("   Trying native Orbbec SDK...")
                self.use_k4a = False

        # Fallback to Orbbec SDK
        if SDK_AVAILABLE:
            try:
                logger.info("📷 Initializing Femto Mega with Orbbec SDK...")
                self.pipeline = Pipeline()

                # Configure streams
                config = Config()
                config.enable_stream(OBSensorType.DEPTH_SENSOR, 640, 576, OBFormat.Y16, 30)
                config.enable_stream(OBSensorType.COLOR_SENSOR, 1920, 1080, OBFormat.RGB, 30)

                # Start pipeline
                self.pipeline.start(config)
                logger.info("✅ Femto Mega camera initialized successfully (No Body Tracking)")
                return True

            except Exception as e:
                logger.error(f"❌ Failed to initialize Orbbec SDK: {e}")

        logger.info("   Falling back to SIMULATION mode")
        self.simulation = True
        return False
    
    def generate_simulated_skeleton(self):
        """Generate simulated skeleton data for testing"""
        # Simulate a person doing a squat movement
        time_now = datetime.now().timestamp()
        squat_phase = (math.sin(time_now * 0.5) + 1) / 2  # 0 to 1
        
        joints = {}
        
        for i, name in enumerate(JOINT_NAMES):
            # Simulate squatting motion (pelvis and legs move down)
            y_offset = 0
            if 'PELVIS' in name or 'HIP' in name or 'KNEE' in name:
                y_offset = -squat_phase * 300  # Squat down by 300mm
            
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
            'simulation': True
        }
    
    def _wait_for_frames_blocking(self):
        """Blocking call to wait for frames"""
        if self.use_k4a:
             return self.k4a.get_capture(timeout_ms=100)
        else:
            return self.pipeline.wait_for_frames(timeout_ms=100)

    async def capture_skeleton(self):
        """Capture skeleton data from Femto Mega (Async)"""
        if self.simulation:
            return self.generate_simulated_skeleton()
        
        loop = asyncio.get_event_loop()

        if self.use_k4a:
            try:
                # Run blocking capture in executor
                capture = await loop.run_in_executor(None, self._wait_for_frames_blocking)

                # Update tracker (this might also block, but usually fast enough.
                # Ideal would be to run update in executor too if it's slow)
                body_frame = self.tracker.update(capture)

                if body_frame.num_bodies == 0:
                    return None

                # Get the first body
                body = body_frame.bodies[0]

                # Map joints to expected format
                joints = {}

                for i, name in enumerate(JOINT_NAMES):
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
                        'confidence': joint.confidence_level
                    }

                return {
                    'timestamp': datetime.now().isoformat(),
                    'body_id': int(body.id),
                    'joints': joints,
                    'simulation': False
                }

            except Exception as e:
                logger.error(f"❌ Error capturing K4A frames: {e}")
                return None

        # Orbbec SDK fallback (no body tracking yet)
        try:
            # Get frames from camera (Async)
            frames = await loop.run_in_executor(None, self._wait_for_frames_blocking)

            if frames is None:
                return None
            
            # Orbbec SDK doesn't support Azure Kinect Body Tracking natively without wrapper
            # Return None to indicate no body detected
            return None
            
        except Exception as e:
            logger.error(f"❌ Error capturing frames: {e}")
            return None
    
    async def stream_skeleton_data(self):
        """Continuously capture and broadcast skeleton data"""
        logger.info("🎥 Starting skeleton data stream...")
        
        while self.is_streaming:
            try:
                # Capture skeleton
                skeleton = await self.capture_skeleton()
                
                if skeleton and self.clients:
                    # Broadcast to all connected clients
                    message = json.dumps({
                        'type': 'skeleton',
                        'skeleton': skeleton
                    })
                    
                    # Send to all clients
                    disconnected = set()
                    for client in self.clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                    
                    # Remove disconnected clients
                    self.clients -= disconnected
                
                # 30 FPS = 33ms between frames
                await asyncio.sleep(0.033)
                
            except Exception as e:
                logger.error(f"❌ Error in streaming loop: {e}")
                await asyncio.sleep(1)
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        client_addr = websocket.remote_address
        logger.info(f"✅ Client connected from {client_addr}")
        self.clients.add(websocket)
        
        try:
            # Send connection success message
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Femto Mega bridge server connected',
                'simulation': self.simulation,
                'timestamp': datetime.now().isoformat()
            }))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get('command')
                    
                    if command == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                    
                    elif command == 'start_streaming':
                        if not self.is_streaming:
                            self.is_streaming = True
                            asyncio.create_task(self.stream_skeleton_data())
                        await websocket.send(json.dumps({
                            'type': 'streaming_started',
                            'simulation': self.simulation
                        }))
                    
                    elif command == 'stop_streaming':
                        self.is_streaming = False
                        await websocket.send(json.dumps({
                            'type': 'streaming_stopped'
                        }))
                    
                    else:
                        logger.warning(f"⚠️  Unknown command: {command}")
                        
                except json.JSONDecodeError:
                    logger.error("❌ Invalid JSON received")
                except Exception as e:
                    logger.error(f"❌ Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_addr}")
        finally:
            self.clients.remove(websocket)
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info("=" * 60)
        logger.info("🚀 Femto Mega Bridge Server")
        logger.info("=" * 60)
        
        # Initialize camera
        self.init_camera()
        
        # Start WebSocket server
        logger.info(f"📡 Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"✅ Server ready at ws://{self.host}:{self.port}")
            logger.info("👉 Open PhysioMotion web app and select 'Femto Mega' camera")
            logger.info("=" * 60)
            
            if self.simulation:
                logger.info("📊 SIMULATION MODE ACTIVE")
                logger.info("   - Generating simulated skeleton data")
                logger.info("   - To use real camera:")
                logger.info("     1. Connect Femto Mega via USB 3.0")
                logger.info("     2. Install: pip install pyorbbecsdk")
                logger.info("     3. Restart server")
                logger.info("=" * 60)
            
            # Start streaming automatically
            self.is_streaming = True
            asyncio.create_task(self.stream_skeleton_data())
            
            # Run forever
            await asyncio.Future()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Femto Mega Bridge Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8765, help='Server port (default: 8765)')
    parser.add_argument('--simulate', action='store_true', help='Force simulation mode')
    
    args = parser.parse_args()
    
    # Create and start server
    server = FemtoBridgeServer(host=args.host, port=args.port, simulation=args.simulate)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down bridge server...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
