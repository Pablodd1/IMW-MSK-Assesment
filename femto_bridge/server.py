#!/usr/bin/env python3
"""
Orbbec Femto Mega Bridge Server
WebSocket server that streams skeleton data from Orbbec Femto Mega camera
to PhysioMotion web application.
"""

import asyncio
import json
import websockets
import logging
from datetime import datetime
import sys
import random
import math

# Import unified tracker
try:
    from body_tracker import FemtoMegaTracker
except ImportError:
    # Handle case where body_tracker is not in path (e.g. running from root)
    try:
        from femto_bridge.body_tracker import FemtoMegaTracker
    except ImportError:
        # Fallback if dependencies missing for tracker import itself?
        # But body_tracker.py handles missing deps gracefully.
        print("❌ Could not import FemtoMegaTracker. Ensure you are in the correct directory.")
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Joint names for Azure Kinect Body Tracking SDK (32 joints)
JOINT_NAMES = [
    'PELVIS', 'SPINE_NAVAL', 'SPINE_CHEST', 'NECK', 'CLAVICLE_LEFT',
    'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT', 'HAND_LEFT', 'HANDTIP_LEFT',
    'THUMB_LEFT', 'CLAVICLE_RIGHT', 'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',
    'HAND_RIGHT', 'HANDTIP_RIGHT', 'THUMB_RIGHT', 'HIP_LEFT', 'KNEE_LEFT',
    'ANKLE_LEFT', 'FOOT_LEFT', 'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT',
    'FOOT_RIGHT', 'HEAD', 'NOSE', 'EYE_LEFT', 'EAR_LEFT', 'EYE_RIGHT', 'EAR_RIGHT'
]

# Pre-calculate simulation metadata for performance
# Format: (name, is_squat_affected, x_base_offset, y_base_offset)
SIMULATION_METADATA = [
    (
        name,
        ('PELVIS' in name or 'HIP' in name or 'KNEE' in name),
        i * 10,
        i * 20
    )
    for i, name in enumerate(JOINT_NAMES)
]


class FemtoBridgeServer:
    """WebSocket server for Femto Mega skeleton streaming"""
    
    def __init__(self, host='0.0.0.0', port=8765, simulation=False):
        self.host = host
        self.port = port
        self.simulation = simulation
        self.clients = set()
        self.tracker = FemtoMegaTracker()
        self.is_streaming = False
        self.stream_task = None
        
        if self.simulation:
            # Force simulation mode
            self.tracker.simulation = True

    async def init_tracker(self):
        """Initialize tracker"""
        if self.simulation:
            logger.info("ℹ️  Simulation mode requested via arguments")
            self.tracker.simulation = True
        
        # Run start() in executor as it might block on camera init
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.tracker.start)

        status = "SIMULATION" if self.tracker.simulation else "CAMERA"
        source = "UNKNOWN"
        if self.tracker.use_k4a: source = "Azure Kinect SDK"
        elif self.tracker.use_mediapipe: source = "Orbbec SDK + MediaPipe"
        elif self.tracker.simulation: source = "Generated Data"

        logger.info(f"✅ Tracker initialized. Mode: {status}, Source: {source}")

    async def stream_skeleton_data(self):
        """Continuously capture and broadcast skeleton data"""
        logger.info("🎥 Starting skeleton data stream...")
        
        while self.is_streaming:
            try:
                # Capture skeleton (run in executor to avoid blocking event loop)
                loop = asyncio.get_event_loop()
                skeleton = await loop.run_in_executor(None, self.tracker.get_skeleton)
                
                if skeleton and self.clients:
                    # Broadcast
                    message = json.dumps({
                        'type': 'skeleton',
                        'skeleton': skeleton
                    })
                    
                    disconnected = set()
                    for client in self.clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)
                    
                    self.clients -= disconnected
                
                # ~30 FPS
                await asyncio.sleep(0.033)
                
            except Exception as e:
                logger.error(f"❌ Error in streaming loop: {e}")
                await asyncio.sleep(1)
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connections"""
        client_addr = websocket.remote_address
        logger.info(f"✅ Client connected from {client_addr}")
        self.clients.add(websocket)
        
        try:
            # Send connection success message
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Femto Mega bridge server connected',
                'simulation': self.tracker.simulation,
                'source': 'k4a' if self.tracker.use_k4a else ('mediapipe' if self.tracker.use_mediapipe else 'simulation'),
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
                            self.stream_task = asyncio.create_task(self.stream_skeleton_data())
                        await websocket.send(json.dumps({
                            'type': 'streaming_started',
                            'simulation': self.tracker.simulation
                        }))
                    
                    elif command == 'stop_streaming':
                        self.is_streaming = False
                        if self.stream_task:
                            self.stream_task.cancel()
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
            self.clients.discard(websocket)
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info("=" * 60)
        logger.info("🚀 Femto Mega Bridge Server")
        logger.info("=" * 60)
        
        # Initialize tracker
        await self.init_tracker()
        
        # Start WebSocket server
        logger.info(f"📡 Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"✅ Server ready at ws://{self.host}:{self.port}")
            
            # Start streaming automatically
            self.is_streaming = True
            self.stream_task = asyncio.create_task(self.stream_skeleton_data())
            
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
        server.tracker.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
