#!/usr/bin/env python3
"""
Orbbec Femto Mega Bridge Server with Body Tracking
Complete production-ready implementation with real camera support
"""

import asyncio
import json
import websockets
import logging
from datetime import datetime
import sys
import signal
import os

# Import tracker
try:
    from tracker import FemtoMegaBodyTracker, SDK_AVAILABLE
except ImportError:
    # Handle case where tracker module is missing or fails to import
    print("⚠️  Failed to import tracker module.")
    SDK_AVAILABLE = False
    class FemtoMegaBodyTracker:
        def __init__(self): pass
        def init_camera(self): return False
        def get_frames(self): return None
        def extract_skeleton_from_depth(self, f): return None
        def generate_simulated_skeleton(self): return None
        def stop(self): pass
        is_started = False

if SDK_AVAILABLE:
    print("✅ OrbbecSDK imported successfully")
else:
    print("⚠️  pyorbbecsdk not available. Install with: pip install pyorbbecsdk")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FemtoBridgeServer:
    """WebSocket server for streaming skeleton data"""
    
    def __init__(self, host='0.0.0.0', port=8765, simulation=False):
        self.host = host
        self.port = port
        self.simulation = simulation or not SDK_AVAILABLE
        self.clients = set()
        self.tracker = FemtoMegaBodyTracker()
        self.is_streaming = False
        self.stream_task = None
        
    async def init_camera(self):
        """Initialize camera in async context"""
        if self.simulation:
            logger.info("📷 Running in SIMULATION mode (no camera required)")
            return True
        
        # Run camera init in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.tracker.init_camera)
    
    async def stream_skeleton_data(self):
        """Continuously capture and broadcast skeleton data"""
        logger.info("🎥 Starting skeleton data stream...")
        
        frame_count = 0
        start_time = datetime.now()
        
        while self.is_streaming:
            try:
                # Get skeleton data
                if self.simulation or not self.tracker.is_started:
                    skeleton = self.tracker.generate_simulated_skeleton()
                else:
                    frames = await asyncio.get_event_loop().run_in_executor(
                        None, self.tracker.get_frames
                    )
                    if frames:
                        skeleton = await asyncio.get_event_loop().run_in_executor(
                            None, self.tracker.extract_skeleton_from_depth, frames
                        )
                    else:
                        skeleton = None
                
                if skeleton and self.clients:
                    # Broadcast to all clients
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
                    
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every second
                        elapsed = (datetime.now() - start_time).total_seconds()
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        logger.info(f"📊 Streaming: {frame_count} frames, {fps:.1f} FPS, {len(self.clients)} clients")
                
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
            # Send connection message
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Femto Mega bridge server connected',
                'simulation': self.simulation,
                'sdk_available': SDK_AVAILABLE,
                'camera_ready': self.tracker.is_started,
                'timestamp': datetime.now().isoformat()
            }))
            
            # Handle commands
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
                            'simulation': self.simulation
                        }))
                    
                    elif command == 'stop_streaming':
                        self.is_streaming = False
                        if self.stream_task:
                            self.stream_task.cancel()
                        await websocket.send(json.dumps({
                            'type': 'streaming_stopped'
                        }))
                    
                    elif command == 'get_status':
                        await websocket.send(json.dumps({
                            'type': 'status',
                            'simulation': self.simulation,
                            'sdk_available': SDK_AVAILABLE,
                            'camera_ready': self.tracker.is_started,
                            'streaming': self.is_streaming,
                            'clients': len(self.clients)
                        }))
                    
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
        logger.info("🚀 Femto Mega Bridge Server v2.0")
        logger.info("=" * 60)
        
        # Initialize camera
        camera_ok = await self.init_camera()
        
        if not camera_ok and not self.simulation:
            logger.warning("⚠️  Camera initialization failed, using simulation mode")
            self.simulation = True
        
        # Start WebSocket server
        logger.info(f"📡 Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        ):
            logger.info(f"✅ Server ready at ws://{self.host}:{self.port}")
            logger.info("👉 Open PhysioMotion web app and select 'Femto Mega' camera")
            logger.info("=" * 60)
            
            if self.simulation:
                logger.info("📊 SIMULATION MODE ACTIVE")
                logger.info("   - Generating simulated skeleton data")
                if not SDK_AVAILABLE:
                    logger.info("   - pyorbbecsdk not installed")
                    logger.info("   - Install with: pip install pyorbbecsdk")
                logger.info("=" * 60)
            else:
                logger.info("📷 REAL CAMERA MODE")
                logger.info("   - Femto Mega connected and ready")
                logger.info("   - Streaming depth + color at 30 FPS")
                logger.info("=" * 60)
            
            # Auto-start streaming
            self.is_streaming = True
            self.stream_task = asyncio.create_task(self.stream_skeleton_data())
            
            # Run forever
            await asyncio.Future()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("\n👋 Shutting down bridge server...")
        
        self.is_streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        # Stop camera
        await asyncio.get_event_loop().run_in_executor(None, self.tracker.stop)
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
        
        logger.info("✅ Shutdown complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Femto Mega Bridge Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with real camera
  python server.py
  
  # Run in simulation mode
  python server.py --simulate
  
  # Custom host/port
  python server.py --host 192.168.1.100 --port 9000
  
  # Enable debug logging
  python server.py --debug
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0',
                       help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8765,
                       help='Server port (default: 8765)')
    parser.add_argument('--simulate', action='store_true',
                       help='Force simulation mode (no camera required)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create server
    server = FemtoBridgeServer(
        host=args.host,
        port=args.port,
        simulation=args.simulate
    )
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def signal_handler(sig, frame):
        logger.info(f"\n⚠️  Received signal {sig}")
        loop.create_task(server.shutdown())
        loop.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run server
    try:
        loop.run_until_complete(server.start())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        loop.close()


if __name__ == '__main__':
    main()
