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

# Import unified tracker
from body_tracker import FemtoMegaTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FemtoBridgeServer:
    """WebSocket server for Femto Mega skeleton streaming"""
    
    def __init__(self, host='0.0.0.0', port=8765, simulation=False):
        self.host = host
        self.port = port
        self.simulation = simulation
        self.clients = set()
        self.tracker = FemtoMegaTracker()
        self.is_streaming = False
        
    def init_camera(self):
        """Initialize Femto Mega camera"""
        return self.tracker.init_camera(simulation=self.simulation)
    
    async def capture_skeleton(self):
        """Capture skeleton data from Femto Mega (Async)"""
        loop = asyncio.get_event_loop()
        try:
            # Run blocking tracker call in executor
            return await loop.run_in_executor(None, self.tracker.get_skeleton)
        except Exception as e:
            logger.error(f"❌ Error capturing skeleton: {e}")
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
                'simulation': self.tracker.mode == 'simulation',
                'mode': self.tracker.mode,
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
                            'simulation': self.tracker.mode == 'simulation'
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
            self.clients.discard(websocket)
    
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
            
            if self.tracker.mode == 'simulation':
                logger.info("📊 SIMULATION MODE ACTIVE")
            else:
                logger.info(f"📷 CAMERA MODE: {self.tracker.mode}")

            # Start streaming automatically
            self.is_streaming = True
            asyncio.create_task(self.stream_skeleton_data())
            
            # Run forever
            await asyncio.Future()

    def shutdown(self):
        """Shutdown server"""
        self.is_streaming = False
        if self.tracker:
            self.tracker.stop()

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
        server.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
