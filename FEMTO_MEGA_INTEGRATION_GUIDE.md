# 🎥 Orbbec Femto Mega Integration Guide

## 📋 Overview

This guide provides complete instructions for integrating your **Orbbec Femto Mega** camera with the PhysioMotion platform for professional-grade clinical movement assessments.

---

## ✅ Current Integration Status

### **Documentation Status:**
- ✅ Mentioned in README.md
- ✅ Mentioned in WORKFLOW_GUIDE.md
- ✅ Camera selection option in frontend
- ⚠️ **Bridge server NOT yet implemented**
- ⚠️ **WebSocket connection NOT yet implemented**
- ⚠️ **Azure Kinect Body Tracking SDK NOT yet integrated**

### **What Works Now:**
- ✅ MediaPipe Pose (33-point browser-based tracking)
- ✅ Phone camera (front/back with flip)
- ✅ Laptop/external USB cameras
- ✅ Video upload for offline analysis

### **What Needs Implementation:**
- ❌ Femto Mega bridge server (Python/Node.js)
- ❌ 32-joint Azure Kinect Body Tracking
- ❌ Depth sensing integration
- ❌ Multi-camera synchronization

---

## 🏗️ Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  FEMTO MEGA INTEGRATION                      │
└─────────────────────────────────────────────────────────────┘

    Femto Mega Camera (Hardware)
            │
            │ USB 3.0 / PoE
            ▼
    Clinic Workstation (Windows/Linux)
            │
            │ OrbbecSDK_v2
            ▼
    Python Bridge Server
            │ - Body tracking
            │ - 32-joint skeleton
            │ - Depth data
            │
            │ WebSocket
            ▼
    PhysioMotion Web App (Browser)
            │ - Receive skeleton data
            │ - Real-time visualization
            │ - Biomechanical analysis
            │
            │ REST API
            ▼
    Cloudflare D1 Database
            │ - Store results
            │ - Generate reports
```

---

## 🔧 Hardware Requirements

### **Orbbec Femto Mega Specifications:**
- **Resolution**: 1280x800 depth, 1920x1080 color
- **Frame Rate**: 30 fps
- **Interface**: USB 3.0 or PoE (Power over Ethernet)
- **Range**: 0.5m - 5.46m
- **SDK**: OrbbecSDK_v2
- **Body Tracking**: Azure Kinect Body Tracking SDK compatible

### **Workstation Requirements:**
- **OS**: Windows 10/11 (64-bit) or Ubuntu 18.04+
- **CPU**: Intel i5-9400 or better (body tracking requires good CPU)
- **GPU**: NVIDIA GPU recommended (CUDA support for faster processing)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 100GB available space
- **USB**: USB 3.0 port (or PoE network adapter)

---

## 📦 Software Installation

### **Step 1: Install OrbbecSDK_v2**

#### **For Windows:**
```bash
# Download from GitHub
https://github.com/orbbec/OrbbecSDK_v2/releases

# Extract to C:\OrbbecSDK_v2
# Add to PATH:
C:\OrbbecSDK_v2\bin

# Install Visual C++ Redistributables
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### **For Linux (Ubuntu):**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    git cmake build-essential \
    libusb-1.0-0-dev libudev-dev \
    python3-dev python3-pip

# Clone SDK
git clone https://github.com/orbbec/OrbbecSDK_v2.git
cd OrbbecSDK_v2

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Configure USB rules
sudo cp ../misc/99-obsensor-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

### **Step 2: Install Python Bindings**

```bash
# Install Python package
pip install pyorbbecsdk

# Verify installation
python -c "import pyorbbecsdk; print(pyorbbecsdk.__version__)"
```

---

### **Step 3: Install Azure Kinect Body Tracking SDK**

#### **For Windows:**
```bash
# Download Body Tracking SDK
https://docs.microsoft.com/en-us/azure/kinect-dk/body-sdk-download

# Install MSI package
# Default location: C:\Program Files\Azure Kinect Body Tracking SDK

# Install CUDA (if using GPU acceleration)
https://developer.nvidia.com/cuda-downloads

# Install cuDNN
https://developer.nvidia.com/cudnn
```

#### **For Linux:**
```bash
# Add Microsoft package repository
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod

# Install Body Tracking SDK
sudo apt-get update
sudo apt-get install -y k4a-tools
sudo apt-get install -y libk4abt1.1-dev

# Verify installation
k4abt_simple_3d_viewer
```

---

## 🐍 Python Bridge Server Implementation

### **File Structure:**
```
femto_mega_bridge/
├── server.py              # Main WebSocket server
├── body_tracker.py        # Body tracking logic
├── skeleton_processor.py  # Data processing
├── config.yaml           # Configuration
├── requirements.txt      # Python dependencies
└── README.md            # Bridge server docs
```

---

### **requirements.txt:**
```txt
pyorbbecsdk>=1.5.0
websockets>=12.0
asyncio>=3.4.3
numpy>=1.24.0
opencv-python>=4.8.0
pyyaml>=6.0
```

---

### **server.py - WebSocket Bridge Server:**
```python
#!/usr/bin/env python3
"""
Femto Mega Bridge Server
WebSocket server that streams skeleton data from Orbbec Femto Mega
to PhysioMotion web application.
"""

import asyncio
import json
import websockets
import logging
from body_tracker import FemtoMegaTracker
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FemtoBridgeServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.tracker = FemtoMegaTracker()
        self.clients = set()
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        logger.info(f"Client connected from {websocket.remote_address}")
        self.clients.add(websocket)
        
        try:
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Femto Mega bridge server connected',
                'timestamp': datetime.now().isoformat()
            }))
            
            # Keep connection alive and handle commands
            async for message in websocket:
                data = json.loads(message)
                await self.handle_command(websocket, data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            self.clients.remove(websocket)
    
    async def handle_command(self, websocket, data):
        """Handle commands from client"""
        command = data.get('command')
        
        if command == 'start_tracking':
            await self.start_tracking(websocket)
        elif command == 'stop_tracking':
            await self.stop_tracking(websocket)
        elif command == 'get_status':
            await self.send_status(websocket)
    
    async def start_tracking(self, websocket):
        """Start streaming skeleton data"""
        logger.info("Starting skeleton tracking...")
        
        try:
            self.tracker.start()
            
            while self.tracker.is_running:
                skeleton_data = self.tracker.get_skeleton()
                
                if skeleton_data:
                    await websocket.send(json.dumps({
                        'type': 'skeleton_frame',
                        'data': skeleton_data,
                        'timestamp': datetime.now().isoformat()
                    }))
                
                await asyncio.sleep(0.033)  # ~30 fps
                
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def stop_tracking(self, websocket):
        """Stop tracking"""
        self.tracker.stop()
        await websocket.send(json.dumps({
            'type': 'tracking_stopped',
            'message': 'Tracking stopped'
        }))
    
    async def send_status(self, websocket):
        """Send server status"""
        await websocket.send(json.dumps({
            'type': 'status',
            'data': {
                'tracking': self.tracker.is_running,
                'connected_clients': len(self.clients),
                'camera_connected': self.tracker.is_camera_connected()
            }
        }))
    
    async def start(self):
        """Start WebSocket server"""
        logger.info(f"Starting Femto Bridge Server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info("Server started successfully")
            await asyncio.Future()  # Run forever

if __name__ == '__main__':
    server = FemtoBridgeServer()
    asyncio.run(server.start())
```

---

### **body_tracker.py - Skeleton Tracking:**
```python
"""
Body tracking using Orbbec Femto Mega + Azure Kinect Body Tracking SDK
"""

import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
import logging

logger = logging.getLogger(__name__)

class FemtoMegaTracker:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.is_running = False
        self.current_skeleton = None
        
    def start(self):
        """Initialize and start camera"""
        try:
            # Initialize pipeline
            self.pipeline = Pipeline()
            self.config = Config()
            
            # Enable depth and color streams
            self.config.enable_stream(OBSensorType.DEPTH_SENSOR, 
                                     width=1024, height=1024, 
                                     fps=30, format=OBFormat.Y16)
            
            self.config.enable_stream(OBSensorType.COLOR_SENSOR,
                                     width=1920, height=1080,
                                     fps=30, format=OBFormat.RGB)
            
            # Start pipeline
            self.pipeline.start(self.config)
            self.is_running = True
            
            logger.info("Femto Mega camera started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            raise
    
    def get_skeleton(self):
        """Get current skeleton data (32 joints)"""
        if not self.is_running:
            return None
        
        try:
            # Get frames
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            
            if frames is None:
                return None
            
            # Get depth frame
            depth_frame = frames.get_depth_frame()
            
            if depth_frame is None:
                return None
            
            # TODO: Integrate Azure Kinect Body Tracking SDK here
            # For now, return placeholder skeleton data
            
            skeleton = self._process_body_tracking(depth_frame)
            return skeleton
            
        except Exception as e:
            logger.error(f"Error getting skeleton: {e}")
            return None
    
    def _process_body_tracking(self, depth_frame):
        """
        Process depth frame with Azure Kinect Body Tracking
        Returns 32-joint skeleton data in standard format
        """
        
        # TODO: Call Azure Kinect Body Tracking SDK
        # This requires C++ bindings or external process
        
        # Placeholder skeleton structure (32 joints)
        joints = {
            'pelvis': {'x': 0.0, 'y': 0.0, 'z': 1.5, 'confidence': 0.95},
            'spine_naval': {'x': 0.0, 'y': 0.1, 'z': 1.5, 'confidence': 0.95},
            'spine_chest': {'x': 0.0, 'y': 0.3, 'z': 1.5, 'confidence': 0.95},
            'neck': {'x': 0.0, 'y': 0.5, 'z': 1.5, 'confidence': 0.95},
            'head': {'x': 0.0, 'y': 0.65, 'z': 1.5, 'confidence': 0.95},
            
            # Left arm
            'left_clavicle': {'x': 0.1, 'y': 0.5, 'z': 1.5, 'confidence': 0.92},
            'left_shoulder': {'x': 0.2, 'y': 0.5, 'z': 1.5, 'confidence': 0.92},
            'left_elbow': {'x': 0.3, 'y': 0.3, 'z': 1.5, 'confidence': 0.90},
            'left_wrist': {'x': 0.35, 'y': 0.1, 'z': 1.5, 'confidence': 0.88},
            'left_hand': {'x': 0.37, 'y': 0.05, 'z': 1.5, 'confidence': 0.85},
            
            # Right arm
            'right_clavicle': {'x': -0.1, 'y': 0.5, 'z': 1.5, 'confidence': 0.92},
            'right_shoulder': {'x': -0.2, 'y': 0.5, 'z': 1.5, 'confidence': 0.92},
            'right_elbow': {'x': -0.3, 'y': 0.3, 'z': 1.5, 'confidence': 0.90},
            'right_wrist': {'x': -0.35, 'y': 0.1, 'z': 1.5, 'confidence': 0.88},
            'right_hand': {'x': -0.37, 'y': 0.05, 'z': 1.5, 'confidence': 0.85},
            
            # Left leg
            'left_hip': {'x': 0.1, 'y': -0.1, 'z': 1.5, 'confidence': 0.95},
            'left_knee': {'x': 0.1, 'y': -0.5, 'z': 1.5, 'confidence': 0.93},
            'left_ankle': {'x': 0.1, 'y': -0.9, 'z': 1.5, 'confidence': 0.90},
            'left_foot': {'x': 0.1, 'y': -1.0, 'z': 1.5, 'confidence': 0.88},
            
            # Right leg
            'right_hip': {'x': -0.1, 'y': -0.1, 'z': 1.5, 'confidence': 0.95},
            'right_knee': {'x': -0.1, 'y': -0.5, 'z': 1.5, 'confidence': 0.93},
            'right_ankle': {'x': -0.1, 'y': -0.9, 'z': 1.5, 'confidence': 0.90},
            'right_foot': {'x': -0.1, 'y': -1.0, 'z': 1.5, 'confidence': 0.88},
        }
        
        return {
            'timestamp': np.datetime64('now').item().isoformat(),
            'joints': joints,
            'tracking_id': 1,
            'body_index': 0
        }
    
    def stop(self):
        """Stop camera and cleanup"""
        if self.pipeline:
            self.pipeline.stop()
            self.is_running = False
            logger.info("Camera stopped")
    
    def is_camera_connected(self):
        """Check if camera is connected"""
        try:
            return self.pipeline is not None and self.is_running
        except:
            return False
```

---

## 🌐 Frontend Integration (JavaScript)

### **Add to assessment-workflow.js:**

```javascript
// Femto Mega WebSocket client
class FemtoMegaClient {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.onSkeletonData = null;
        this.serverUrl = 'ws://localhost:8765';
    }
    
    async connect() {
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.serverUrl);
                
                this.ws.onopen = () => {
                    console.log('Connected to Femto Mega bridge server');
                    this.isConnected = true;
                    resolve();
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    reject(error);
                };
                
                this.ws.onclose = () => {
                    console.log('Disconnected from bridge server');
                    this.isConnected = false;
                };
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'skeleton_frame':
                if (this.onSkeletonData) {
                    this.onSkeletonData(data.data);
                }
                break;
            case 'error':
                console.error('Bridge server error:', data.message);
                break;
            case 'tracking_stopped':
                console.log('Tracking stopped');
                break;
        }
    }
    
    startTracking() {
        if (this.ws && this.isConnected) {
            this.ws.send(JSON.stringify({
                command: 'start_tracking'
            }));
        }
    }
    
    stopTracking() {
        if (this.ws && this.isConnected) {
            this.ws.send(JSON.stringify({
                command: 'stop_tracking'
            }));
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Initialize Femto Mega when selected
async function initializeFemtoMega() {
    try {
        femtoClient = new FemtoMegaClient();
        await femtoClient.connect();
        
        femtoClient.onSkeletonData = (skeleton) => {
            // Draw skeleton overlay
            drawFemtoSkeleton(skeleton);
            
            // Store for analysis
            recordedFrames.push({
                timestamp: Date.now(),
                skeleton: skeleton
            });
        };
        
        femtoClient.startTracking();
        
        showToast('Femto Mega connected', 'success');
        
    } catch (error) {
        console.error('Failed to connect to Femto Mega:', error);
        showToast('Failed to connect to Femto Mega bridge server', 'error');
    }
}

function drawFemtoSkeleton(skeleton) {
    // Clear canvas
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    
    const joints = skeleton.joints;
    
    // Draw joints
    for (const [jointName, joint] of Object.entries(joints)) {
        if (joint.confidence > 0.5) {
            // Project 3D to 2D
            const screenPos = project3DTo2D(joint, overlayCanvas);
            
            // Draw joint circle
            overlayCtx.beginPath();
            overlayCtx.arc(screenPos.x, screenPos.y, 8, 0, 2 * Math.PI);
            overlayCtx.fillStyle = '#06b6d4'; // Cyan
            overlayCtx.fill();
            overlayCtx.strokeStyle = '#fff';
            overlayCtx.lineWidth = 2;
            overlayCtx.stroke();
        }
    }
    
    // Draw skeleton connections (simplified)
    drawConnection(joints.left_shoulder, joints.left_elbow);
    drawConnection(joints.left_elbow, joints.left_wrist);
    drawConnection(joints.right_shoulder, joints.right_elbow);
    drawConnection(joints.right_elbow, joints.right_wrist);
    // ... add more connections
}

function project3DTo2D(joint, canvas) {
    // Simple perspective projection
    // Adjust based on camera calibration
    const fov = 60;
    const scale = canvas.width / (2 * Math.tan(fov * Math.PI / 360));
    
    const x = (joint.x / joint.z) * scale + canvas.width / 2;
    const y = (joint.y / joint.z) * scale + canvas.height / 2;
    
    return { x, y };
}
```

---

## 🚀 Deployment Instructions

### **Step 1: Start Bridge Server**

```bash
# On clinic workstation with Femto Mega connected

cd /path/to/femto_mega_bridge
python3 server.py

# Expected output:
# INFO:__main__:Starting Femto Bridge Server on 0.0.0.0:8765
# INFO:__main__:Server started successfully
# INFO:__main__:Femto Mega camera started successfully
```

---

### **Step 2: Configure Frontend**

Update the WebSocket URL in assessment-workflow.js:

```javascript
// For local testing
this.serverUrl = 'ws://localhost:8765';

// For production (must be HTTPS + WSS)
this.serverUrl = 'wss://your-clinic-workstation.local:8765';
```

---

### **Step 3: Test Connection**

1. Open PhysioMotion web app
2. Navigate to Assessment page
3. Click "Femto Mega" camera option
4. Check browser console for connection logs
5. Verify skeleton overlay appears

---

## 🧪 Testing Checklist

- [ ] Femto Mega camera powers on
- [ ] OrbbecSDK_v2 detects camera
- [ ] Bridge server starts without errors
- [ ] WebSocket connection establishes
- [ ] Skeleton data streams at 30fps
- [ ] Browser displays skeleton overlay
- [ ] Joint angles calculate correctly
- [ ] Analysis saves to database

---

## 🔒 Security Considerations

### **For Production Deployment:**

1. **Use WSS (Secure WebSocket)**
   ```python
   import ssl
   
   ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
   ssl_context.load_cert_chain('cert.pem', 'key.pem')
   
   async with websockets.serve(
       self.handle_client, 
       self.host, 
       self.port,
       ssl=ssl_context
   ):
       ...
   ```

2. **Add Authentication**
   - API key validation
   - JWT token verification
   - IP whitelisting

3. **CORS Configuration**
   - Allow only trusted origins
   - Validate WebSocket origin

---

## 📊 Performance Optimization

### **Recommended Settings:**

```yaml
# config.yaml
camera:
  depth_resolution: 1024x1024
  color_resolution: 1920x1080
  frame_rate: 30
  
tracking:
  max_bodies: 1
  detection_threshold: 0.5
  
streaming:
  compression: true
  target_fps: 30
  max_latency_ms: 100
```

---

## 🐛 Troubleshooting

### **Camera Not Detected:**
```bash
# Linux: Check USB permissions
lsusb | grep Orbbec
sudo chmod 666 /dev/bus/usb/XXX/YYY

# Windows: Check Device Manager
# Look for "Orbbec Depth Camera"
```

### **Body Tracking Fails:**
```bash
# Verify Azure Kinect SDK installation
k4abt_simple_3d_viewer

# Check GPU acceleration
nvidia-smi  # Should show GPU usage
```

### **WebSocket Connection Fails:**
```bash
# Check firewall
sudo ufw allow 8765/tcp

# Test connection
wscat -c ws://localhost:8765
```

---

## 📚 Next Steps

### **To Complete Full Integration:**

1. **Implement Azure Kinect Body Tracking** in `body_tracker.py`
2. **Add multi-camera sync** for bilateral views
3. **Implement depth-based filtering** for better accuracy
4. **Add calibration wizard** for camera setup
5. **Create recording/replay** functionality
6. **Add bandwidth optimization** for remote access

---

## 📖 Additional Resources

### **Official Documentation:**
- OrbbecSDK_v2: https://github.com/orbbec/OrbbecSDK_v2
- Azure Kinect Body Tracking: https://learn.microsoft.com/azure-kinect
- WebSocket Protocol: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API

### **Community Resources:**
- Orbbec Forum: https://community.orbbec3d.com/
- Azure Kinect Samples: https://github.com/microsoft/Azure-Kinect-Samples

---

## ✅ Summary

**Current Status:**
- ✅ Documentation complete
- ✅ Architecture designed
- ✅ Code templates provided
- ⚠️ Bridge server needs implementation
- ⚠️ Azure Kinect SDK needs integration

**Estimated Implementation Time:**
- Bridge Server Setup: 4-6 hours
- Body Tracking Integration: 8-12 hours
- Testing & Calibration: 4-8 hours
- **Total: 2-3 days of development**

---

**Last Updated:** October 21, 2025  
**Status:** 📝 Integration Guide Complete  
**Implementation:** ⚠️ Requires Developer Effort
