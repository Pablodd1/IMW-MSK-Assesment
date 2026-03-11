# 🎥 Femto Mega Bridge Server

WebSocket server for streaming Orbbec Femto Mega skeleton data to PhysioMotion web app.

## 🚀 Quick Start

### **Option 1: Simulation Mode (No Camera Required)**

Test the WebSocket integration without actual hardware:

```bash
cd /home/user/webapp/femto_bridge

# Install Python dependencies
pip install -r requirements_manual.txt

# Run server in simulation mode
python server.py --simulate
```

**What happens:**
- ✅ WebSocket server starts on `ws://localhost:8765`
- ✅ Generates simulated 32-joint skeleton data
- ✅ Simulates squat movement (person moving up/down)
- ✅ Web app can connect and receive data

### **Option 2: Real Femto Mega Camera**

For production use with actual Orbbec Femto Mega camera:

```bash
# 1. Install OrbbecSDK_v2
git clone https://github.com/orbbec/OrbbecSDK_v2.git
cd OrbbecSDK_v2
# Follow installation instructions for your OS

# 2. Install Python SDK
pip install pyorbbecsdk

# 3. Connect Femto Mega via USB 3.0

# 4. Install Azure Kinect Body Tracking SDK (for skeleton tracking)
# Windows: Download from Microsoft
# Linux: sudo apt-get install libk4abt1.1-dev

# 5. Start server
cd /home/user/webapp/femto_bridge
python server.py
```

---

## 📡 WebSocket Protocol

### **Connection**

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => {
  console.log('Connected to Femto Mega bridge');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'connected') {
    console.log('Bridge server ready');
    console.log('Simulation mode:', data.simulation);
  }
  
  if (data.type === 'skeleton') {
    const skeleton = data.skeleton;
    console.log('Body ID:', skeleton.body_id);
    console.log('Joints:', Object.keys(skeleton.joints).length);
    // Process 32-joint data
  }
};
```

### **Commands**

```javascript
// Start streaming
ws.send(JSON.stringify({ command: 'start_streaming' }));

// Stop streaming
ws.send(JSON.stringify({ command: 'stop_streaming' }));

// Ping
ws.send(JSON.dumps({ command: 'ping' }));
```

### **Skeleton Data Format**

```json
{
  "type": "skeleton",
  "skeleton": {
    "timestamp": "2025-01-30T22:15:30.123456",
    "body_id": 0,
    "simulation": false,
    "joints": {
      "PELVIS": {
        "position": { "x": 0.0, "y": 500.0, "z": 1500.0 },
        "orientation": { "w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0 },
        "confidence": "HIGH"
      },
      "SPINE_NAVAL": { ... },
      "SPINE_CHEST": { ... },
      ... (32 joints total)
    }
  }
}
```

---

## 🦴 Joint Names (32 Joints)

Based on Azure Kinect Body Tracking SDK:

```
Torso (4):
- PELVIS, SPINE_NAVAL, SPINE_CHEST, NECK

Head (6):
- HEAD, NOSE, EYE_LEFT, EAR_LEFT, EYE_RIGHT, EAR_RIGHT

Left Arm (7):
- CLAVICLE_LEFT, SHOULDER_LEFT, ELBOW_LEFT, WRIST_LEFT
- HAND_LEFT, HANDTIP_LEFT, THUMB_LEFT

Right Arm (7):
- CLAVICLE_RIGHT, SHOULDER_RIGHT, ELBOW_RIGHT, WRIST_RIGHT
- HAND_RIGHT, HANDTIP_RIGHT, THUMB_RIGHT

Left Leg (4):
- HIP_LEFT, KNEE_LEFT, ANKLE_LEFT, FOOT_LEFT

Right Leg (4):
- HIP_RIGHT, KNEE_RIGHT, ANKLE_RIGHT, FOOT_RIGHT
```

---

## 🧪 Testing

### **Test 1: Server Start**

```bash
python server.py --simulate

# Expected output:
# ============================================================
# 🚀 Femto Mega Bridge Server
# ============================================================
# 📷 Running in SIMULATION mode (no camera required)
# 📡 Starting WebSocket server on 0.0.0.0:8765
# ✅ Server ready at ws://0.0.0.0:8765
# 👉 Open PhysioMotion web app and select 'Femto Mega' camera
# ============================================================
# 📊 SIMULATION MODE ACTIVE
#    - Generating simulated skeleton data
# ============================================================
# 🎥 Starting skeleton data stream...
```

### **Test 2: Client Connection**

```bash
# In another terminal, test with websocat:
websocat ws://localhost:8765

# Or use Python:
python -c "
import asyncio
import websockets

async def test():
    async with websockets.connect('ws://localhost:8765') as ws:
        msg = await ws.recv()
        print('Received:', msg)

asyncio.run(test())
"
```

### **Test 3: Web App Integration**

```
1. Start bridge server: python server.py --simulate
2. Start web app: cd /home/user/webapp && pm2 start ecosystem.config.cjs
3. Open: https://3000-xxx.sandbox.novita.ai/static/assessment
4. Select "Femto Mega" camera
5. Should connect and show green skeleton overlay
```

---

## 🛠️ Troubleshooting

### **Port Already in Use**

```bash
# Check what's using port 8765
lsof -i :8765

# Kill existing process
fuser -k 8765/tcp
```

### **Camera Not Detected**

```bash
# Check USB connection
lsusb | grep Orbbec

# Check SDK installation
python -c "import pyorbbecsdk; print(pyorbbecsdk.__version__)"

# Check camera permissions (Linux)
sudo usermod -a -G plugdev $USER
```

### **Body Tracking Not Working**

```bash
# Verify Azure Kinect SDK installation
# Windows: Check C:\Program Files\Azure Kinect Body Tracking SDK
# Linux: dpkg -l | grep k4abt
```

---

## 📊 Performance

- **Frame Rate**: 30 FPS
- **Latency**: ~50-100ms (local network)
- **Joint Count**: 32
- **Confidence Levels**: HIGH, MEDIUM, LOW, NONE

---

## 🔒 Security

**For Production:**

```bash
# Use SSL/TLS
python server.py --ssl-cert /path/to/cert.pem --ssl-key /path/to/key.pem

# Restrict to localhost only
python server.py --host 127.0.0.1

# Use authentication token
python server.py --token YOUR_SECRET_TOKEN
```

---

## 📝 Logs

Logs are written to console with timestamps:

```
2025-01-30 22:15:30 - INFO - Server ready at ws://0.0.0.0:8765
2025-01-30 22:15:35 - INFO - ✅ Client connected from ('192.168.1.100', 54321)
2025-01-30 22:15:40 - INFO - 🎥 Starting skeleton data stream...
```

---

## 📞 Support

- Orbbec SDK: https://github.com/orbbec/OrbbecSDK_v2
- Azure Kinect Body Tracking: https://learn.microsoft.com/azure/kinect-dk/body-sdk-download
- WebSocket Protocol: https://websockets.readthedocs.io/
