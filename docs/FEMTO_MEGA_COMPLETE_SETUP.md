# 🚀 COMPLETE FEMTO MEGA SETUP GUIDE

## 📋 Overview

This guide will help you set up the Orbbec Femto Mega camera for use with PhysioMotion. Since the camera must be connected to your local machine (not the sandbox), we'll set up a two-machine architecture.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────┐
│  YOUR LOCAL MACHINE                  │
│  (Windows/Linux/Mac)                 │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Femto Mega Camera (USB 3.0)   │ │
│  └────────────────────────────────┘ │
│              ↓                       │
│  ┌────────────────────────────────┐ │
│  │  OrbbecSDK_v2                  │ │
│  │  + pyorbbecsdk                 │ │
│  └────────────────────────────────┘ │
│              ↓                       │
│  ┌────────────────────────────────┐ │
│  │  Python Bridge Server          │ │
│  │  ws://YOUR_IP:8765             │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
              │
              │ WebSocket
              ↓
┌──────────────────────────────────────┐
│  SANDBOX / CLOUD                     │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  PhysioMotion Web App          │ │
│  │  (Hono + Cloudflare D1)        │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
```

---

## ✅ OPTION 1: Quick Automated Setup (Recommended)

### **Step 1: Download Setup Package**

Download all files from the sandbox to your local machine:

```bash
# On your local machine, create project directory
mkdir -p ~/femto_mega_setup
cd ~/femto_mega_setup

# Download the installation script
# (You'll need to copy this file from the sandbox)
```

**Files to download from `/home/user/webapp/femto_bridge/`:**
1. `install_femto_mega.sh` - Automated installation script
2. `server_production.py` - Production bridge server
3. `requirements.txt` - Python dependencies
4. `README.md` - Documentation

### **Step 2: Run Automated Installer**

```bash
# Make script executable
chmod +x install_femto_mega.sh

# Run installer (will prompt for sudo password)
./install_femto_mega.sh

# Follow the prompts
# The script will:
# ✅ Install system dependencies
# ✅ Download OrbbecSDK_v2
# ✅ Build and install SDK
# ✅ Install Python packages
# ✅ Configure USB permissions
# ✅ Test camera connection
```

### **Step 3: Start Bridge Server**

```bash
# Navigate to bridge directory
cd ~/femto_bridge

# Start server with real camera
python3 server_production.py

# You should see:
# ============================================================
# 🚀 Femto Mega Bridge Server v2.0
# ============================================================
# ✅ Femto Mega initialized successfully
#    - Depth: 640x576 @ 30fps
#    - Color: 1920x1080 @ 30fps
# ✅ Server ready at ws://0.0.0.0:8765
# 📷 REAL CAMERA MODE
#    - Femto Mega connected and ready
# ============================================================
# 🎥 Starting skeleton data stream...
```

### **Step 4: Update Web App Connection**

In the web app, you need to update the WebSocket URL to point to your machine:

```javascript
// Find your local machine IP address
// Linux/Mac: ip addr show or ifconfig
// Windows: ipconfig

// Example: If your IP is 192.168.1.100
// Change in assessment-workflow.js:
const femtoClient = new FemtoMegaClient('ws://192.168.1.100:8765');
```

### **Step 5: Test Connection**

```
1. Open web app: https://3000-xxx.sandbox.novita.ai/static/assessment
2. Click "Femto Mega" button
3. Should connect to your local bridge server
4. See skeleton overlay streaming at 30 FPS
```

---

## ✅ OPTION 2: Manual Step-by-Step Setup

### **Prerequisites**

**Operating System:**
- ✅ Linux (Ubuntu 18.04+, Debian 10+)
- ✅ Windows 10/11 (64-bit)
- ✅ macOS 10.15+ (limited support)

**Hardware:**
- Orbbec Femto Mega camera
- USB 3.0 port (blue port)
- 8GB+ RAM
- Intel i5 or better CPU

**Software:**
- Python 3.8+
- Git
- CMake 3.14+
- C++ compiler (GCC 7+ / MSVC 2019+ / Clang 9+)

---

### **Step 1: Install System Dependencies**

#### **Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y \
    git cmake build-essential \
    libusb-1.0-0-dev libudev-dev \
    python3-dev python3-pip \
    usbutils pkg-config

# Add user to plugdev group for USB access
sudo usermod -a -G plugdev $USER

# Log out and back in for group changes to take effect
```

#### **Windows:**
```powershell
# Install chocolatey package manager
# (Run PowerShell as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install dependencies
choco install git cmake python visualstudio2019buildtools -y

# Restart PowerShell
```

#### **macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install git cmake python@3.11 libusb
```

---

### **Step 2: Download and Build OrbbecSDK_v2**

```bash
# Clone SDK repository
cd ~
git clone https://github.com/orbbec/OrbbecSDK_v2.git
cd OrbbecSDK_v2

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build (use all CPU cores)
# Linux/Mac:
make -j$(nproc)
# Windows (from Visual Studio Developer Command Prompt):
# msbuild OrbbecSDK.sln /p:Configuration=Release

# Install SDK
# Linux/Mac:
sudo make install
# Windows: Run installer from Release folder
```

---

### **Step 3: Configure USB Permissions (Linux only)**

```bash
# Copy udev rules
cd ~/OrbbecSDK_v2
sudo cp misc/99-obsensor-libusb.rules /etc/udev/rules.d/

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Verify camera is detected
lsusb | grep -i orbbec
# Should show: Bus XXX Device XXX: ID 2bc5:XXXX Orbbec 3D Technology...
```

---

### **Step 4: Install Python SDK**

```bash
# Install pip packages
pip3 install --user pyorbbecsdk
pip3 install --user websockets
pip3 install --user numpy
pip3 install --user opencv-python

# Verify installation
python3 -c "from pyorbbecsdk import Pipeline; print('✅ pyorbbecsdk installed successfully')"
```

---

### **Step 5: Test Camera Connection**

Create test script:

```python
#!/usr/bin/env python3
"""Test Femto Mega camera connection"""

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

try:
    print("🔍 Testing Femto Mega connection...")
    
    # Create pipeline
    pipeline = Pipeline()
    print("✅ Pipeline created")
    
    # Configure streams
    config = Config()
    config.enable_stream(OBSensorType.DEPTH_SENSOR, 640, 576, OBFormat.Y16, 30)
    config.enable_stream(OBSensorType.COLOR_SENSOR, 1920, 1080, OBFormat.RGB, 30)
    
    # Start pipeline
    pipeline.start(config)
    print("✅ Camera started")
    
    # Get a frame
    frames = pipeline.wait_for_frames(timeout_ms=1000)
    if frames:
        print("✅ Frame captured successfully!")
        print("✅✅✅ FEMTO MEGA IS WORKING! ✅✅✅")
        
        # Get frame details
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        
        if depth:
            print(f"   Depth: {depth.get_width()}x{depth.get_height()}")
        if color:
            print(f"   Color: {color.get_width()}x{color.get_height()}")
    else:
        print("⚠️  No frames received")
    
    pipeline.stop()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check camera is connected to USB 3.0 (blue port)")
    print("2. Check: lsusb | grep -i orbbec")
    print("3. Check USB permissions: groups | grep plugdev")
    print("4. Try: sudo chmod 666 /dev/bus/usb/*/*")
```

Run test:
```bash
python3 test_camera.py
```

---

### **Step 6: Setup Bridge Server**

```bash
# Create bridge directory
mkdir -p ~/femto_bridge
cd ~/femto_bridge

# Copy files from sandbox:
# - server_production.py
# - requirements.txt
# - README.md

# Install dependencies
pip3 install -r requirements.txt

# Make server executable
chmod +x server_production.py
```

---

### **Step 7: Start Bridge Server**

```bash
# Start server with real camera
python3 server_production.py

# Or start in simulation mode for testing
python3 server_production.py --simulate

# Start on custom port
python3 server_production.py --port 9000

# Enable debug logging
python3 server_production.py --debug
```

**Expected output:**
```
============================================================
🚀 Femto Mega Bridge Server v2.0
============================================================
📷 Initializing Femto Mega camera...
✅ Femto Mega initialized successfully
   - Depth: 640x576 @ 30fps
   - Color: 1920x1080 @ 30fps
   - Alignment: Depth-to-Color
📡 Starting WebSocket server on 0.0.0.0:8765
✅ Server ready at ws://0.0.0.0:8765
👉 Open PhysioMotion web app and select 'Femto Mega' camera
============================================================
📷 REAL CAMERA MODE
   - Femto Mega connected and ready
   - Streaming depth + color at 30 FPS
============================================================
🎥 Starting skeleton data stream...
```

---

### **Step 8: Update Web App Configuration**

You need to tell the web app where to find your bridge server.

#### **Option A: Modify Frontend Code (Temporary)**

Edit `assessment-workflow.js`:

```javascript
// Line 1017 - Change localhost to your machine's IP
async function initializeFemtoMega() {
  try {
    // Find your IP with: ip addr show (Linux) or ipconfig (Windows)
    const femtoClient = new FemtoMegaClient('ws://192.168.1.100:8765');
    // ... rest of code
```

#### **Option B: Environment Variable (Better)**

Add configuration option to frontend:

```javascript
// Get bridge server URL from environment or use default
const FEMTO_BRIDGE_URL = 
  window.FEMTO_BRIDGE_URL || 
  localStorage.getItem('femto_bridge_url') || 
  'ws://localhost:8765';

const femtoClient = new FemtoMegaClient(FEMTO_BRIDGE_URL);
```

Then in browser console:
```javascript
localStorage.setItem('femto_bridge_url', 'ws://192.168.1.100:8765');
```

---

### **Step 9: Test Complete Workflow**

```
1. Bridge server running on your machine ✅
2. Camera connected via USB 3.0 ✅
3. Web app open in browser ✅

Test Steps:
1. Open: https://3000-xxx.sandbox.novita.ai/static/assessment
2. Click "Femto Mega" button
3. Should see: "✅ Connected to Femto Mega bridge"
4. Green skeleton overlay should appear
5. Joints tracked in real-time (32 joints)
6. Click "Start Recording"
7. Perform movement (squat, reach, etc.)
8. Click "Stop Recording"
9. Analysis runs with depth data
10. Results show 3D measurements
```

---

## 🔧 Troubleshooting

### **Camera Not Detected**

```bash
# Check USB connection
lsusb | grep -i orbbec

# Should show:
# Bus 001 Device 005: ID 2bc5:XXXX Orbbec 3D Technology...

# If not shown:
1. Try different USB port (must be USB 3.0 - blue port)
2. Check cable is connected firmly
3. Try: sudo dmesg | tail -20
4. Restart computer
```

### **Permission Denied**

```bash
# Check groups
groups | grep plugdev

# If not in plugdev:
sudo usermod -a -G plugdev $USER

# Log out and back in

# Temporary fix:
sudo chmod 666 /dev/bus/usb/*/*
```

### **Bridge Server Won't Start**

```bash
# Check port availability
lsof -i :8765

# Kill existing process
fuser -k 8765/tcp

# Try different port
python3 server_production.py --port 9000
```

### **Web App Can't Connect**

```bash
# Check firewall
# Linux:
sudo ufw allow 8765
# Windows:
# Add inbound rule in Windows Firewall for port 8765

# Check bridge server is listening
netstat -an | grep 8765

# Test connection
curl ws://localhost:8765
# Or use websocat:
websocat ws://localhost:8765
```

### **No Skeleton Detected**

```
1. Check lighting - need good lighting for body tracking
2. Stand 1-2 meters from camera
3. Face camera directly
4. Ensure full body visible
5. Check bridge server logs for errors
```

---

## 📊 Performance Tuning

### **Improve Frame Rate**

```python
# In server_production.py, adjust resolution:
config.enable_stream(
    OBSensorType.DEPTH_SENSOR,
    320, 288,  # Lower resolution
    OBFormat.Y16,
    60  # Higher FPS
)
```

### **Reduce Latency**

```python
# Disable frame buffering
pipeline.wait_for_frames(timeout_ms=10)  # Reduce timeout

# Increase network buffer
# In WebSocket server:
websockets.serve(..., max_size=10**7, compression=None)
```

### **Better Tracking Quality**

```python
# Enable alignment for better color/depth sync
config.set_align_mode(OBAlignMode.ALIGN_D2C_HW_MODE)

# Use hardware acceleration if available
config.enable_hardware_d2c_align(True)
```

---

## 🎯 Next Steps

1. ✅ **Complete this setup** - Get camera working locally
2. ✅ **Test with web app** - Verify end-to-end connection
3. ⚠️ **Optional: Integrate body tracking** - Add Azure Kinect Body Tracking SDK for automatic skeleton detection
4. ⚠️ **Optional: Deploy to production** - Set up dedicated workstation for clinical use

---

## 📞 Support

**Common Issues:**
- Camera not detected → Check USB 3.0 connection
- Permission denied → Run: `sudo usermod -a -G plugdev $USER`
- Port in use → Kill process: `fuser -k 8765/tcp`
- Web app can't connect → Check firewall allows port 8765

**Resources:**
- OrbbecSDK: https://github.com/orbbec/OrbbecSDK_v2
- pyorbbecsdk: https://pypi.org/project/pyorbbecsdk/
- Femto Mega Docs: https://doc.orbbec.com/

---

**Ready to set up? Follow Option 1 (Automated) for fastest setup!** 🚀
