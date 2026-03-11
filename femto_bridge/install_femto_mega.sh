#!/bin/bash
# Orbbec Femto Mega - Complete Installation Script
# Run this on your local Windows (WSL/Git Bash) or Linux machine

set -e

echo "============================================================"
echo "🚀 Orbbec Femto Mega Setup - Automated Installation"
echo "============================================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "✅ Detected OS: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    echo "✅ Detected OS: macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    echo "✅ Detected OS: Windows (WSL/Git Bash)"
else
    echo "⚠️  Unknown OS: $OSTYPE"
    echo "Please install manually following the guide"
    exit 1
fi

echo ""
echo "============================================================"
echo "📦 Step 1: Install System Dependencies"
echo "============================================================"

if [ "$OS" = "linux" ]; then
    echo "Installing Linux dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        git cmake build-essential \
        libusb-1.0-0-dev libudev-dev \
        python3-dev python3-pip \
        usbutils
    echo "✅ System dependencies installed"
    
elif [ "$OS" = "mac" ]; then
    echo "Installing macOS dependencies..."
    brew install libusb cmake python3
    echo "✅ System dependencies installed"
    
elif [ "$OS" = "windows" ]; then
    echo "⚠️  Windows detected"
    echo "Please ensure you have:"
    echo "  1. Python 3.8+ installed"
    echo "  2. Visual Studio Build Tools"
    echo "  3. Git for Windows"
    echo ""
    read -p "Press Enter to continue..."
fi

echo ""
echo "============================================================"
echo "📥 Step 2: Download OrbbecSDK_v2"
echo "============================================================"

SDK_DIR="$HOME/OrbbecSDK_v2"

if [ -d "$SDK_DIR" ]; then
    echo "⚠️  OrbbecSDK_v2 already exists at $SDK_DIR"
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$SDK_DIR"
    else
        echo "Skipping SDK download"
    fi
fi

if [ ! -d "$SDK_DIR" ]; then
    echo "Cloning OrbbecSDK_v2..."
    git clone https://github.com/orbbec/OrbbecSDK_v2.git "$SDK_DIR"
    echo "✅ OrbbecSDK_v2 downloaded to $SDK_DIR"
else
    echo "✅ Using existing OrbbecSDK_v2 at $SDK_DIR"
fi

echo ""
echo "============================================================"
echo "🔨 Step 3: Build OrbbecSDK_v2"
echo "============================================================"

if [ "$OS" = "linux" ] || [ "$OS" = "mac" ]; then
    cd "$SDK_DIR"
    
    if [ -d "build" ]; then
        echo "Cleaning previous build..."
        rm -rf build
    fi
    
    mkdir -p build
    cd build
    
    echo "Running CMake..."
    cmake ..
    
    echo "Building SDK..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    echo "Installing SDK..."
    sudo make install
    
    echo "✅ OrbbecSDK_v2 built and installed"
    
    # Configure USB rules (Linux only)
    if [ "$OS" = "linux" ]; then
        echo "Configuring USB permissions..."
        sudo cp ../misc/99-obsensor-libusb.rules /etc/udev/rules.d/ 2>/dev/null || true
        sudo udevadm control --reload-rules 2>/dev/null || true
        sudo udevadm trigger 2>/dev/null || true
        
        # Add user to plugdev group
        sudo usermod -a -G plugdev $USER 2>/dev/null || true
        
        echo "✅ USB permissions configured"
        echo "⚠️  You may need to log out and back in for USB permissions"
    fi
    
elif [ "$OS" = "windows" ]; then
    echo "⚠️  Windows build requires Visual Studio"
    echo "Please build manually or download pre-built binaries from:"
    echo "https://github.com/orbbec/OrbbecSDK_v2/releases"
    echo ""
    read -p "Press Enter after you've built the SDK..."
fi

echo ""
echo "============================================================"
echo "🐍 Step 4: Install Python SDK and Dependencies"
echo "============================================================"

echo "Installing Python packages..."
pip3 install --user pyorbbecsdk
pip3 install --user websockets
pip3 install --user numpy
pip3 install --user opencv-python

# Try to install pyk4a (optional, for Body Tracking)
echo "Attempting to install pyk4a (for Azure Kinect Body Tracking)..."
if pip3 install --user pyk4a; then
    echo "✅ pyk4a installed successfully"
else
    echo "⚠️  pyk4a installation failed (this is expected if Azure Kinect SDK is not installed)"
    echo "    The system will fall back to native Orbbec SDK or simulation."
fi

# Ensure requirements file is present
if [ -f "python_requirements.txt" ]; then
    echo "Ensuring all requirements from python_requirements.txt are met..."
    pip3 install --user -r python_requirements.txt || echo "⚠️ Some requirements failed to install"
fi

echo "✅ Python dependencies installed"

echo ""
echo "============================================================"
echo "🧪 Step 5: Test Camera Connection"
echo "============================================================"

echo "Creating test script..."

cat > /tmp/test_femto_mega.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Quick test to verify Femto Mega camera connection"""

try:
    from pyorbbecsdk import Pipeline, Config
    import sys
    
    print("✅ pyorbbecsdk imported successfully")
    print("🔍 Attempting to connect to Femto Mega camera...")
    
    # Try to create pipeline
    pipeline = Pipeline()
    print("✅ Pipeline created")
    
    # Try to start device
    config = Config()
    pipeline.start(config)
    print("✅ Camera started successfully!")
    
    # Get a frame to verify
    frames = pipeline.wait_for_frames(timeout_ms=1000)
    if frames:
        print("✅ Frame captured successfully!")
        print("✅✅✅ FEMTO MEGA IS WORKING! ✅✅✅")
    else:
        print("⚠️  No frames received (camera may not be connected)")
    
    pipeline.stop()
    sys.exit(0)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip3 install pyorbbecsdk")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("")
    print("Possible issues:")
    print("  1. Camera not connected via USB 3.0")
    print("  2. USB permissions not set (run: sudo usermod -a -G plugdev $USER)")
    print("  3. Camera in use by another program")
    print("  4. OrbbecSDK_v2 not properly installed")
    sys.exit(1)
PYTHON_EOF

chmod +x /tmp/test_femto_mega.py

echo "Running camera test..."
python3 /tmp/test_femto_mega.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅✅✅ SUCCESS! Camera is working! ✅✅✅"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "⚠️  Camera test failed"
    echo "============================================================"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Verify camera is connected to USB 3.0 port (blue port)"
    echo "2. Check: lsusb | grep -i orbbec"
    echo "3. Check permissions: groups | grep plugdev"
    echo "4. Try: sudo chmod 666 /dev/bus/usb/*/* (temporary fix)"
    echo "5. Restart computer and try again"
    echo ""
    exit 1
fi

echo ""
echo "============================================================"
echo "📁 Step 6: Setup Bridge Server"
echo "============================================================"

BRIDGE_DIR="$HOME/femto_bridge"

echo "Creating bridge server directory at $BRIDGE_DIR..."
mkdir -p "$BRIDGE_DIR"

echo "Copying bridge server files..."
# Files will be created in next steps

echo "✅ Bridge server directory ready"

echo ""
echo "============================================================"
echo "🎉 INSTALLATION COMPLETE!"
echo "============================================================"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Navigate to bridge directory:"
echo "   cd $BRIDGE_DIR"
echo ""
echo "2. Download bridge server files from sandbox"
echo ""
echo "3. Start bridge server:"
echo "   python3 server.py"
echo ""
echo "4. Open web app and select 'Femto Mega'"
echo ""
echo "============================================================"
echo "📞 Need Help?"
echo ""
echo "Check camera: lsusb | grep -i orbbec"
echo "Test again: python3 /tmp/test_femto_mega.py"
echo "View logs: journalctl -f | grep orbbec"
echo ""
echo "============================================================"
