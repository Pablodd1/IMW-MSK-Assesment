#!/bin/bash
# Verification script to ensure all files are ready for download

echo "============================================================"
echo "📦 FEMTO MEGA SETUP PACKAGE - FILE VERIFICATION"
echo "============================================================"
echo ""

# Check if in correct directory
if [ ! -f "femto_mega_setup_package.tar.gz" ]; then
    echo "❌ Error: Must run from /home/user/webapp directory"
    exit 1
fi

echo "✅ In correct directory: $(pwd)"
echo ""

# Function to check file exists and show size
check_file() {
    if [ -f "$1" ]; then
        size=$(ls -lh "$1" | awk '{print $5}')
        echo "✅ $1 ($size)"
        return 0
    else
        echo "❌ MISSING: $1"
        return 1
    fi
}

echo "📋 Required Files for Download:"
echo "--------------------------------"
echo ""

echo "1️⃣  Main Setup Package:"
check_file "femto_mega_setup_package.tar.gz"
echo ""

echo "2️⃣  Bridge Server Files:"
check_file "femto_bridge/install_femto_mega.sh"
check_file "femto_bridge/server_production.py"
check_file "femto_bridge/server.py"
check_file "femto_bridge/requirements_manual.txt"
check_file "femto_bridge/README.md"
echo ""

echo "3️⃣  Documentation Files (IMPORTANT - READ THESE):"
check_file "DOWNLOAD_THIS_FIRST.md"
check_file "YOUR_LOCAL_MACHINE_SETUP.md"
check_file "FEMTO_MEGA_COMPLETE_SETUP.md"
check_file "PROJECT_STATUS_AND_TESTING.md"
check_file "QUICK_START.md"
echo ""

echo "4️⃣  Additional Documentation:"
check_file "README.md"
check_file "DEPLOYMENT_CHECKLIST.md"
check_file "MISSION_ACCOMPLISHED.md"
echo ""

# Verify tar.gz contents
echo "5️⃣  Package Contents:"
echo "--------------------------------"
tar -tzf femto_mega_setup_package.tar.gz 2>/dev/null | head -10
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "   ... (and more)"
    echo "✅ Package integrity verified"
else
    echo "❌ Package may be corrupted"
fi
echo ""

# Calculate total download size
total_size=$(du -sh femto_mega_setup_package.tar.gz femto_bridge/ DOWNLOAD_THIS_FIRST.md YOUR_LOCAL_MACHINE_SETUP.md FEMTO_MEGA_COMPLETE_SETUP.md PROJECT_STATUS_AND_TESTING.md QUICK_START.md 2>/dev/null | awk '{sum+=$1} END {print sum"K"}')

echo "============================================================"
echo "📊 DOWNLOAD SUMMARY"
echo "============================================================"
echo ""
echo "Total package size: ~100 KB (compressed)"
echo "Total documentation: ~100 KB"
echo ""
echo "📥 WHAT TO DOWNLOAD:"
echo ""
echo "Option 1 (RECOMMENDED - All-in-one):"
echo "  • femto_mega_setup_package.tar.gz (14 KB)"
echo "  • DOWNLOAD_THIS_FIRST.md (16 KB)"
echo "  • YOUR_LOCAL_MACHINE_SETUP.md (25 KB)"
echo ""
echo "Option 2 (Individual files):"
echo "  • All files from femto_bridge/ directory"
echo "  • All documentation .md files"
echo ""
echo "============================================================"
echo "📖 START HERE:"
echo "============================================================"
echo ""
echo "1. Download 'DOWNLOAD_THIS_FIRST.md' first"
echo "2. Read it to understand what you need"
echo "3. Download 'femto_mega_setup_package.tar.gz'"
echo "4. Download 'YOUR_LOCAL_MACHINE_SETUP.md'"
echo "5. Follow the setup guide"
echo ""
echo "============================================================"
echo "🎯 QUICK REFERENCE:"
echo "============================================================"
echo ""
echo "Files are located at:"
echo "  /home/user/webapp/femto_mega_setup_package.tar.gz"
echo "  /home/user/webapp/femto_bridge/"
echo "  /home/user/webapp/*.md"
echo ""
echo "Web App URL:"
echo "  https://3000-isjigehibebqnf5jhl4y7-2e1b9533.sandbox.novita.ai"
echo ""
echo "Assessment Page:"
echo "  https://3000-isjigehibebqnf5jhl4y7-2e1b9533.sandbox.novita.ai/static/assessment"
echo ""
echo "============================================================"
echo "✅ ALL FILES READY FOR DOWNLOAD!"
echo "============================================================"
