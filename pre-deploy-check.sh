#!/bin/bash

# Pre-deployment check script

echo "=========================================="
echo "    PhysioMotion Pre-Deployment Audit"
echo "=========================================="

# 1. Check for Critical Incompatibilities
echo ""
echo "[1/4] Checking Environment Compatibility..."
if grep -q "c.env.DB" src/index.tsx; then
    echo "⚠️  CRITICAL WARNING: Cloudflare D1 bindings detected."
    echo "    This application IS NOT COMPATIBLE with Vercel Node.js runtime as-is."
    echo "    The code uses 'c.env.DB' which is specific to Cloudflare Workers."
    echo "    See 'vercel-env-setup.md' and 'PRE-DEPLOYMENT-REPORT.md' for details."
else
    echo "✅ Environment checks passed."
fi

# 2. Build
echo ""
echo "[2/4] Building Project..."
npm run build
if [ $? -eq 0 ]; then
    echo "✅ Build Successful"
else
    echo "❌ Build Failed"
    exit 1
fi

# 3. Security Scan
echo ""
echo "[3/4] Scanning for Secrets..."
# We filter out known safe patterns like password_hash variable names
SECRETS=$(grep -rE "key|secret|token|password" src/ | grep -v "password_hash" | grep -v "verifyPassword" | grep -v "hashPassword" | grep -v "physiomotion-salt-2025")

if [ -n "$SECRETS" ]; then
    echo "⚠️  Potential secrets found:"
    echo "$SECRETS"
else
    echo "✅ No obvious secrets found."
fi

# Check for hardcoded salt specifically
if grep -q "physiomotion-salt-2025" src/index.tsx; then
     echo "⚠️  WARNING: Hardcoded password salt detected in src/index.tsx ('physiomotion-salt-2025')."
     echo "    Recommendation: Move this to an environment variable."
fi

# 4. Endpoint Tests
echo ""
echo "[4/4] Testing Endpoints..."
echo "Note: This step requires the server to be running locally on port 3000."
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null ; then
    node test-endpoints.cjs
else
    echo "⚠️  Skipping endpoint tests: No server detected on port 3000."
    echo "    Run 'npm run dev' in another terminal to enable this check."
fi

echo ""
echo "=========================================="
echo "    Audit Complete"
echo "=========================================="
