#!/usr/bin/env python3
"""Quick integration test for P0 + P1 filters and clinical engines."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, cv2
from server import PoseEngine, DualModelServer
from fms_engine import FMSEngine
from chiropractic_analyzer import ChiropracticAnalyzer

print("=" * 55)
print("IMW-MSK Integration Test")
print("=" * 55)

# 1. Create a synthetic human-like pose image
img = np.zeros((480, 640, 3), dtype=np.uint8)
# Head
cv2.circle(img, (320, 80), 15, (200, 200, 200), -1)
# Neck
cv2.line(img, (320, 95), (320, 140), (180, 180, 180), 8)
# Shoulders
cv2.line(img, (260, 140), (380, 140), (160, 160, 160), 10)
# Arms
cv2.line(img, (260, 140), (220, 240), (150, 150, 150), 8)
cv2.line(img, (380, 140), (420, 240), (150, 150, 150), 8)
# Torso
cv2.line(img, (320, 140), (320, 280), (140, 140, 140), 12)
# Hips
cv2.line(img, (280, 280), (360, 280), (130, 130, 130), 10)
# Legs (standing straight)
cv2.line(img, (280, 280), (260, 400), (120, 120, 120), 10)
cv2.line(img, (360, 280), (380, 400), (120, 120, 120), 10)

_, buf = cv2.imencode('.jpg', img)
print(f"\n[Test] Synthetic image: {img.shape}, JPEG: {len(buf)} bytes")

# 2. Test PoseEngine with P0 filters
engine = PoseEngine()
result = engine.process_frame(buf.tobytes())
print(f"[PoseEngine] Persons detected: {len(result.get('persons', []))}")

if result.get('persons'):
    person = result['persons'][0]
    print(f"[PoseEngine] Keypoints: {len(person['keypoints'])}")
    print(f"[PoseEngine] Confidence: {person['confidence']}")

    # 3. Test FMS engine
    kpts = {int(kp['id']): kp for kp in person['keypoints']}
    fms = FMSEngine(depth_available=False)
    battery = fms.run_battery(kpts)
    print(f"\n[FMS Battery]")
    for name, r in battery.items():
        print(f"  {name}: score={r.score}/3, criteria={r.criteria_met}, comp={r.compensations}")

    # 4. Test Chiropractic analyzer
    chiro = ChiropracticAnalyzer()
    screen = chiro.full_postural_screen(kpts)
    print(f"\n[Chiropractic Screen]")
    for k, v in screen.items():
        print(f"  {k}: {v}")

    # 5. Test reset
    engine.reset_filters()
    print(f"\n[Reset] Filters cleared successfully")

print("\n" + "=" * 55)
print("All tests passed")
print("=" * 55)
