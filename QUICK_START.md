# 🚀 QUICK START GUIDE - Test Everything Locally

## ✅ Option 1: Test Laptop Camera (Simplest)

**Time:** 5 minutes
**Requirements:** Just a webcam

### **Steps:**

1. **Open the web app:**
   ```
   https://3000-isjigehibebqnf5jhl4y7-2e1b9533.sandbox.novita.ai/static/assessment
   ```

2. **Click "Laptop Camera"**

3. **Allow camera access** when browser prompts

4. **See your video** with skeleton overlay (RED joints + YELLOW lines)

5. **Click "Start Recording"**

6. **Do a squat** (5-10 seconds)

7. **Click "Stop Recording"**

8. **View analysis results:**
   - Joint angles
   - Movement deficiencies
   - Quality score (0-100)
   - Exercise recommendations

---

## ✅ Option 2: Test Femto Mega Simulation

**Time:** 10 minutes
**Requirements:** None - all simulated

### **Steps:**

**Terminal 1 - Start Bridge Server:**
```bash
cd /home/user/webapp
./start_femto_bridge.sh

# You should see:
# ============================================================
# 🚀 Femto Mega Bridge Server
# ============================================================
# 📷 Running in SIMULATION mode (no camera required)
# ✅ Server ready at ws://0.0.0.0:8765
# 🎥 Starting skeleton data stream...
```

**Browser - Connect Web App:**
```
1. Open: https://3000-isjigehibebqnf5jhl4y7-2e1b9533.sandbox.novita.ai/static/assessment

2. Click "Femto Mega" button

3. Should see:
   ✅ "Connected to Femto Mega bridge"
   ✅ Green skeleton overlay
   ✅ 32 joints tracked
   ✅ Simulated squat movement

4. Click "Start Recording"

5. Watch simulated person squat

6. Click "Stop Recording" after 5-10 seconds

7. View analysis with depth data (Z axis)
```

---

## 🧪 Verify Everything Works

### **Checklist:**

```
✅ Web app accessible
✅ Patient intake form submits
✅ Laptop camera connects
✅ MediaPipe skeleton tracking works
✅ Movement recording works
✅ Analysis calculates joint angles
✅ Deficiencies detected
✅ Exercise recommendations generated
✅ Femto bridge server starts
✅ WebSocket connection works
✅ Simulated skeleton streams
```

---

## 📊 Expected Results

### **Laptop Camera Output:**
```json
{
  "movement_quality_score": 72,
  "tracking_method": "MediaPipe",
  "joints_tracked": 33,
  "deficiencies": [
    {
      "area": "Hip Flexion",
      "severity": "moderate",
      "angle_detected": 95,
      "normal_range": 120
    }
  ]
}
```

### **Femto Mega Output:**
```json
{
  "movement_quality_score": 75,
  "tracking_method": "Azure Kinect",
  "joints_tracked": 32,
  "has_depth_data": true,
  "simulation_mode": true,
  "deficiencies": [
    {
      "area": "Hip Flexion",
      "severity": "moderate",
      "angle_detected": 95,
      "depth_accuracy": "±2mm"
    }
  ]
}
```

---

## 🔧 Troubleshooting

### **Problem: Camera access denied**
```
Solution:
1. Click address bar camera icon
2. Select "Always allow"
3. Refresh page
```

### **Problem: Skeleton not showing**
```
Solution:
1. Open browser console (F12)
2. Look for MediaPipe loading messages
3. Ensure good lighting
4. Stand 1-2 meters from camera
```

### **Problem: Femto bridge won't connect**
```
Solution:
1. Check bridge server running: ps aux | grep server.py
2. Check port 8765: lsof -i :8765
3. Restart bridge: ./start_femto_bridge.sh
```

---

## 🎯 What You Can Test Right NOW

✅ **Patient intake workflow**
✅ **Laptop camera tracking** (33 joints, MediaPipe)
✅ **Phone camera tracking** (front/back flip)
✅ **Femto Mega simulation** (32 joints, simulated depth)
✅ **Movement recording** (5-10 seconds)
✅ **Biomechanical analysis** (joint angles, ROM)
✅ **Deficiency detection** (automated identification)
✅ **Exercise prescription** (17 exercises in database)
✅ **Dashboard monitoring** (patient progress)

---

## 🚫 What You CANNOT Test (Requires Hardware)

❌ **Real Femto Mega camera** - Needs physical device + external workstation
❌ **Clinical-grade 3D measurements** - Requires actual depth sensor
❌ **Multi-camera sync** - Needs multiple Femto Mega cameras

---

## 💡 Next Steps

1. **Test laptop camera workflow** (5 min) ← START HERE
2. **Test Femto simulation** (10 min)
3. **Review analysis results**
4. **Test exercise prescription**
5. **Optional: Integrate your AI API**
6. **Optional: Deploy to Cloudflare Pages**

---

## 📞 Quick Commands

```bash
# Start web app (if not running)
cd /home/user/webapp && pm2 start ecosystem.config.cjs

# Start Femto bridge
./start_femto_bridge.sh

# Check services
pm2 list
ps aux | grep server.py

# View logs
pm2 logs webapp --nostream

# Restart everything
pm2 restart webapp
pkill -f server.py && ./start_femto_bridge.sh &
```

---

**Ready to start testing? Begin with Option 1 (Laptop Camera) - it's the simplest!** 🚀
