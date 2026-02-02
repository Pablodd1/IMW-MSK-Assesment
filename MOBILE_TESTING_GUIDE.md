# Mobile Testing Guide - PhysioMotion Assessment

## 📱 Testing on Your Phone

### Quick Access URLs
**Primary:** https://3000-isjigehibebqnf5jhl4y7-2e1b9533.sandbox.novita.ai/assessment

### Step-by-Step Mobile Testing

#### 1. Open the Assessment Page
- Open your mobile browser (Chrome, Safari, Firefox)
- Navigate to the URL above
- The page should load with:
  - Navigation bar at top
  - 4-step progress indicator
  - "Select Camera Type" modal in center

#### 2. Camera Selection Modal
You should see **4 camera options**:
- 📱 **Mobile Phone** ← Select this one for phone testing
- 💻 **Laptop Camera**
- 🎥 **Femto Mega**
- 📹 **Upload Video**

**Action:** Tap on "Mobile Phone" option
- The option should highlight with blue border
- "Start Assessment" button should become enabled (blue)

#### 3. Start Assessment
**Action:** Tap "Start Assessment" button
- Modal should close
- Browser will ask for camera permission
- **IMPORTANT:** Allow camera access when prompted

#### 4. Camera Preview
After granting permission, you should see:
- ✅ Live video feed from your camera
- ✅ Default starts with **front camera** (selfie mode)
- ✅ Green "Connected" indicator in top-right
- ✅ Progress updated to "Step 2: Position Patient"

#### 5. Camera Controls (Bottom of Screen)
You should see these buttons:
- 🔄 **Flip Camera** - Switch between front/back camera
- ▶️ **Start Recording** - Begin motion capture
- ⏹️ **Stop Recording** - End capture (appears after starting)
- 🔍 **Analyze Movement** - Process skeleton data (appears after stopping)

#### 6. Test Flip Camera
**Action:** Tap "Flip Camera" button
- Camera should switch from front to back (or back to front)
- Video feed should update smoothly
- **Tip:** Back camera usually has better quality

#### 7. Position Yourself in Frame
**What to look for:**
- Position camera so your whole body is visible
- Stand 6-8 feet away from camera
- Ensure good lighting
- When MediaPipe detects you:
  - **RED circles** appear on your joints
  - **YELLOW lines** connect the joints
  - Skeleton tracking follows your movement

#### 8. Start Recording
**Action:** Tap "Start Recording"
- Recording indicator appears (top-left, blinking red)
- Timer starts counting (00:00, 00:01, etc.)
- **Live Joint Angles panel** appears (right side or below on mobile)
- Panel shows real-time angles:
  - Left Elbow: XXX°
  - Right Elbow: XXX°
  - Left Knee: XXX°
  - Right Knee: XXX°
  - Left Hip: XXX°
  - Right Hip: XXX°
- Angles are color-coded:
  - 🟢 Green = Normal range
  - 🟡 Yellow = Limited
  - 🔴 Red = Excessive

#### 9. Perform Movement
**What to do:**
- Perform a squat, arm raise, or any movement
- Watch the RED/YELLOW skeleton track your body
- See joint angles update in real-time
- Recording captures all skeleton data

#### 10. Stop Recording
**Action:** Tap "Stop Recording"
- Recording indicator disappears
- Timer stops
- "Analyze Movement" button appears
- Status shows: "Recorded XXX frames"

#### 11. Analyze Movement
**Action:** Tap "Analyze Movement"
- Processing notification appears
- AI analyzes skeleton data
- Results display shows:
  - Movement Quality Score (0-100)
  - Detected Deficiencies (with severity)
  - Recommended Exercises
  - Compensations detected

## 📐 Mobile Layout Verification

### Portrait Mode (Recommended)
- ✅ Camera selection modal: 1 column (stacked options)
- ✅ Camera controls: Vertical stack at bottom
- ✅ Joint info panel: Full width below video
- ✅ Progress steps: Wrapped layout
- ✅ Video: Full width with 16:9 ratio

### Landscape Mode
- ✅ Camera selection modal: 2 columns
- ✅ Camera controls: Horizontal row at bottom
- ✅ Joint info panel: Right side of video
- ✅ Progress steps: Single row
- ✅ Video: Full height with letterboxing

## 🎨 Visual Verification Checklist

### Camera Selection Modal
- [ ] Modal centered on screen
- [ ] 4 camera options clearly visible
- [ ] Icons large and recognizable
- [ ] Text readable without zooming
- [ ] Tap target size adequate (min 44x44px)
- [ ] "Start Assessment" button prominent

### Camera View
- [ ] Video fills screen width (portrait) or height (landscape)
- [ ] No black bars or stretching
- [ ] RED joint circles visible (12px major, 5px minor)
- [ ] YELLOW skeleton lines clearly visible
- [ ] Smooth skeleton tracking (30 fps)
- [ ] No lag or stuttering

### Recording Indicator
- [ ] Top-left corner, not blocking view
- [ ] Red dot blinking animation
- [ ] Timer updating every second
- [ ] Readable text size

### Joint Angles Panel
- [ ] Appears when recording starts
- [ ] Background semi-transparent black
- [ ] White text on dark background (readable)
- [ ] Angle values update smoothly
- [ ] Color coding visible (green/yellow/red)

### Camera Controls
- [ ] Buttons large enough to tap easily
- [ ] Icons clear and recognizable
- [ ] Blue primary button for main action
- [ ] Gray buttons for secondary actions
- [ ] Adequate spacing between buttons (min 8px)

## 🔧 Troubleshooting

### Issue: Camera permission denied
**Solution:**
- Go to browser settings → Site settings → Camera
- Allow camera access for this URL
- Reload the page

### Issue: No skeleton overlay appears
**Check:**
- Is MediaPipe loading? (Check console logs)
- Are you fully visible in frame?
- Is lighting adequate?
- Try moving to better lit area

### Issue: Flip camera not working
**Solution:**
- Some phones only have one camera
- Try reloading page and granting permission again
- Check if other apps can access both cameras

### Issue: Video is upside down or mirrored
**Note:**
- Front camera is mirrored by default (this is normal)
- Back camera shows correct orientation
- Skeleton tracking works correctly regardless

### Issue: Joint angles not updating
**Check:**
- Recording must be active (tap "Start Recording")
- Joint angles panel only shows during recording
- Make sure skeleton overlay is visible first

### Issue: Page loads but camera doesn't start
**Solutions:**
1. Reload page
2. Clear browser cache
3. Try different browser (Chrome recommended)
4. Check if other apps have camera locked

## 📊 Expected Performance

### Good Performance Indicators
- ✅ Page loads in < 3 seconds
- ✅ Camera starts within 2 seconds of permission grant
- ✅ Skeleton detection appears within 1 second
- ✅ Frame rate: 25-30 fps (smooth)
- ✅ Joint angle updates: Real-time (< 100ms delay)
- ✅ No camera freezing or stuttering

### Device Requirements
- **Minimum:**
  - Any smartphone from 2018+
  - iOS 12+ or Android 8+
  - Modern browser (Chrome 80+, Safari 13+)
  - Rear camera: 720p
  - Front camera: 480p

- **Recommended:**
  - Smartphone from 2020+
  - iOS 14+ or Android 10+
  - Chrome 90+ or Safari 14+
  - Rear camera: 1080p
  - Front camera: 720p

## 🎯 Test Scenarios

### Scenario 1: Basic Squat Assessment (Phone)
1. Select "Mobile Phone" camera
2. Grant camera permission
3. Flip to back camera
4. Position phone on stable surface facing you
5. Stand 6-8 feet away
6. Start recording
7. Perform 3 squats slowly
8. Stop recording
9. Analyze movement
10. Review quality score and deficiencies

### Scenario 2: Upper Body Assessment (Phone)
1. Select "Mobile Phone" camera
2. Use front camera (selfie mode)
3. Hold phone at arm's length or prop on table
4. Frame shows torso and arms
5. Start recording
6. Raise arms overhead 5 times
7. Perform arm circles
8. Stop recording
9. Analyze results
10. Check shoulder ROM and asymmetry

### Scenario 3: Balance Test (Phone)
1. Use back camera on stable surface
2. Full body in frame
3. Start recording
4. Stand on one leg for 30 seconds
5. Switch legs
6. Stop and analyze
7. Review balance deficiencies

## ✅ Final Verification

### Before Reporting Success
Confirm all items work:
- [ ] Page loads on mobile browser
- [ ] Camera selection modal displays correctly
- [ ] Can select camera type
- [ ] Camera permission request appears
- [ ] Camera starts and shows live feed
- [ ] Flip camera works (front ↔ back)
- [ ] RED/YELLOW skeleton overlay appears
- [ ] Skeleton tracks body movement smoothly
- [ ] Can start recording
- [ ] Recording indicator shows and timer counts
- [ ] Joint angles panel appears with real-time data
- [ ] Can stop recording
- [ ] Can analyze movement
- [ ] Results display with quality score

### Report Results
When testing, please note:
1. **Device:** (e.g., iPhone 13, Samsung Galaxy S21)
2. **OS:** (e.g., iOS 16, Android 12)
3. **Browser:** (e.g., Safari, Chrome Mobile)
4. **Working Features:** (List what works)
5. **Issues Found:** (List any problems)
6. **Screenshots:** (If possible, share screenshots)

---

**Status:** ✅ Mobile workflow is complete and ready for testing!

**Test Now:** Open the URL on your phone and follow the steps above.
