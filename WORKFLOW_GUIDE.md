# 🏥 PhysioMotion - Complete Workflow Guide

## 📋 Table of Contents
1. [Workflow Overview](#workflow-overview)
2. [Phase 1: Patient Intake](#phase-1-patient-intake)
3. [Phase 2: Movement Assessment](#phase-2-movement-assessment)
4. [Phase 3: Analysis & Results](#phase-3-analysis--results)
5. [Phase 4: Exercise Prescription](#phase-4-exercise-prescription)
6. [Phase 5: Medical Documentation](#phase-5-medical-documentation)
7. [Camera Setup Guide](#camera-setup-guide)
8. [Team Collaboration](#team-collaboration)

---

## 🎯 Workflow Overview

PhysioMotion follows a comprehensive medical assessment workflow designed for physical therapy and chiropractic clinics:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  COMPLETE CLINICAL WORKFLOW                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. PATIENT INTAKE                                                       │
│     └─→ Multi-step form with medical history                           │
│                                                                          │
│  2. MOVEMENT ASSESSMENT                                                  │
│     ├─→ Camera selection (Phone/Laptop/External/Femto Mega)           │
│     ├─→ Patient positioning and instructions                            │
│     ├─→ Movement recording with live joint tracking                     │
│     └─→ Real-time biomechanical feedback                               │
│                                                                          │
│  3. AI ANALYSIS                                                          │
│     ├─→ Joint angle calculations                                        │
│     ├─→ ROM measurements                                                │
│     ├─→ Deficiency detection                                            │
│     └─→ Movement quality scoring                                        │
│                                                                          │
│  4. EXERCISE PRESCRIPTION                                                │
│     ├─→ Automated exercise recommendations                              │
│     ├─→ Customized sets, reps, frequency                               │
│     └─→ Clinical reasoning documentation                                │
│                                                                          │
│  5. MEDICAL DOCUMENTATION                                                │
│     ├─→ SOAP note generation                                            │
│     ├─→ Treatment plan creation                                         │
│     ├─→ CPT code integration                                            │
│     └─→ Clinician review and approval                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Phase 1: Patient Intake

### **Purpose**
Collect comprehensive patient information before physical assessment.

### **URL:**
```
/static/intake
```

### **Steps**

#### **Step 1: Personal Information**
Collect basic demographics:
- ✅ First Name, Last Name
- ✅ Date of Birth
- ✅ Gender
- ✅ Email & Phone
- ✅ Address (Street, City, State, ZIP)
- ✅ Emergency Contact Information

**Required Fields:** Name, DOB, Email, Phone

**Validation:**
- Email format validation
- Phone number format validation
- Required fields cannot be empty

---

#### **Step 2: Medical History**
Gather health-related information:
- ✅ Height (cm) & Weight (kg)
- ✅ Insurance Provider
- ✅ Chief Complaint (free text)
- ✅ Current Pain Level (0-10 scale)
- ✅ Activity Level (Sedentary → Very Active)

**Pain Scale:**
- Interactive slider (0-10)
- Real-time display
- Helps prioritize treatment urgency

**Activity Level Options:**
- Sedentary
- Light Activity
- Moderate Activity
- Active
- Very Active

---

#### **Step 3: Assessment Reason**
Select primary reason for assessment:

| Reason | Description | Use Case |
|--------|-------------|----------|
| **Pre-Surgery** | Baseline assessment before surgery | Document pre-operative function |
| **Post-Surgery** | Recovery tracking after surgery | Monitor rehabilitation progress |
| **Injury Recovery** | Rehabilitation from injury | Track healing and function |
| **Athletic Performance** | Performance optimization | Enhance athletic movement |
| **General Wellness** | Preventive care | Maintain health and prevent injury |

**Visual Selection:**
- Color-coded cards
- Icon-based interface
- Clear descriptions

---

#### **Step 4: Review & Submit**
Final review before submission:
- ✅ Personal Information Summary
- ✅ Emergency Contact Summary
- ✅ Medical Information Summary
- ✅ Assessment Reason Badge
- ✅ Next Steps Preview

**Submit Button:** "Create Patient & Start Assessment"

---

### **After Submission**

Success screen shows:
- ✅ Patient name confirmation
- ✅ Patient ID assigned
- ✅ Two action buttons:
  - **Start Movement Assessment** → Proceed to camera assessment
  - **View All Patients** → Return to patient list

**Automatic Features:**
- Patient record created in D1 database
- Patient ID generated for tracking
- Ready for immediate assessment

---

## 🎥 Phase 2: Movement Assessment

### **Purpose**
Capture biomechanical data using camera-based motion tracking.

### **URL:**
```
/static/assessment
/static/assessment?patient_id=123  (with patient ID)
```

### **Workflow Steps**

---

### **Step 1: Camera Selection**

**Choose camera type based on environment:**

#### **Option A: Mobile Phone Camera**
**When to use:**
- Home assessments
- Remote patient monitoring
- Telehealth sessions

**Features:**
- ✅ Front camera (selfie mode)
- ✅ Back camera (rear-facing)
- ✅ Flip camera button
- ✅ MediaPipe Pose tracking
- ✅ Works on iOS & Android

**Instructions for Patient:**
```
1. Grant camera permission when prompted
2. Position phone 6-8 feet away
3. Ensure full body is visible in frame
4. Use back camera for best angles
5. Ensure good lighting (face the light)
```

---

#### **Option B: Laptop/Desktop Camera**
**When to use:**
- In-clinic assessments
- Workstation setups
- Larger screen viewing

**Features:**
- ✅ Built-in webcam support
- ✅ External USB camera support
- ✅ Camera switching (if multiple cameras)
- ✅ Higher resolution options
- ✅ Desktop controls

**Instructions for Clinician:**
```
1. Position laptop/camera at chest height
2. Patient stands 8-10 feet away
3. Ensure full body visible
4. Check lighting and background
5. Test camera before starting
```

---

#### **Option C: External USB Camera**
**When to use:**
- Professional clinic setups
- Multiple camera angles needed
- Higher quality capture required

**Features:**
- ✅ Automatic detection
- ✅ Multiple camera support
- ✅ Camera cycling/switching
- ✅ Professional quality
- ✅ Configurable settings

**Setup:**
```
1. Connect USB camera to computer
2. Refresh assessment page
3. Select "Laptop Camera" option
4. Use "Switch Camera" to choose external
5. Position for optimal viewing angle
```

---

#### **Option D: Femto Mega (Professional)**
**When to use:**
- Clinical environments
- Research settings
- High-precision assessments

**Features:**
- ⚠️ Requires bridge server setup
- ⚠️ Advanced configuration needed
- ✅ Azure Kinect Body Tracking SDK
- ✅ 32-point skeleton tracking
- ✅ Depth sensing

**Note:** Femto Mega requires additional setup not covered in this guide.

---

### **Step 2: Camera Permission Check**

**Pre-assessment check button:**
```
📹 Check Camera Access
```

**What it does:**
- Tests camera availability
- Checks browser permissions
- Detects multiple cameras
- Provides troubleshooting help

**Permission Troubleshooting:**

**iPhone:**
```
Settings → Safari → Camera → Allow
```

**Android:**
```
Settings → Apps → Chrome → Permissions → Camera → Allow
```

**Desktop Chrome:**
```
Click 🔒 icon → Camera → Allow → Refresh
```

**Desktop Firefox:**
```
Click 🔒 icon → Permissions → Camera → Allow
```

---

### **Step 3: Patient Positioning**

**Optimal Setup:**

**Distance from Camera:**
- Phone: 6-8 feet
- Laptop: 8-10 feet
- External: 10-12 feet

**Camera Angle:**
- At chest/shoulder height
- Straight-on view (not angled up/down)
- Patient centered in frame

**Full Body Visibility:**
```
┌─────────────────────────────┐
│                             │
│        👤                   │  ← Head visible
│       /│\                   │  ← Arms fully extended visible
│        │                    │  ← Torso visible
│       / \                   │  ← Legs fully extended visible
│      👟 👟                  │  ← Feet visible
│                             │
└─────────────────────────────┘
```

**Lighting Requirements:**
- Face light source (window/lamp)
- Avoid backlighting
- Even lighting (no harsh shadows)
- Indoor lighting preferred

**Background:**
- Plain wall (solid color preferred)
- No clutter
- No moving objects
- Contrasting color to clothing

---

### **Step 4: Movement Execution**

**Recording Process:**

1. **Start Recording**
   - Click "Start Recording" button
   - RED indicator appears
   - Timer starts (MM:SS)

2. **Perform Movement Tests**

   **Example Test Sequence:**

   **A. Overhead Reach**
   ```
   - Stand with feet shoulder-width apart
   - Raise both arms overhead
   - Hold for 3 seconds
   - Lower slowly
   - Repeat 3 times
   ```

   **B. Forward Bend**
   ```
   - Stand with feet together
   - Keep legs straight
   - Bend forward, reaching toward toes
   - Hold for 3 seconds
   - Return to standing
   - Repeat 3 times
   ```

   **C. Squat Assessment**
   ```
   - Stand with feet shoulder-width apart
   - Lower into squat position
   - Keep heels on ground
   - Hold for 3 seconds
   - Return to standing
   - Repeat 3 times
   ```

   **D. Single Leg Balance**
   ```
   - Stand on one leg
   - Maintain balance for 10 seconds
   - Switch legs
   - Repeat each side 2 times
   ```

3. **Live Feedback**

   **Visual Overlays:**
   - 🔴 RED circles on joints (12px)
   - 💛 YELLOW skeleton connections
   - ⚪ WHITE borders on joints
   - 🔵 Glow effects on major joints

   **Live Joint Angles Panel:**
   ```
   ┌────────────────────────────┐
   │ Live Joint Angles          │
   ├────────────────────────────┤
   │ Left Elbow:     139° ✅   │
   │ Right Elbow:    127° ⚠️   │
   │ Left Knee:      112° ⚠️   │
   │ Right Knee:     134° ✅   │
   │ Left Hip:       95° ✅    │
   │ Right Hip:      88° ⚠️    │
   └────────────────────────────┘
   ```

   **Color Coding:**
   - ✅ Green = Normal range
   - ⚠️ Orange = Limited range
   - 🔴 Red = Severely limited

4. **Stop Recording**
   - Click "Stop Recording"
   - Frame count displayed
   - "Analyze Movement" button appears

---

### **Step 5: Data Capture**

**What Gets Recorded:**
- ✅ Skeleton landmarks (33 points)
- ✅ X, Y, Z coordinates per joint
- ✅ Visibility confidence scores
- ✅ Timestamp for each frame
- ✅ Multiple frames over time (30fps)

**Frame Data Example:**
```json
{
  "timestamp": 1698765432000,
  "landmarks": {
    "left_shoulder": { "x": 0.45, "y": 0.35, "z": -0.2, "visibility": 0.98 },
    "left_elbow": { "x": 0.50, "y": 0.45, "z": -0.15, "visibility": 0.95 },
    "left_wrist": { "x": 0.55, "y": 0.55, "z": -0.1, "visibility": 0.92 },
    ...
  }
}
```

**Storage:**
- Middle frame selected as representative
- Stored in D1 database
- Linked to assessment and test IDs

---

## 📊 Phase 3: Analysis & Results

### **Automatic AI Analysis**

When "Analyze Movement" is clicked:

**Backend Processing:**
1. ✅ Skeleton data sent to analysis API
2. ✅ Joint angles calculated for all major joints
3. ✅ ROM measurements computed
4. ✅ Bilateral asymmetry detected
5. ✅ Compensation patterns identified
6. ✅ Movement quality score generated (0-100)
7. ✅ Deficiencies classified by severity

**Analysis Components:**

### **A. Joint Angle Calculations**
```javascript
// Example angles computed:
- Shoulder Flexion: 145° (Normal: 150-180°)
- Elbow Extension: 165° (Normal: 160-180°)
- Hip Flexion: 85° (Normal: 110-120°)
- Knee Extension: 175° (Normal: 170-180°)
- Ankle Dorsiflexion: 12° (Normal: 15-20°)
```

### **B. Deficiency Detection**

**Deficiency Format:**
```json
{
  "area": "Right Shoulder ROM",
  "severity": "moderate",
  "description": "Limited overhead reach with 35° deficit in flexion",
  "measurement": {
    "actual": 145,
    "expected": 180,
    "deficit": 35
  }
}
```

**Severity Levels:**
- 🟢 **Mild**: 0-15% deficit
- 🟠 **Moderate**: 15-35% deficit
- 🔴 **Severe**: >35% deficit

### **C. Movement Quality Score**

**Scoring Algorithm:**
```
Score = (
  Joint Angle Adequacy × 0.40 +
  ROM Completeness × 0.30 +
  Bilateral Symmetry × 0.20 +
  Compensation Absence × 0.10
) × 100
```

**Score Interpretation:**
- **80-100**: Good movement quality
- **60-79**: Fair movement quality
- **<60**: Poor movement quality

---

### **Results Display**

**Visual Results Screen:**

```
┌────────────────────────────────────────────────────────────┐
│  📊 Movement Analysis Results                               │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Movement Quality Score: [72/100]                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  Based on joint angles, ROM, and compensation patterns     │
│                                                             │
├────────────────────────────────────────────────────────────┤
│  ⚠️ Detected Deficiencies (3)                              │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  🟠 RIGHT SHOULDER ROM - MODERATE                          │
│     Limited overhead reach with 35° deficit in flexion     │
│                                                             │
│  🟠 LEFT HIP MOBILITY - MODERATE                           │
│     Restricted hip flexion during squat (85° vs 110°)      │
│                                                             │
│  🟢 RIGHT KNEE STABILITY - MILD                            │
│     Minor tracking issue during single leg stance          │
│                                                             │
├────────────────────────────────────────────────────────────┤
│  💪 Recommended Exercises (5)                               │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ Shoulder Wall Slides                                   │
│     Improve overhead shoulder mobility                      │
│                                                             │
│  ✅ Hip Flexor Stretches                                   │
│     Increase hip flexion range of motion                    │
│                                                             │
│  ✅ Single Leg Balance Drills                              │
│     Enhance knee stability and proprioception              │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## 💊 Phase 4: Exercise Prescription

### **Automated Prescription System**

**Click:** "Prescribe Exercises" button

**Prescription Process:**

1. **Exercise Matching**
   - AI matches deficiencies to exercise library
   - Considers severity and patient ability
   - Selects evidence-based interventions

2. **Parameter Customization**
   ```
   For each exercise:
   - Sets: 3
   - Repetitions: 10-15
   - Frequency: 3-5x per week
   - Duration: 4-6 weeks
   - Hold time: 5-10 seconds (if applicable)
   ```

3. **Clinical Documentation**
   - Reason for prescription
   - Target deficiency
   - Expected outcome
   - Progression criteria

**Prescription Record:**
```json
{
  "patient_id": 123,
  "assessment_id": 456,
  "exercises": [
    {
      "exercise_id": 10,
      "exercise_name": "Shoulder Wall Slides",
      "sets": 3,
      "reps": 15,
      "frequency_per_week": 5,
      "target_deficiency": "Right Shoulder ROM",
      "clinical_reason": "Improve overhead shoulder flexion ROM"
    }
  ]
}
```

---

## 📄 Phase 5: Medical Documentation

### **SOAP Note Generation**

**Click:** "Generate Medical Note" button

**Automatic Generation Includes:**

### **S - Subjective**
From intake form:
- Chief complaint
- Pain level (0-10)
- Activity level
- Patient history

**Example:**
```
Patient presents for post-surgery assessment following
right shoulder rotator cuff repair. Reports pain level
of 4/10 with overhead activities. Currently light
activity level with goal to return to recreational
tennis.
```

---

### **O - Objective**
From movement assessment:
- Tests completed
- Movement quality score
- Joint angle measurements
- ROM deficits
- Compensation patterns

**Example:**
```
FUNCTIONAL MOVEMENT ASSESSMENT:
- Overall Movement Quality Score: 72/100
- Tests Completed: 4
  - Overhead Reach Assessment
  - Forward Bend Test
  - Squat Assessment
  - Single Leg Balance Test

MEASUREMENTS:
- Right Shoulder Flexion: 145° (Deficit: 35°)
- Left Hip Flexion: 85° (Deficit: 25°)
- Bilateral Symmetry: 85%

DEFICIENCIES IDENTIFIED:
1. Right Shoulder ROM - Moderate (35° flexion deficit)
2. Left Hip Mobility - Moderate (restricted flexion)
3. Right Knee Stability - Mild (tracking issue)
```

---

### **A - Assessment**
AI-generated summary:
- Movement quality interpretation
- Functional limitations
- Impact on daily activities
- Prognosis

**Example:**
```
Patient demonstrates FAIR movement quality (72/100)
with 3 significant deficiencies identified.

PRIMARY FINDINGS:
1. Right Shoulder ROM - Moderate severity
2. Left Hip Mobility - Moderate severity
3. Right Knee Stability - Mild severity

FUNCTIONAL IMPACT:
Moderate functional limitations present. Patient would
benefit from targeted therapeutic exercise program
focusing on shoulder and hip mobility.
```

---

### **P - Plan**
Treatment recommendations:
- Exercise prescription summary
- Frequency and duration
- Remote monitoring plan
- Re-assessment schedule
- Patient education
- CPT codes

**Example:**
```
TREATMENT PLAN:

1. THERAPEUTIC EXERCISES:
   - Shoulder Wall Slides (3x15, 5x/week)
   - Hip Flexor Stretches (3x30sec, 5x/week)
   - Single Leg Balance (3x30sec, 3x/week)

2. FREQUENCY: 3-5x per week for 6 weeks

3. REMOTE MONITORING:
   - Daily exercise tracking via mobile app
   - Real-time form feedback
   - Weekly progress reports

4. RE-ASSESSMENT: Schedule in 4 weeks

5. PATIENT EDUCATION:
   - Proper exercise technique
   - Pain monitoring guidelines
   - Progression criteria

CPT CODES: 97163, 97110, 97112, 98975, 98977
```

---

## 📷 Camera Setup Guide

### **Optimal Camera Positioning**

```
                 CAMERA
                   📹
                   │
                   │
                   │ 6-10 feet
                   │
                   ▼
                  👤 PATIENT
                 /│\
                  │
                 / \
```

**Height:** Camera at chest/shoulder level
**Distance:** 6-10 feet depending on camera type
**Angle:** Straight-on, not angled up or down

---

### **Lighting Setup**

**Best Practice:**
```
     LIGHT
      💡
       │
       ▼
      👤 ← CAMERA
     PATIENT    📹
```

**Avoid:**
```
     CAMERA        LIGHT
        📹   ←   💡
         \       /
          \     /
           \   /
            👤
          PATIENT
     (Backlit - Bad!)
```

---

### **Camera Comparison**

| Feature | Mobile Phone | Laptop Webcam | External USB | Femto Mega |
|---------|-------------|---------------|--------------|------------|
| **Portability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Time** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Cost** | Free | Free | $50-200 | $500+ |
| **Best For** | Home/Remote | In-office | Professional | Research |

---

## 👨‍⚕️ Team Collaboration

### **Roles**

**Clinician:**
- Conducts assessments
- Reviews AI analysis
- Prescribes exercises
- Generates medical notes
- Approves documentation

**Physical Therapist:**
- Executes assessment protocol
- Guides patient movements
- Monitors form quality
- Documents observations

**Patient:**
- Provides medical history
- Performs movement tests
- Follows exercise program
- Reports progress

**Admin:**
- Schedules appointments
- Manages patient records
- Handles billing
- Tracks compliance

---

### **Clinician Review Interface**

**Access:** `/clinician-review/:assessment_id`

**Review Checklist:**
- [ ] Patient information accurate
- [ ] Assessment captured properly
- [ ] AI analysis reasonable
- [ ] Deficiencies correctly identified
- [ ] Exercise prescriptions appropriate
- [ ] Medical note complete
- [ ] CPT codes correct

**Clinician Actions:**
- ✏️ Edit AI recommendations
- 📝 Add clinical notes
- ✅ Approve documentation
- 📧 Send to patient
- 💾 Save as final

---

## 🔗 Complete Workflow URLs

### **Public Access:**
```
https://3000-isjigehibebqnf5jhl4y7-2e1b9533.sandbox.novita.ai
```

### **Key Pages:**

| Page | URL | Purpose |
|------|-----|---------|
| **Home** | `/` | Landing page |
| **Intake** | `/static/intake` | Patient registration |
| **Patients** | `/static/patients` | Patient list |
| **Assessment** | `/static/assessment?patient_id=123` | Movement capture |
| **Review** | (Coming soon) | Clinician review |

---

## ✅ Workflow Completion Checklist

### **For Each Patient:**

- [ ] **Intake Complete**
  - [ ] Personal info collected
  - [ ] Medical history documented
  - [ ] Assessment reason selected
  - [ ] Patient record created

- [ ] **Assessment Complete**
  - [ ] Camera permissions granted
  - [ ] Patient positioned correctly
  - [ ] Movement tests performed
  - [ ] Data captured successfully

- [ ] **Analysis Complete**
  - [ ] AI analysis run
  - [ ] Deficiencies identified
  - [ ] Quality score calculated
  - [ ] Recommendations generated

- [ ] **Prescription Complete**
  - [ ] Exercises prescribed
  - [ ] Parameters customized
  - [ ] Clinical reasoning documented
  - [ ] Patient education provided

- [ ] **Documentation Complete**
  - [ ] SOAP note generated
  - [ ] Treatment plan created
  - [ ] CPT codes assigned
  - [ ] Clinician approved

---

## 📞 Support & Troubleshooting

### **Common Issues**

**Camera not working:**
- Check browser permissions
- Verify HTTPS connection
- Try different browser
- Restart browser/device

**Poor tracking quality:**
- Improve lighting
- Check patient distance
- Ensure full body visible
- Remove background clutter

**Analysis failed:**
- Check data capture
- Verify sufficient frames
- Retry movement test
- Contact support

---

## 🎓 Training Resources

### **For Clinicians:**
1. Review this workflow guide
2. Practice with demo patient
3. Test all camera types
4. Familiarize with AI analysis
5. Review medical documentation

### **For Patients:**
1. Review movement instructions
2. Practice camera positioning
3. Understand exercise program
4. Learn progress tracking

---

**Last Updated:** October 2025
**Version:** 1.0
**Platform:** Cloudflare Pages + Hono + D1 Database
