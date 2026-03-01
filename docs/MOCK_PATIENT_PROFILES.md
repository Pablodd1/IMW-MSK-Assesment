# 👥 MOCK PATIENT PROFILES - Complete Test Cases

## 📋 Overview

This document contains 3 comprehensive mock patient profiles for testing the PhysioMotion assessment system:

1. **Case 1:** Young child (8 years old) - Developmental coordination disorder
2. **Case 2:** Young adult (28 years old) - Sports injury (ACL reconstruction)
3. **Case 3:** Elderly patient (72 years old) - Osteoarthritis and balance issues

Each profile includes complete patient information, medical history, assessment results, and expected system outputs.

---

## 🧒 CASE 1: PEDIATRIC PATIENT - Emma Rodriguez

### **PATIENT DEMOGRAPHICS**

```json
{
  "patient_id": 1001,
  "name": "Emma Rodriguez",
  "date_of_birth": "2017-03-15",
  "age": 8,
  "sex": "Female",
  "height": 127,
  "height_unit": "cm",
  "weight": 25.5,
  "weight_unit": "kg",
  "phone": "(555) 234-5678",
  "email": "parent.rodriguez@email.com",
  "address": "456 Maple Street, Apt 3B, Springfield, IL 62701",
  "emergency_contact_name": "Maria Rodriguez (Mother)",
  "emergency_contact_phone": "(555) 234-5679",
  "insurance_provider": "Blue Cross Blue Shield",
  "insurance_id": "BC123456789",
  "primary_care_physician": "Dr. Sarah Johnson",
  "referring_physician": "Dr. Michael Chen (Pediatrics)"
}
```

### **MEDICAL HISTORY**

**Chief Complaint:**
"Emma has trouble with coordination and balance. She falls frequently during play and has difficulty with activities like catching a ball or riding a bicycle."

**Current Diagnoses:**
- Developmental Coordination Disorder (DCD)
- Mild hypotonia (low muscle tone)
- Delayed gross motor development

**Past Medical History:**
- Born full-term, normal delivery
- Met early milestones slightly delayed (walked at 15 months)
- No major illnesses or hospitalizations
- Up to date on vaccinations

**Medications:**
- None currently

**Allergies:**
- No known drug allergies (NKDA)
- Mild seasonal allergies (pollen)

**Social History:**
- Lives with both parents and younger brother (age 5)
- Attends 3rd grade at local elementary school
- Enjoys art and reading, avoids sports activities
- Has occupational therapy at school 2x/week

**Family History:**
- Mother: History of ADHD
- Father: Healthy
- Maternal uncle: Dyspraxia

**Prior Therapy:**
- Occupational therapy: 6 months (ongoing)
- Physical therapy: 3 sessions (discontinued due to insurance)

**Functional Limitations:**
- Difficulty catching/throwing balls
- Cannot ride bicycle without training wheels
- Falls frequently during running
- Poor handwriting (receives accommodations at school)
- Difficulty with stairs (requires handrail)
- Cannot hop on one foot
- Clumsy during play activities

---

### **ASSESSMENT DATA - EMMA RODRIGUEZ**

**Date:** January 31, 2026  
**Time:** 10:00 AM  
**Clinician:** Dr. Jennifer Martinez, PT, DPT  
**Assessment Type:** Initial Evaluation  
**Duration:** 45 minutes

#### **Camera Used:**
- Laptop Camera (MediaPipe 33-point tracking)
- Resolution: 1280x720
- Frame rate: 30 FPS

#### **Tests Performed:**

**1. Squat Test (5 seconds)**
```json
{
  "test_name": "Bilateral Squat Assessment",
  "duration": 5,
  "camera_angle": "front",
  "skeleton_frames": 150,
  "quality_score": 45
}
```

**Movement Quality Score:** 45/100 (Poor)

**Joint Angles Measured:**
- Hip flexion: 75° (Limited - normal for age: 110-120°)
- Knee flexion: 85° (Limited - normal: 120-130°)
- Ankle dorsiflexion: 10° (Severely limited - normal: 20-25°)
- Trunk angle: 45° forward lean (Excessive)

**Observed Deficiencies:**
- Poor squat depth (quarter squat only)
- Excessive forward trunk lean for compensation
- Knees collapsing inward (valgus collapse)
- Poor balance (needed wall support)
- Unable to maintain heel contact with ground
- Limited ankle mobility restricting squat depth

**2. Single Leg Balance Test - Right (10 seconds)**
```json
{
  "test_name": "Single Leg Balance - Right",
  "duration": 10,
  "attempts": 3,
  "best_time": 3.2,
  "quality_score": 30
}
```

**Results:**
- Balance time: 3.2 seconds (Normal for age: 10+ seconds)
- Multiple balance corrections (6 instances)
- Arms extended for balance
- Hip hiking on stance leg
- Unable to maintain vertical alignment

**3. Single Leg Balance Test - Left (10 seconds)**
```json
{
  "test_name": "Single Leg Balance - Left",
  "duration": 10,
  "attempts": 3,
  "best_time": 2.8,
  "quality_score": 28
}
```

**Results:**
- Balance time: 2.8 seconds (Poor)
- More unstable than right leg
- Frequent foot touches to ground
- Excessive upper body sway

**4. Forward Reach Test**
```json
{
  "test_name": "Forward Reach",
  "duration": 3,
  "reach_distance": 15,
  "quality_score": 52
}
```

**Results:**
- Reach distance: 15 cm (Normal for age: 20-25 cm)
- Moderate balance impairment
- Fear of falling evident
- Required multiple attempts

**5. Heel-to-Toe Walk (Tandem Gait)**
```json
{
  "test_name": "Tandem Gait",
  "duration": 10,
  "distance": 3,
  "quality_score": 35
}
```

**Results:**
- Completed 3 meters in 10 seconds (slow)
- Multiple step-offs (7 errors)
- Arms extended for balance
- Eyes looking down at feet
- Difficulty maintaining straight line

---

### **BIOMECHANICAL ANALYSIS RESULTS**

**Overall Movement Quality:** 38/100 (Poor)

**Deficiencies Identified:**

1. **Severe Balance Deficit**
   - **Area:** Static and dynamic balance
   - **Severity:** Severe
   - **Description:** Single leg balance time 70% below age norms. Multiple balance corrections required. High fall risk.
   - **Clinical Significance:** Limits participation in age-appropriate activities

2. **Limited Lower Extremity Flexibility**
   - **Area:** Ankle and hip mobility
   - **Severity:** Moderate
   - **Description:** Ankle dorsiflexion 10° (50% reduction). Hip flexion limited to 75° during squat.
   - **Clinical Significance:** Compensatory movement patterns, increased fall risk

3. **Muscle Weakness - Lower Extremities**
   - **Area:** Hip and knee stabilizers
   - **Severity:** Moderate
   - **Description:** Knee valgus during squat. Poor hip control. Unable to maintain alignment.
   - **Clinical Significance:** Poor movement quality, increased injury risk

4. **Poor Postural Control**
   - **Area:** Core stability and trunk control
   - **Severity:** Moderate
   - **Description:** Excessive trunk lean during squat (45°). Poor vertical alignment during balance.
   - **Clinical Significance:** Compensatory patterns, difficulty with complex movements

5. **Reduced Proprioception**
   - **Area:** Body awareness and spatial orientation
   - **Severity:** Moderate
   - **Description:** Eyes-dependent balance. Poor joint position sense.
   - **Clinical Significance:** Difficulty with motor planning and coordination

---

### **RECOMMENDED EXERCISES - EMMA RODRIGUEZ**

**Exercise Program Goals:**
1. Improve balance and postural control
2. Increase lower extremity strength
3. Enhance ankle mobility
4. Develop body awareness and proprioception
5. Build confidence in movement

**Exercise Prescription (3x/week for 8 weeks):**

**1. Balance Training**
- **Exercise:** Single leg stance with support
- **Sets/Reps:** 3 sets x 10 seconds each leg
- **Progression:** Start with hand on wall → fingertip support → no support → eyes closed
- **Frequency:** Daily
- **Focus:** Maintain level hips, look straight ahead

**2. Ankle Mobility**
- **Exercise:** Ankle dorsiflexion stretch (wall lean)
- **Sets/Reps:** 3 sets x 30 seconds each side
- **Progression:** Increase distance from wall
- **Frequency:** Daily
- **Focus:** Keep heel on ground, knee moves toward wall

**3. Squat Training**
- **Exercise:** Box squat with support
- **Sets/Reps:** 3 sets x 8 reps
- **Progression:** Higher box → lower box → no box
- **Frequency:** 3x/week
- **Focus:** Sit back, chest up, knees out

**4. Hip Strengthening**
- **Exercise:** Side-lying hip abduction (clamshells)
- **Sets/Reps:** 2 sets x 10 reps each side
- **Progression:** Add resistance band
- **Frequency:** 3x/week
- **Focus:** Keep hips stacked, controlled movement

**5. Core Stability**
- **Exercise:** Quadruped arm/leg raises (bird dog)
- **Sets/Reps:** 2 sets x 6 each side
- **Progression:** Increase hold time (5s → 10s)
- **Frequency:** 3x/week
- **Focus:** Keep back flat, no rotation

**6. Proprioception Training**
- **Exercise:** Balance board activities (sit on wobble cushion)
- **Sets/Reps:** 2 sets x 1 minute
- **Progression:** Sitting → kneeling → standing
- **Frequency:** Daily
- **Focus:** Maintain balance, controlled movements

**7. Fun Activities**
- Hopscotch (balance + coordination)
- Simon Says (body awareness)
- Obstacle course (functional movement)
- Balloon volleyball (tracking + coordination)

---

### **EXPECTED OUTCOMES - 8 WEEKS**

**Baseline (Week 0):**
- Movement quality: 38/100
- Single leg balance: 3.2 seconds
- Squat depth: 75° hip flexion
- Ankle dorsiflexion: 10°

**Expected Progress (Week 8):**
- Movement quality: 55-65/100 (40-70% improvement)
- Single leg balance: 7-10 seconds (120-210% improvement)
- Squat depth: 95-105° hip flexion (25-40% improvement)
- Ankle dorsiflexion: 15-18° (50-80% improvement)

**Functional Goals:**
- Able to hop on one foot for 3 consecutive hops
- Ride bicycle with training wheels confidently
- Navigate stairs with decreased handrail reliance
- Participate in playground activities with peers
- Reduced fall frequency by 50%

**Long-term Goals (6-12 months):**
- Achieve age-appropriate balance (10+ seconds single leg)
- Learn to ride bicycle without training wheels
- Participate in organized sports/activities
- Improved self-confidence and peer interaction
- No accommodations needed for physical education class

---

### **CLINICAL NOTES**

**Prognosis:** Good with consistent therapy and home exercise program

**Barriers to Success:**
- Limited insurance coverage (10 visits/year)
- Requires parental supervision for home exercises
- May require school-based therapy support

**Recommendations:**
1. Continue PT 1x/week for 8 weeks
2. Home exercise program daily (parent-supervised)
3. School OT/PT consultation for carry-over
4. Re-evaluation in 8 weeks
5. Consider referral to adaptive sports program

**Billing Codes:**
- CPT 97110 (Therapeutic exercise) - 30 min
- CPT 97112 (Neuromuscular re-education) - 15 min
- ICD-10: F82 (Developmental coordination disorder)
- ICD-10: M62.81 (Muscle weakness)

---

## 👨 CASE 2: YOUNG ADULT - Marcus Thompson

### **PATIENT DEMOGRAPHICS**

```json
{
  "patient_id": 2001,
  "name": "Marcus Thompson",
  "date_of_birth": "1997-08-22",
  "age": 28,
  "sex": "Male",
  "height": 185,
  "height_unit": "cm",
  "weight": 88,
  "weight_unit": "kg",
  "phone": "(555) 789-4321",
  "email": "marcus.thompson@email.com",
  "address": "2847 Oak Avenue, Unit 12, Portland, OR 97204",
  "emergency_contact_name": "Jessica Thompson (Wife)",
  "emergency_contact_phone": "(555) 789-4322",
  "insurance_provider": "Aetna PPO",
  "insurance_id": "AT987654321",
  "primary_care_physician": "Dr. Robert Kim",
  "referring_physician": "Dr. Amanda Foster (Orthopedic Surgery)"
}
```

### **MEDICAL HISTORY**

**Chief Complaint:**
"I had ACL reconstruction surgery 6 weeks ago and need to get back to playing soccer. My knee feels unstable and I'm having trouble with the exercises my surgeon gave me."

**Current Diagnoses:**
- Status post ACL reconstruction (left knee) - 6 weeks
- Patellar tendonitis (chronic, left knee)
- Muscle atrophy (left quadriceps)

**Surgical History:**
- **Date:** December 20, 2025 (6 weeks ago)
- **Procedure:** Arthroscopic ACL reconstruction with hamstring autograft (left knee)
- **Surgeon:** Dr. Amanda Foster, MD
- **Complications:** None
- **Meniscus:** Intact (no meniscal tear)

**Mechanism of Injury:**
- Non-contact injury during soccer match (November 28, 2025)
- Planted left foot, twisted to change direction
- Felt "pop" and immediate instability
- Unable to continue play

**Medications:**
- Ibuprofen 600mg TID PRN pain (decreased from 800mg)
- Acetaminophen 500mg PRN
- No opioids (patient declined)

**Allergies:**
- Penicillin (rash)
- No other known allergies

**Social History:**
- Married, no children
- Works as software engineer (remote work)
- Competitive soccer player (semi-professional level)
- Gym member (strength training 4x/week prior to injury)
- Non-smoker, social drinker (occasional)
- High activity level, athletic lifestyle

**Past Medical History:**
- Generally healthy
- Previous right ankle sprain (2020) - fully recovered
- No chronic conditions
- No surgeries prior to ACL repair

**Athletic History:**
- Played soccer since age 8
- College varsity soccer (Division II)
- Currently plays in adult competitive league
- Training 5-6 days/week prior to injury
- Position: Midfielder

**Goals:**
- Return to soccer at pre-injury level
- Timeline: 9-12 months (per surgeon)
- Short-term: Walk without limp, climb stairs normally
- Medium-term: Running, jumping, cutting
- Long-term: Full return to competitive soccer

**Functional Limitations:**
- Walks with slight limp (antalgic gait)
- Difficulty descending stairs (requires handrail)
- Unable to jog or run
- Cannot squat past 90°
- Difficulty with single leg balance (left)
- Quad muscle visible atrophy (left vs. right)
- Knee swelling after activity
- Occasional sharp pain with certain movements

---

### **ASSESSMENT DATA - MARCUS THOMPSON**

**Date:** January 31, 2026  
**Time:** 2:00 PM  
**Clinician:** Dr. Kevin Rodriguez, PT, DPT, SCS  
**Assessment Type:** Post-operative evaluation (6 weeks post-op)  
**Duration:** 60 minutes

#### **Camera Used:**
- Femto Mega (OrbbecSDK 32-joint tracking with 3D depth)
- Resolution: 1920x1080 (color), 640x576 (depth)
- Frame rate: 30 FPS
- Depth accuracy: ±2mm

#### **Physical Examination:**

**Observation:**
- Antalgic gait pattern (reduced stance time on left)
- Quad atrophy visible (left < right, ~2 cm circumference difference)
- Minimal knee effusion
- Surgical scar well-healed, no signs of infection
- ROM limited in flexion

**Measurements:**
- **Knee circumference** (mid-patella):
  - Left: 38 cm
  - Right: 40 cm
  - Difference: -2 cm (quad atrophy)

- **Thigh circumference** (15 cm above patella):
  - Left: 52 cm
  - Right: 55 cm
  - Difference: -3 cm (significant atrophy)

- **Range of Motion (ROM):**
  - Left knee flexion: 0-115° (Limited)
  - Right knee flexion: 0-140° (Normal)
  - Left knee extension: -5° to 0° (Slight extension lag)
  - Right knee extension: 0° (Normal)

- **Strength Testing (Manual Muscle Test):**
  - Quad strength (left): 3+/5 (Fair+ with gravity)
  - Quad strength (right): 5/5 (Normal)
  - Hamstring (left): 4/5 (Good, protected post-op)
  - Hamstring (right): 5/5 (Normal)

---

#### **Tests Performed:**

**1. Bilateral Squat Assessment (10 seconds)**
```json
{
  "test_name": "Bilateral Squat - Post-ACL Repair",
  "duration": 10,
  "camera_angle": "front",
  "depth_data": true,
  "skeleton_frames": 300,
  "quality_score": 58
}
```

**Movement Quality Score:** 58/100 (Fair)

**Joint Angles Measured (3D Depth Data):**

**Left Knee (Surgical):**
- Hip flexion: 85° (Limited by pain/apprehension)
- Knee flexion: 95° (Limited - normal: 120-130°)
- Ankle dorsiflexion: 22° (Normal)
- Depth at max squat: 1.85m from camera
- Knee anterior translation: +15mm (excessive, indicates quad weakness)

**Right Knee (Non-surgical):**
- Hip flexion: 110° (Normal)
- Knee flexion: 125° (Normal)
- Ankle dorsiflexion: 24° (Normal)
- Depth: 1.78m (deeper squat on right side)

**Observed Deficiencies:**
- Asymmetrical squat depth (left side shallow)
- Weight shift toward right leg (70/30 distribution)
- Left knee valgus collapse at bottom (3° valgus angle)
- Quad lag on left side (delayed activation)
- Apprehension evident (hesitant descent)
- Limited loading tolerance on left

**2. Single Leg Squat - Right (Non-surgical)**
```json
{
  "test_name": "Single Leg Squat - Right",
  "duration": 5,
  "reps": 5,
  "quality_score": 82
}
```

**Results:**
- Completed 5 reps successfully
- Good depth (hip to 90° flexion)
- Minimal knee valgus (<5°)
- Controlled descent and ascent
- No pain reported

**3. Single Leg Squat - Left (Surgical) - MODIFIED**
```json
{
  "test_name": "Single Leg Squat - Left (Modified)",
  "duration": 5,
  "reps": 3,
  "quality_score": 42,
  "modification": "Fingertip support on wall"
}
```

**Results:**
- Required wall support for balance
- Shallow squat only (30° knee flexion)
- Significant knee valgus (8° medial collapse)
- Poor control during descent
- Reported 4/10 pain at bottom
- Early fatigue (tremor after 3 reps)

**4. Single Leg Balance - Right**
```json
{
  "test_name": "Single Leg Balance - Right",
  "duration": 30,
  "eyes_open": 30,
  "eyes_closed": 18,
  "quality_score": 88
}
```

**Results:**
- Eyes open: 30 seconds (excellent)
- Eyes closed: 18 seconds (good)
- Minimal sway
- Confident balance

**5. Single Leg Balance - Left (Surgical)**
```json
{
  "test_name": "Single Leg Balance - Left",
  "duration": 30,
  "eyes_open": 12,
  "eyes_closed": 3,
  "quality_score": 45
}
```

**Results:**
- Eyes open: 12 seconds (poor)
- Eyes closed: 3 seconds (very poor)
- Significant sway (hip strategy)
- Multiple foot touches
- Decreased proprioception post-op

**6. Hop Test - DEFERRED**
```json
{
  "test_name": "Single Leg Hop for Distance",
  "status": "Deferred - too early post-op",
  "note": "Will assess at 12 weeks post-op per protocol"
}
```

**7. Gait Analysis (Femto Mega 3D Tracking)**
```json
{
  "test_name": "Gait Analysis - Level Surface",
  "distance": 10,
  "duration": 8,
  "steps": 14,
  "quality_score": 62
}
```

**Gait Parameters (3D Depth Measurement):**
- **Step length:**
  - Left: 58 cm (shortened)
  - Right: 72 cm (normal)
  - Asymmetry: 24% (significant)

- **Stance time:**
  - Left: 0.52 seconds (reduced)
  - Right: 0.78 seconds (increased)
  - Asymmetry: 50% (compensatory)

- **Knee flexion during gait:**
  - Left: 45° (reduced - normal: 60-70°)
  - Right: 65° (normal)

- **Vertical displacement:**
  - Left side lower (reduced push-off power)
  - Limp pattern evident in 3D analysis

**Observed Deficiencies:**
- Antalgic gait pattern (protecting left leg)
- Reduced stance time on surgical side
- Decreased knee flexion during swing phase
- Shortened stride length on left
- Reduced push-off power (weak gastrocnemius/soleus)

---

### **BIOMECHANICAL ANALYSIS RESULTS**

**Overall Movement Quality:** 54/100 (Fair - expected at 6 weeks post-op)

**Comparison to Non-surgical Side:**
- Quad strength: 40% deficit (left vs. right)
- Balance: 60% deficit
- Functional movement: 35% deficit

**Deficiencies Identified:**

1. **Severe Quadriceps Weakness (Post-surgical)**
   - **Area:** Left thigh (quadriceps muscle group)
   - **Severity:** Severe
   - **Description:** 3+ cm atrophy, MMT 3+/5, unable to perform single leg squat without support. Visible lag during knee extension.
   - **Clinical Significance:** Primary limiting factor for return to sport. Critical for knee stability and function.
   - **3D Depth Data:** Anterior knee translation +15mm during squat (excessive tibial sag)

2. **Reduced Proprioception (Left Knee)**
   - **Area:** Joint position sense and neuromuscular control
   - **Severity:** Moderate-Severe
   - **Description:** Single leg balance 60% worse than right. Eyes-closed balance severely impaired (3 vs. 18 seconds).
   - **Clinical Significance:** Increased re-injury risk. Essential for cutting/pivoting movements in soccer.

3. **Movement Asymmetry**
   - **Area:** Bilateral movement patterns (squat, gait)
   - **Severity:** Moderate
   - **Description:** Weight shift toward right leg (70/30). Step length 24% shorter on left. Reduced knee flexion 25%.
   - **Clinical Significance:** Compensatory patterns may lead to overuse injuries on right side.

4. **Dynamic Knee Stability**
   - **Area:** Knee valgus control during functional movement
   - **Severity:** Moderate
   - **Description:** 8° medial knee collapse during single leg squat (left). Poor hip abductor/external rotator control.
   - **Clinical Significance:** Risk factor for ACL graft failure. Must address before return to sport.

5. **Range of Motion Limitation**
   - **Area:** Left knee flexion
   - **Severity:** Mild-Moderate
   - **Description:** Flexion 115° (normal: 135-140°). 20° deficit. No significant extension lag.
   - **Clinical Significance:** May limit squatting, running mechanics. Typically improves with progressive loading.

---

### **RECOMMENDED EXERCISES - MARCUS THOMPSON**

**Exercise Program Goals:**
1. Restore quadriceps strength (primary goal)
2. Improve proprioception and neuromuscular control
3. Normalize movement patterns (symmetry)
4. Progress ROM to full flexion
5. Gradual return to sport-specific activities

**Phase 1 (Weeks 6-12 post-op): Strength & Control**

**1. Quadriceps Strengthening**
- **Exercise:** Straight leg raises (SLR) with 10-second hold
- **Sets/Reps:** 3 sets x 15 reps
- **Progression:** Add ankle weight (5 lb → 10 lb → 15 lb)
- **Frequency:** Daily
- **Focus:** Full quad contraction, no knee flexion, maintain leg straight

- **Exercise:** Terminal knee extension (TKE) with resistance band
- **Sets/Reps:** 3 sets x 20 reps
- **Progression:** Increase band resistance (light → medium → heavy)
- **Frequency:** Daily
- **Focus:** Lock out knee fully, squeeze quad at end range

- **Exercise:** Wall sits with Swiss ball
- **Sets/Reps:** 3 sets x 30-60 seconds
- **Progression:** Increase hold time, add single leg slides
- **Frequency:** Daily
- **Focus:** Equal weight distribution, knee at 90°

**2. Proprioception & Balance Training**
- **Exercise:** Single leg stance on balance board (left)
- **Sets/Reps:** 3 sets x 30 seconds
- **Progression:** Stable surface → foam pad → wobble board → eyes closed
- **Frequency:** Daily
- **Focus:** Minimal sway, hip level, knee slightly bent

- **Exercise:** Single leg mini squats with perturbations
- **Sets/Reps:** 3 sets x 10 reps
- **Progression:** Add external perturbations (ball tosses)
- **Frequency:** 5x/week
- **Focus:** Control knee alignment, no valgus collapse

**3. Hip Strengthening (Prevent Valgus)**
- **Exercise:** Side-lying hip abduction (clamshells)
- **Sets/Reps:** 3 sets x 15 reps each side
- **Progression:** Add resistance band (light → medium → heavy)
- **Frequency:** Daily
- **Focus:** Prevent knee valgus, control from hip

- **Exercise:** Monster walks (lateral band walks)
- **Sets/Reps:** 3 sets x 20 steps each direction
- **Progression:** Increase band resistance
- **Frequency:** 5x/week
- **Focus:** Keep tension in band, knees aligned over toes

**4. ROM & Flexibility**
- **Exercise:** Heel slides for knee flexion
- **Sets/Reps:** 3 sets x 15 reps
- **Progression:** Increase flexion angle gradually
- **Frequency:** 3x/day
- **Focus:** Gentle stretch, no pain past 4/10

- **Exercise:** Prone hangs for knee flexion (gravity-assisted)
- **Sets/Reps:** 3 sets x 2 minutes
- **Frequency:** 2x/day
- **Focus:** Relax, let gravity assist

**5. Gait Training**
- **Exercise:** Backwards walking on treadmill
- **Sets/Reps:** 5 minutes at 1.5 mph
- **Progression:** Increase speed and duration
- **Frequency:** Daily
- **Focus:** Equal step length, symmetrical loading

- **Exercise:** Forward walking with focus on symmetry
- **Sets/Reps:** 10 minutes
- **Progression:** Increase speed, add incline
- **Frequency:** Daily
- **Focus:** Equal stance time, full knee extension

**6. Stationary Bike**
- **Exercise:** Cycling (low resistance)
- **Duration:** 15-20 minutes
- **Progression:** Increase duration and resistance
- **Frequency:** Daily
- **Focus:** Smooth pedal stroke, no pain, ROM work

**7. Pool Therapy (Highly Recommended)**
- Water walking/jogging (buoyancy reduces load)
- Pool squats and lunges
- High knee marching
- Frequency: 2-3x/week if available

---

**Phase 2 (Weeks 12-16 post-op): Power & Agility** (Future)

Will include:
- Box step-ups with weight
- Single leg Romanian deadlifts (RDL)
- Forward/lateral lunges with resistance
- Hop progression (double leg → single leg)
- Agility ladder drills
- Sport-specific movements

**Phase 3 (Weeks 16-24 post-op): Return to Sport** (Future)

Will include:
- Plyometric training
- Cutting drills
- Sprint training
- Soccer-specific movements
- Scrimmage progression

---

### **EXPECTED OUTCOMES**

**6-Week Baseline (Current):**
- Quad strength (MMT): 3+/5
- Single leg balance: 12 seconds
- Squat depth: 95° knee flexion
- Gait: Antalgic, 24% asymmetry
- Movement quality: 54/100

**12-Week Target:**
- Quad strength: 4+/5 (80% of contralateral)
- Single leg balance: 25 seconds
- Squat depth: 120° knee flexion (bilateral)
- Gait: Minimal limp, <10% asymmetry
- Movement quality: 70-75/100
- Cleared for light jogging

**16-Week Target:**
- Quad strength: 5-/5 (90% of contralateral)
- Full ROM (0-140° flexion)
- Single leg squat without support
- Begin hop testing (pass 80% limb symmetry index)
- Cleared for agility drills

**24-Week (6-month) Target:**
- Quad strength: 5/5 (≥95% of contralateral)
- Pass all hop tests (>90% LSI)
- Full agility and sport-specific movements
- Begin return-to-sport progressions (practice drills)

**36-Week (9-month) Target:**
- Cleared for full contact practice
- Functional movement screen: Pass all tests
- Movement quality: 85-90/100
- Psychological readiness: Confident in knee

**48-Week (12-month) Target:**
- Full return to competitive soccer
- Maintenance strength program
- Injury prevention strategies
- Movement quality: 90-95/100

---

### **CLINICAL NOTES**

**Prognosis:** Excellent with appropriate rehabilitation progression

**Positive Factors:**
- Young, healthy, highly motivated athlete
- Excellent surgical result (per surgeon report)
- High baseline fitness level
- Good compliance with early exercises
- Strong social support (wife very supportive)
- Understands timeline and importance of not rushing

**Risk Factors:**
- High-demand sport (soccer requires cutting/pivoting)
- Desire to return quickly (must manage expectations)
- Prior history of knee pain (patellar tendonitis)
- Will need to address biomechanics to prevent re-injury

**Recommendations:**
1. PT 2x/week for 8-12 weeks
2. Home exercise program daily (1-1.5 hours)
3. Continue with surgeon for follow-up (12 weeks post-op)
4. Re-evaluation at 12 weeks for progression to Phase 2
5. Functional testing at 6, 9, and 12 months before return to sport
6. Recommend ACL injury prevention program upon return

**Return to Sport Criteria (Must meet ALL before clearance):**
- Quad strength ≥90% of contralateral
- Hop test limb symmetry index ≥90%
- Full ROM (equal to contralateral)
- Pass functional movement screen
- Psychological readiness (ACL-RSI score >56)
- Physician clearance

**Billing Codes:**
- CPT 97110 (Therapeutic exercise) - 30 min
- CPT 97112 (Neuromuscular re-education) - 15 min
- CPT 97530 (Therapeutic activities) - 15 min
- ICD-10: M23.622 (Spontaneous disruption of anterior cruciate ligament of left knee)
- ICD-10: Z98.89 (Post-surgical state, ACL reconstruction)
- ICD-10: M62.562 (Muscle wasting and atrophy, left lower leg)

---

## 👵 CASE 3: ELDERLY PATIENT - Margaret "Maggie" Chen

### **PATIENT DEMOGRAPHICS**

```json
{
  "patient_id": 3001,
  "name": "Margaret Chen",
  "preferred_name": "Maggie",
  "date_of_birth": "1953-11-08",
  "age": 72,
  "sex": "Female",
  "height": 158,
  "height_unit": "cm",
  "weight": 68,
  "weight_unit": "kg",
  "phone": "(555) 456-7890",
  "email": "m.chen1953@email.com",
  "address": "1234 Elm Drive, Senior Living Community, Building C, #305, Seattle, WA 98101",
  "emergency_contact_name": "David Chen (Son)",
  "emergency_contact_phone": "(555) 456-7891",
  "emergency_contact_relationship": "Son",
  "secondary_emergency_contact": "Susan Lee (Daughter)",
  "secondary_phone": "(555) 456-7892",
  "insurance_provider": "Medicare Part B + AARP Supplement",
  "insurance_id": "MEDICARE123456789A",
  "primary_care_physician": "Dr. Patricia Wong",
  "referring_physician": "Dr. James Park (Geriatric Medicine)"
}
```

### **MEDICAL HISTORY**

**Chief Complaint:**
"I'm having more trouble getting around. My knees hurt, especially going up and down stairs. I've fallen twice in the past month and I'm afraid I'll fall again. I want to keep living independently but I'm worried."

**Current Diagnoses:**
- Osteoarthritis (bilateral knees, worse on right) - severe
- Osteoarthritis (lumbar spine) - moderate
- History of falls (2 falls in past month)
- Sarcopenia (age-related muscle loss)
- Hypertension (controlled)
- Type 2 Diabetes Mellitus (well-controlled)
- Osteoporosis
- Mild cognitive impairment (early stage)

**Fall History (CRITICAL):**

**Fall #1 - December 28, 2025:**
- Location: Bathroom at home (early morning)
- Cause: Lost balance while turning after using toilet
- Injury: Bruised left hip and elbow
- Medical attention: None sought initially, reported to PCP at next visit
- No loss of consciousness

**Fall #2 - January 15, 2026:**
- Location: Kitchen (evening)
- Cause: Tripped over rug edge while carrying dishes
- Injury: Bruised right knee, mild abrasion on palm
- Medical attention: Evaluated by facility nurse, no fractures
- Increased fear of falling since this incident

**Past Medical History:**
- Hypertension (diagnosed 1998) - controlled with medication
- Type 2 Diabetes (diagnosed 2005) - HbA1c 6.8% (good control)
- Osteoporosis (diagnosed 2015) - on bisphosphonate therapy
- Osteoarthritis (diagnosed 2008) - progressive worsening
- Bilateral cataract surgery (2018) - successful
- Total hysterectomy (1995) - benign indication
- Compression fracture L1 vertebra (2020) - treated conservatively
- Peripheral neuropathy (mild, feet) - secondary to diabetes
- Hyperlipidemia - controlled with statin

**Medications (11 total):**
1. Lisinopril 20mg daily (hypertension)
2. Metformin 1000mg BID (diabetes)
3. Atorvastatin 40mg daily (cholesterol)
4. Alendronate 70mg weekly (osteoporosis)
5. Calcium + Vitamin D3 daily
6. Aspirin 81mg daily (cardiovascular protection)
7. Meloxicam 15mg daily PRN (knee pain - uses 4-5x/week)
8. Acetaminophen 650mg TID PRN (pain)
9. Gabapentin 300mg nightly (neuropathy)
10. Multivitamin daily
11. Vitamin B12 1000mcg daily (neuropathy)

**Allergies:**
- Sulfa drugs (rash)
- No other known allergies

**Social History:**
- Widowed 5 years (husband died 2020)
- Lives alone in senior living community (independent apartment)
- Two adult children: son (Seattle), daughter (Portland)
- Former elementary school teacher (retired 2018)
- Drives locally (stopped highway driving 2 years ago)
- Active in senior center activities (book club, art class)
- Uses cane for outdoor walking (inconsistent use indoors)
- Has medical alert pendant (activated during Fall #2)

**Functional Status (ADLs/IADLs):**

**Activities of Daily Living (ADLs):**
- Bathing: Independent (uses shower chair)
- Dressing: Independent (some difficulty with socks/shoes)
- Toileting: Independent (uses grab bars)
- Transfers: Independent (slow, uses furniture for support)
- Feeding: Independent
- Grooming: Independent

**Instrumental ADLs (IADLs):**
- Meal preparation: Independent (simple meals)
- Housekeeping: Requires assistance (cleaning service 1x/week)
- Laundry: Independent (uses facility laundry room on same floor)
- Shopping: Modified independent (son drives her 1x/week)
- Medication management: Independent (uses pill organizer)
- Phone use: Independent
- Finances: Independent (online banking with son's oversight)

**Living Environment:**
- Senior living community apartment (one bedroom)
- Elevator building (lives on 3rd floor)
- Community dining room available (uses 2-3x/week)
- On-site activities and social programs
- Emergency call system in apartment
- Bathroom has grab bars and raised toilet seat
- No stairs within apartment (single level)
- Concerned about ability to maintain independence

**Mobility:**
- Ambulates with cane outdoors (inconsistent use indoors)
- Does NOT use walker (refuses - "makes me feel old")
- Drives car (local errands only, daylight only)
- No longer comfortable with public transportation
- Difficulty with stairs (requires handrail, one step at a time)
- Walks slowly, careful with turns
- Avoids uneven surfaces (fear of falling)

**Prior Therapy:**
- Physical therapy after L1 compression fracture (2020) - 8 visits
- No PT since then
- Receives cortisone injections in knees (last: 4 months ago)

**Family History:**
- Mother: Osteoporosis, hip fracture at age 78, died at 85
- Father: Stroke, died at 72
- Sister: Type 2 diabetes, hypertension
- No family history of dementia/Alzheimer's

**Goals (Patient-Stated):**
1. "I don't want to fall anymore. It scares me."
2. "I want to keep living in my apartment, not a nursing home."
3. "I'd like to walk to the community center without my son worrying."
4. "I want to play with my grandchildren without pain."
5. "I want to feel confident and safe when I move around."

**Functional Limitations:**
- Difficulty rising from low chairs (uses armrests)
- Cannot kneel or squat (knee pain)
- Difficulty with stairs (requires handrail, slow)
- Reduced walking speed (uses furniture/walls for support)
- Difficulty with balance during turns or reaching
- Avoids crowded areas due to fear of being bumped
- Limits outdoor walking due to uneven surfaces
- Pain with prolonged standing (>10 minutes)
- Difficulty with car transfers (low seat height)
- Cannot carry items while walking (needs cane)

---

### **ASSESSMENT DATA - MARGARET CHEN**

**Date:** January 31, 2026  
**Time:** 9:00 AM  
**Clinician:** Dr. Lisa Chang, PT, DPT, GCS (Geriatric Clinical Specialist)  
**Assessment Type:** Fall risk evaluation & functional assessment  
**Duration:** 90 minutes (includes rest breaks)

#### **Camera Used:**
- Laptop Camera (MediaPipe 33-point tracking)
- Resolution: 1280x720
- Frame rate: 30 FPS
- Patient wore contrast clothing for better tracking

#### **Physical Examination:**

**Vital Signs:**
- BP: 142/85 mmHg (slightly elevated)
- HR: 78 bpm (regular)
- RR: 16/min
- SpO2: 97% on room air
- Pain (at rest): 3/10 (bilateral knees)
- Pain (with movement): 6-7/10 (bilateral knees)

**Observation:**
- Slight forward head posture
- Rounded shoulders (kyphotic posture)
- Knee flexion contractures (mild, bilateral)
- Uses furniture for support when walking indoors
- Antalgic gait (short, shuffling steps)
- Decreased arm swing
- Fear of movement evident (hesitant with new tasks)

**Measurements:**

**Range of Motion (ROM):**
- **Cervical spine:** Rotation limited (60° each side, normal 80°)
- **Lumbar spine:** Flexion limited (fingertips to mid-shin, normal to toes)
- **Right knee:** 
  - Flexion: 0-110° (limited, pain at end range)
  - Extension: -10° (flexion contracture, cannot fully straighten)
- **Left knee:**
  - Flexion: 0-115° (limited)
  - Extension: -8° (flexion contracture)
- **Ankle dorsiflexion:**
  - Right: 5° (severely limited)
  - Left: 8° (limited)

**Strength Testing (Manual Muscle Test):**
- Hip flexors (bilateral): 4-/5 (reduced for age)
- Hip extensors (bilateral): 3+/5 (weak)
- Hip abductors (bilateral): 3/5 (weak - critical for balance)
- Knee extensors (quads):
  - Right: 3/5 (significant weakness)
  - Left: 3+/5 (weakness)
- Knee flexors (hamstrings): 4/5 (bilateral)
- Ankle plantar flexors: 3+/5 (bilateral, weak)
- Ankle dorsiflexors: 3/5 (bilateral, weak - foot drop risk)

**Sensation Testing:**
- **Light touch:** Intact in upper extremities
- **Light touch (feet):** Diminished bilaterally (peripheral neuropathy)
- **Proprioception (great toe):** Impaired bilaterally (diabetic neuropathy)
- **Vibration (tuning fork):** Absent at ankles, reduced at knees

**Special Tests:**

**Berg Balance Scale:** 38/56 (HIGH FALL RISK)
- Scores < 45 indicate high fall risk
- Significant deficits in:
  - Standing unsupported with eyes closed (1/4)
  - Standing on one leg (0/4 - unable)
  - Turning 360° (2/4 - unsteady)
  - Reaching forward (2/4 - limited)
  - Picking up object from floor (1/4 - requires support)

**Timed Up and Go (TUG):** 18.5 seconds (HIGH FALL RISK)
- <10 seconds: Normal
- 10-20 seconds: Increased fall risk
- >20 seconds: High fall risk, mobility impairment
- Patient required cane and demonstrated unsteady turning

**30-Second Chair Stand Test:** 6 repetitions (POOR)
- Age-norm (70-79 years, female): 12-14 reps
- Score <10 indicates lower extremity weakness
- Patient used armrests for assistance (should be no arms)

**Gait Speed (4-Meter Walk Test):** 0.62 m/s (IMPAIRED)
- <0.8 m/s: Mobility impairment, increased fall risk
- <0.6 m/s: Severe mobility limitation
- Patient walked with cane, short shuffling steps

**Functional Reach Test:** 18 cm (IMPAIRED)
- Age-norm (70-87 years, female): 25-30 cm
- <15 cm: High fall risk
- Patient demonstrated fear of losing balance

---

#### **Tests Performed (Video-Based):**

**1. Sit-to-Stand Assessment (5 repetitions)**
```json
{
  "test_name": "Five Times Sit-to-Stand Test",
  "duration": 28,
  "reps": 5,
  "use_of_arms": true,
  "quality_score": 35
}
```

**Movement Quality Score:** 35/100 (Poor)

**Joint Angles Measured:**
- Hip flexion (standing): 95° (reduced from chair)
- Knee extension (standing): -9° average (bilateral flexion contractures)
- Ankle dorsiflexion: 6° (limited)
- Trunk forward lean: 55° (excessive compensation)

**Observed Deficiencies:**
- Time: 28 seconds (Normal <12 seconds, >15 seconds = high fall risk)
- Required use of armrests for all repetitions
- Significant forward trunk lean (poor quad strength)
- Multiple attempts to rise (rocking motion)
- Used momentum to stand (poor muscle control)
- Visible tremor in legs (fatigue after 3 reps)
- Complained of knee pain 7/10 during test

**2. Modified Squat Assessment (Limited depth)**
```json
{
  "test_name": "Modified Squat - Partial Depth",
  "duration": 8,
  "reps": 3,
  "support": "parallel bars for safety",
  "quality_score": 28,
  "notes": "Test stopped due to pain and safety concerns"
}
```

**Movement Quality Score:** 28/100 (Poor)

**Results:**
- Squat depth: Maximum 30° knee flexion (very shallow)
- Pain reported: 8/10 at end range
- Severe knee crepitus (grinding) audible
- Required hand support for balance
- Test stopped after 3 reps (patient requested)

**Observed Deficiencies:**
- Severe pain limiting depth
- Bilateral knee valgus (knees caving inward)
- Excessive forward lean (>60°)
- Loss of balance without hand support
- Severe ROM and strength limitations

**3. Static Balance Assessment - Eyes Open**
```json
{
  "test_name": "Static Balance - Feet Together, Eyes Open",
  "duration": 30,
  "successful_time": 11,
  "quality_score": 42
}
```

**Results:**
- Maintained position: 11 seconds (target: 30 seconds)
- Multiple balance corrections (hip and ankle strategy)
- Required standby assistance for safety
- Demonstrates poor static balance control

**4. Static Balance - Semi-Tandem Stance**
```json
{
  "test_name": "Semi-Tandem Stance (one foot partially ahead)",
  "duration": 10,
  "successful_time": 4,
  "quality_score": 25,
  "note": "Test stopped due to loss of balance"
}
```

**Results:**
- Maintained position: 4 seconds only
- Immediate loss of balance
- Required therapist assistance to prevent fall
- Unable to complete full test duration

**5. Static Balance - Tandem Stance**
```json
{
  "test_name": "Tandem Stance (heel-to-toe)",
  "status": "Unable to attempt",
  "note": "Patient unable to achieve position safely"
}
```

**6. Single Leg Stance - UNABLE**
```json
{
  "test_name": "Single Leg Stance - Right",
  "status": "Unable to attempt",
  "note": "Immediate loss of balance, requires bilateral support"
}
```

```json
{
  "test_name": "Single Leg Stance - Left",
  "status": "Unable to attempt",
  "note": "Immediate loss of balance, requires bilateral support"
}
```

**7. Gait Analysis with Cane**
```json
{
  "test_name": "Gait Analysis - Level Surface with Cane",
  "distance": 10,
  "duration": 16,
  "steps": 24,
  "quality_score": 38
}
```

**Gait Parameters:**
- **Gait speed:** 0.625 m/s (severely impaired)
- **Step length:** 
  - Right: 35 cm (shortened)
  - Left: 38 cm (shortened)
  - Normal for age: 55-65 cm
- **Step width:** 18 cm (widened base for stability)
- **Cadence:** 90 steps/min (slow, cautious)
- **Double support time:** 45% of gait cycle (increased, normal 20-25%)

**Observed Deficiencies:**
- Shuffling gait pattern (reduced foot clearance)
- Decreased step length bilaterally
- Widened base of support (instability compensation)
- Reduced arm swing
- Forward head posture during walking
- Hesitant with turns (multiple small steps)
- High double-support time (fear of single-leg stance)
- Visible knee flexion throughout gait cycle (cannot fully extend)

**8. Stair Climbing - 4 Steps**
```json
{
  "test_name": "Stair Ascent and Descent - 4 steps",
  "ascent_time": 22,
  "descent_time": 28,
  "use_of_rail": "Required for safety",
  "quality_score": 32,
  "pattern": "Step-to pattern (one step at a time)"
}
```

**Results:**
- **Ascent (going up):**
  - Time: 22 seconds for 4 steps (very slow)
  - Pattern: Step-to (lead with right, bring left to meet)
  - Required handrail for all steps
  - Significant pull with arms (compensating for leg weakness)
  - Pain reported: 7/10 in knees
  
- **Descent (going down):**
  - Time: 28 seconds for 4 steps (very slow, fearful)
  - Pattern: Step-to (lead with left, bring right to meet)
  - Required handrail and therapist standby
  - Demonstrated high fear of falling
  - Described as "most difficult activity"

**Observed Deficiencies:**
- Unable to perform step-over-step pattern
- Severe quad weakness limits eccentric control on descent
- Significant fear of falling (especially descending)
- Requires handrail for safety
- Functional limitation for community mobility

---

### **BIOMECHANICAL ANALYSIS RESULTS**

**Overall Movement Quality:** 32/100 (Poor)

**Fall Risk Assessment:** HIGH RISK (Multiple indicators present)

**Fall Risk Factors Identified:**

1. **History of Falls** ✅
   - 2 falls in past month
   - Both indoors (typically indicates high risk)

2. **Balance Impairment** ✅
   - Berg Balance Scale: 38/56 (HIGH RISK)
   - Unable to stand on one leg
   - Impaired tandem/semi-tandem stance
   - Poor functional reach

3. **Gait Abnormalities** ✅
   - Gait speed: 0.62 m/s (IMPAIRED)
   - TUG: 18.5 seconds (HIGH RISK)
   - Shuffling gait (trip risk)
   - Widened base (instability)

4. **Lower Extremity Weakness** ✅
   - Quad strength: 3/5 (bilateral)
   - Hip abductors: 3/5 (critical for balance)
   - Poor chair stand performance

5. **Sensory Impairment** ✅
   - Peripheral neuropathy (feet)
   - Impaired proprioception
   - Decreased vibration sense

6. **Pain** ✅
   - Chronic knee pain
   - Limits functional movement
   - Avoidance behaviors

7. **Environmental Factors** ✅
   - Area rugs (Fall #2 cause)
   - Poor lighting (bathroom fall)
   - Lives alone

8. **Fear of Falling** ✅
   - Activities-Specific Balance Confidence (ABC) Scale: 58/100 (LOW)
   - Score <67% indicates low confidence, increased fall risk
   - Avoidance of activities due to fear

9. **Medication Factors** ✅
   - Takes 11 medications
   - Includes medications associated with fall risk:
     - Gabapentin (dizziness)
     - Meloxicam (dizziness)
     - Lisinopril (orthostatic hypotension)

10. **Medical Comorbidities** ✅
    - Osteoarthritis (severe)
    - Diabetes with neuropathy
    - Osteoporosis (fracture risk)
    - Mild cognitive impairment

**STEADI Fall Risk Score: 12/12 risk factors present - HIGHEST RISK CATEGORY**

---

**Deficiencies Identified:**

1. **Severe Lower Extremity Weakness**
   - **Area:** Quadriceps, hip abductors, ankle dorsiflexors
   - **Severity:** Severe
   - **Description:** Quad strength 3/5 bilateral (50% below normal). Unable to rise from chair without arms. Cannot stand on one leg. Significant functional limitation.
   - **Clinical Significance:** Primary contributor to fall risk. Limits ability to recover from loss of balance. Affects all functional mobility (stairs, transfers, walking).

2. **Impaired Balance (Multi-factorial)**
   - **Area:** Static and dynamic balance, postural control
   - **Severity:** Severe
   - **Description:** Berg Balance Scale 38/56 (high fall risk). Unable to complete single leg stance, tandem stance. TUG 18.5 seconds. Poor balance confidence (ABC 58%).
   - **Clinical Significance:** Direct predictor of falls. Limits safe community ambulation. Reduces independence.

3. **Gait Dysfunction**
   - **Area:** Walking pattern, speed, step length, stability
   - **Severity:** Moderate-Severe
   - **Description:** Gait speed 0.62 m/s (severely impaired). Shuffling steps, widened base, reduced step length (35 cm vs. 60 cm normal). High double support time (45%).
   - **Clinical Significance:** Functional limitation for community mobility. Increased trip/fall risk. Unable to cross streets safely (need >1.0 m/s).

4. **Joint Range of Motion Limitations**
   - **Area:** Bilateral knee flexion contractures, ankle dorsiflexion
   - **Severity:** Moderate
   - **Description:** Knee extension -9° deficit bilateral (flexion contractures). Ankle dorsiflexion 6° (normal 20°). Hip flexion limited.
   - **Clinical Significance:** Altered gait mechanics, increased fall risk, limits functional movements (stairs, transfers).

5. **Chronic Pain (Bilateral Knees)**
   - **Area:** Osteoarthritis-related pain
   - **Severity:** Moderate-Severe
   - **Description:** Resting pain 3/10, movement pain 6-7/10. Severe with stairs (7/10), squatting (8/10). Limits functional activities.
   - **Clinical Significance:** Activity avoidance leads to deconditioning, worsening weakness. Pain-related fear contributes to fall risk.

6. **Sensory Deficits (Feet)**
   - **Area:** Proprioception, vibration sense, light touch
   - **Severity:** Moderate
   - **Description:** Diabetic peripheral neuropathy. Impaired proprioception (great toe position sense). Absent vibration sense at ankles.
   - **Clinical Significance:** Reduced ability to detect foot position and surface changes. Increased trip/fall risk.

7. **Fear of Falling / Low Balance Confidence**
   - **Area:** Psychological barrier to activity
   - **Severity:** Moderate-Severe
   - **Description:** ABC Scale 58/100 (low confidence). Avoids activities due to fear. Limits outdoor walking.
   - **Clinical Significance:** Activity avoidance creates deconditioning cycle. Self-imposed functional limitations. Reduced quality of life.

8. **Postural Abnormalities**
   - **Area:** Kyphotic posture, forward head position
   - **Severity:** Moderate
   - **Description:** Rounded shoulders, forward head (>2 inches anterior to plumb line). Affects center of mass.
   - **Clinical Significance:** Alters balance strategies. Increases fall risk during perturbations.

---

### **RECOMMENDED EXERCISES - MARGARET CHEN**

**Exercise Program Goals:**
1. **PRIMARY:** Reduce fall risk (improve balance, strength)
2. Improve lower extremity strength (quads, hip abductors)
3. Enhance gait quality and speed
4. Reduce pain through strengthening and ROM
5. Improve balance confidence
6. Maintain independence in ADLs/IADLs

**CRITICAL SAFETY NOTE:**
- All exercises performed with stable support (chair, counter, parallel bars)
- Therapist supervision initially (first 2-3 weeks)
- Home exercises only after demonstrated safety
- Medical alert pendant worn at all times

---

**Exercise Prescription (3x/week supervised PT + daily home program)**

#### **1. BALANCE TRAINING (Most Critical)**

**Exercise A: Static Balance Progression**
- **Level 1:** Feet together stance with hand support (chair back)
  - Goal: 30 seconds x 3 sets
  - Progress: → One hand support → Fingertip support → No support
- **Level 2:** Semi-tandem stance with support
  - Goal: 20 seconds x 3 sets each foot
  - Progress: → Reduce hand support
- **Level 3 (Advanced):** Tandem stance (heel-to-toe) with support
  - Goal: 10 seconds x 3 sets
  - Not attempted until Level 2 mastered
- **Frequency:** Daily (at kitchen counter for safety)
- **Focus:** Level pelvis, tall posture, breathe normally

**Exercise B: Dynamic Weight Shifts**
- **Description:** Stand with hand support, shift weight side to side
- **Sets/Reps:** 2 sets x 10 shifts each direction
- **Progression:** Increase speed, reduce hand support, add forward/backward
- **Frequency:** Daily
- **Focus:** Controlled movement, lift foot completely off ground

**Exercise C: Standing Marching**
- **Description:** Marching in place with hand support
- **Sets/Reps:** 2 sets x 20 steps
- **Progression:** Higher knee lift, faster cadence, reduce hand support
- **Frequency:** Daily
- **Focus:** Lift knee to 90° if possible, maintain balance

**Exercise D: Heel-Toe Raises**
- **Description:** Stand with hand support, rise to toes, then rock back to heels
- **Sets/Reps:** 2 sets x 10 reps (5 heel raises, 5 toe raises)
- **Progression:** Increase repetitions, reduce hand support
- **Frequency:** Daily
- **Focus:** Controlled movement, full range

#### **2. LOWER EXTREMITY STRENGTHENING**

**Exercise A: Chair Sit-to-Stand**
- **Description:** Sit and stand from chair
- **Sets/Reps:** 3 sets x 8-10 reps
- **Progression:** 
  - Phase 1: Use armrests
  - Phase 2: Arms crossed on chest
  - Phase 3: Add pause at bottom (eccentric control)
  - Phase 4: Lower chair height
- **Frequency:** Daily
- **Focus:** Lean forward, push through heels, squeeze glutes at top

**Exercise B: Mini Squats (Quarter Squats)**
- **Description:** Partial squat with hand support at counter
- **Sets/Reps:** 3 sets x 10 reps
- **Progression:** Increase depth gradually (only to pain tolerance)
- **Frequency:** Daily
- **Focus:** Sit back, knees track over toes, controlled movement

**Exercise C: Step-Ups (Low Step)**
- **Description:** Step up onto 4-inch step with handrail support
- **Sets/Reps:** 2 sets x 8 reps each leg
- **Progression:** 
  - Phase 1: Step up, step down (lead with same leg)
  - Phase 2: Alternate legs
  - Phase 3: Increase step height (6 inches)
- **Frequency:** 3x/week
- **Focus:** Push through heel, stand fully upright on step

**Exercise D: Hip Abduction (Side Leg Raises)**
- **Description:** Stand with hand support, lift leg to side
- **Sets/Reps:** 3 sets x 10 reps each leg
- **Progression:** Add ankle weight (1 lb → 2 lb), reduce hand support
- **Frequency:** Daily
- **Focus:** Keep toes forward, don't lean, controlled movement

**Exercise E: Heel Slides (Knee Flexion ROM)**
- **Description:** Seated, slide heel toward buttocks to bend knee
- **Sets/Reps:** 3 sets x 10 reps each leg
- **Progression:** Increase flexion angle, add hold at end range
- **Frequency:** Daily
- **Focus:** Gentle stretch, no pain past 5/10

**Exercise F: Terminal Knee Extension**
- **Description:** Seated, straighten knee fully, hold, lower slowly
- **Sets/Reps:** 3 sets x 15 reps each leg
- **Progression:** Add ankle weight (1 lb → 2 lb → 3 lb)
- **Frequency:** Daily
- **Focus:** Lock knee out fully, squeeze quad, 5-second hold

#### **3. GAIT TRAINING**

**Exercise A: Tandem Walking (Heel-Toe)**
- **Description:** Walk along straight line (tape on floor) with hand support
- **Distance:** 10 feet x 3 repetitions
- **Progression:** Reduce hand support, increase speed, add head turns
- **Frequency:** Daily
- **Focus:** Heel of front foot touches toe of back foot, maintain balance

**Exercise B: High Stepping (Exaggerated Gait)**
- **Description:** Walk with exaggerated knee lift
- **Distance:** 20 feet x 3 repetitions
- **Progression:** Increase knee height, faster cadence
- **Frequency:** Daily
- **Focus:** Lift knee to 90° if possible, clear foot from ground

**Exercise C: Backward Walking**
- **Description:** Walk backwards with hand on counter for safety
- **Distance:** 10 feet x 3 repetitions
- **Progression:** Increase distance, reduce hand support
- **Frequency:** Daily
- **Focus:** Reach back with toes, maintain upright posture

**Exercise D: Sidestepping**
- **Description:** Side-step along counter (like crab walk)
- **Distance:** 10 feet each direction x 2 repetitions
- **Progression:** Increase speed, reduce hand support, add resistance band
- **Frequency:** Daily
- **Focus:** Wide steps, don't cross feet, control movement

#### **4. FLEXIBILITY & ROM**

**Exercise A: Ankle Pumps (Dorsiflexion/Plantarflexion)**
- **Description:** Seated, pump ankles up and down
- **Sets/Reps:** 3 sets x 20 reps
- **Frequency:** Multiple times daily (especially morning/evening)
- **Focus:** Full range of motion, slow and controlled

**Exercise B: Ankle Circles**
- **Description:** Seated, circle ankles clockwise and counter-clockwise
- **Sets/Reps:** 2 sets x 10 circles each direction, each ankle
- **Frequency:** Daily
- **Focus:** Full circular motion, stretch end ranges

**Exercise C: Seated Hamstring Stretch**
- **Description:** Seated on chair, extend leg, lean forward at hips
- **Sets/Reps:** 3 sets x 30-second hold each leg
- **Frequency:** Daily
- **Focus:** Keep back straight, feel stretch behind knee

**Exercise D: Standing Calf Stretch**
- **Description:** Stand facing wall, step back with one leg, lean forward
- **Sets/Reps:** 3 sets x 30-second hold each leg
- **Frequency:** Daily (especially before walking)
- **Focus:** Keep heel down, back knee straight, feel stretch in calf

#### **5. POSTURE & CORE**

**Exercise A: Chin Tucks**
- **Description:** Seated or standing, pull chin back (make double chin)
- **Sets/Reps:** 3 sets x 10 reps, 5-second hold
- **Frequency:** Multiple times daily
- **Focus:** Lengthen back of neck, don't tilt head down

**Exercise B: Shoulder Blade Squeezes**
- **Description:** Seated, squeeze shoulder blades together
- **Sets/Reps:** 3 sets x 10 reps, 5-second hold
- **Frequency:** Daily
- **Focus:** Pinch shoulder blades, open chest

**Exercise C: Seated March with Arm Reach**
- **Description:** Seated, march in place while reaching overhead alternately
- **Sets/Reps:** 2 sets x 20 total (10 each side)
- **Frequency:** Daily
- **Focus:** Tall posture, coordinated movement

#### **6. FUNCTIONAL ACTIVITIES**

**Exercise A: Practice Transfers**
- Sit-to-stand from various heights (bed, toilet, car seat)
- Goal: Minimize armrest use over time
- **Frequency:** Incorporate into daily activities

**Exercise B: Stair Practice**
- Practice on 2-3 steps with handrail
- Ascent and descent, both step-to and step-over-step patterns
- **Frequency:** 2-3x/week (supervised initially)

**Exercise C: Floor-to-Stand Practice**
- Practice getting up from floor using step-by-step method:
  1. Roll to side
  2. Get on hands and knees
  3. Crawl to stable surface
  4. Half-kneel position
  5. Stand with support
- **Frequency:** 1x/week (supervised until confident)
- **Purpose:** Prepare for potential fall scenario

---

**WEEKLY SCHEDULE:**

**Monday/Wednesday/Friday (Supervised PT):**
- Balance exercises (all)
- Strengthening exercises (all)
- Gait training
- Stair practice
- Duration: 45-60 minutes per session

**Tuesday/Thursday/Saturday/Sunday (Home Program):**
- Balance exercises (A, B, C)
- Chair sit-to-stands
- Mini squats
- Hip abduction
- Terminal knee extension
- Flexibility exercises (all)
- Duration: 30-40 minutes
- Caregiver supervision recommended initially

**Daily:**
- Ankle pumps (morning/evening)
- Posture exercises (chin tucks, shoulder squeezes)
- Walking practice (indoor, with cane)

---

### **EXPECTED OUTCOMES**

**4-Week Baseline (Current):**
- Berg Balance Scale: 38/56 (HIGH FALL RISK)
- TUG: 18.5 seconds
- Gait speed: 0.62 m/s
- Chair stands (30-sec): 6 reps
- Falls: 2 in past month
- Pain: 6-7/10 with activity
- Balance confidence (ABC): 58%
- Movement quality: 32/100

**8-Week Target (Realistic Short-term Goals):**
- Berg Balance Scale: 42-45/56 (MODERATE FALL RISK) - 10-18% improvement
- TUG: 14-16 seconds (20-25% improvement)
- Gait speed: 0.75-0.80 m/s (20-30% improvement)
- Chair stands: 9-10 reps (50-65% improvement)
- Falls: ZERO (with consistent exercise + home modifications)
- Pain: 4-5/10 with activity (20-40% reduction)
- Balance confidence: 65-70% (improved confidence)
- Movement quality: 42-48/100 (30-50% improvement)

**Functional Outcomes (8 weeks):**
- Able to walk to community center independently with cane
- Climb stairs with handrail using step-over-step pattern (ascent)
- Stand from chair without using armrests (most of the time)
- Perform ADLs with increased confidence and safety
- Reduced fear of falling
- Maintained independent living status

**16-Week Target (Intermediate Goals):**
- Berg Balance Scale: 46-48/56 (LOW-MODERATE FALL RISK)
- TUG: 12-13 seconds (30-35% improvement from baseline)
- Gait speed: 0.85-0.90 m/s (approaching normal)
- Chair stands: 11-12 reps (approaching age norm)
- Single leg stance: 3-5 seconds each leg (achievable with practice)
- Pain: 3-4/10 with activity (40-50% reduction)
- Balance confidence: 72-75%
- Movement quality: 52-58/100

**Functional Outcomes (16 weeks):**
- Walk outdoors confidently with cane on level surfaces
- Navigate community with minimal assistance
- Climb/descend stairs safely with handrail (step-over-step both directions)
- Participate in senior center activities without fear
- Reduced medication usage for pain (fewer NSAIDs)
- Improved sleep quality (less pain)

**Long-term Goals (6-12 months):**
- Berg Balance Scale: >50/56 (LOW FALL RISK)
- TUG: <10-12 seconds
- Gait speed: >1.0 m/s (safe community ambulation, cross streets)
- Maintain zero falls with continued exercise
- Able to play with grandchildren on floor
- Travel to visit daughter (Portland) with confidence
- Consider discontinuing cane for indoor use (with physician approval)
- Movement quality: 62-68/100

---

### **ADDITIONAL INTERVENTIONS & RECOMMENDATIONS**

**1. HOME SAFETY EVALUATION & MODIFICATIONS**

**Identified Hazards:**
- Area rugs (caused Fall #2) → REMOVE or secure with double-sided tape
- Poor bathroom lighting (contributed to Fall #1) → Install nightlight, increase wattage
- Low toilet seat → Install raised toilet seat (add 4 inches)
- Bathtub without shower chair → Already has chair (good!)
- No grab bars in strategic locations → Add near bed, in hallway

**Recommended Modifications:**
- Remove all area rugs or secure edges
- Increase lighting throughout apartment (especially bathroom, hallways)
- Install grab bars near toilet, in shower, beside bed
- Organize apartment to minimize reaching/bending
- Keep frequently used items at waist height
- Ensure phone/medical alert pendant always accessible
- Consider motion-sensor nightlights for nighttime bathroom trips
- Remove clutter from walkways
- Secure electrical cords
- Non-slip mat in bathtub (if not already present)

**2. ASSISTIVE DEVICE RECOMMENDATIONS**

**Current:** Single-point cane (uses inconsistently)

**Recommendations:**
- Continue cane use (especially outdoors and stairs)
- Ensure cane is proper height (wrist level when standing upright, elbow ~30° flexion)
- Consider quad cane for outdoor use (increased stability)
- **DO NOT recommend walker at this time** (patient resistance, stigma) - revisit if no progress in 8 weeks
- Ensure cane has good rubber tip (replace if worn)

**3. MEDICATION REVIEW**

**Concern:** 11 medications, several with fall-risk potential

**Recommendations:**
- Refer back to PCP for medication review
- Consider:
  - Timing of medications (avoid clustering medications that cause dizziness)
  - Gabapentin: Taken at night (good), but may cause morning grogginess
  - Blood pressure medications: Check for orthostatic hypotension
  - Meloxicam: Daily NSAID use - discuss alternatives (topical NSAIDs, acetaminophen)
- Ensure patient takes medications as prescribed (no missed doses)
- Use pill organizer (already doing - good!)

**4. VISION & HEARING CHECKS**

**Recommendations:**
- Refer to ophthalmologist for eye exam (last cataract surgery 2018, needs update)
- Check glasses prescription (may need bifocal adjustment for fall prevention)
- Hearing test (impaired hearing can affect balance)

**5. FOOTWEAR ASSESSMENT**

**Current:** Wears slip-on shoes (easy on/off but may lack support)

**Recommendations:**
- Proper athletic shoes with:
  - Good arch support
  - Non-slip soles
  - Secure closure (laces or Velcro, not slip-on)
  - Low heel height
- Avoid:
  - Backless slippers
  - High heels
  - Worn-out soles
  - Shoes that are too loose

**6. NUTRITIONAL CONSIDERATIONS**

**Concerns:** Diabetes, osteoporosis, sarcopenia

**Recommendations:**
- Adequate protein intake (1.0-1.2 g/kg/day = 68-82g protein)
  - Current weight: 68 kg
  - Ensures muscle maintenance/growth
- Vitamin D supplementation (already taking - good!)
- Calcium intake (dietary + supplement)
- Ensure adequate hydration (dehydration affects balance)
- Refer to dietitian if needed

**7. FALL PREVENTION EDUCATION**

**Topics to Cover:**
- Recognize fall risk factors
- Safe movement strategies (e.g., sit at edge of bed before standing)
- What to do if you fall (practiced floor-to-stand)
- Importance of reporting falls/near-falls
- Medication side effects to watch for
- When to use assistive device
- Home safety awareness

**8. SOCIAL SUPPORT & MONITORING**

**Recommendations:**
- Continue medical alert pendant (excellent!)
- Daily check-in with son or daughter (phone call or text)
- Encourage participation in senior center activities
  - Social interaction reduces depression/isolation
  - Physical activities (tai chi, seated yoga classes)
  - Builds community support network
- Consider emergency response plan with neighbors
- Ensure family members aware of fall risk and prevention strategies

**9. PSYCHOLOGICAL SUPPORT**

**Concern:** Fear of falling, anxiety about independence

**Recommendations:**
- Address fear openly (normalize concerns)
- Celebrate small victories (build confidence)
- Set realistic, achievable goals
- Consider support group for fall survivors (if available)
- May benefit from counseling if anxiety becomes overwhelming

---

### **CLINICAL NOTES**

**Prognosis:** Fair to Good with consistent participation in therapy and home program

**Positive Factors:**
- Motivated to maintain independence
- Strong family support (children involved)
- Lives in supportive senior community
- Good cognitive function (mild impairment but able to learn)
- Medically stable (diabetes and hypertension controlled)
- Already has medical alert system
- Financially able to afford modifications/equipment

**Barriers to Success:**
- Multiple comorbidities (osteoarthritis, diabetes, osteoporosis)
- Significant baseline weakness and balance impairment
- Chronic pain limits exercise tolerance
- Fear of falling (may resist challenging exercises)
- Lives alone (compliance monitoring challenging)
- Medicare limitations (may not cover all therapy visits)

**Risk Factors for Falls (Ongoing):**
- High fall risk score (STEADI 12/12)
- Previous falls (2 in past month)
- Multiple medications
- Sensory deficits (peripheral neuropathy)
- Environmental hazards (partially addressed)

**Recommendations:**
1. Physical therapy 3x/week for 12 weeks (36 visits)
   - May need to extend based on progress
2. Home exercise program daily (caregiver supervision initially)
3. Home safety evaluation (OT consult or PT home visit)
4. Refer to PCP for medication review
5. Vision and hearing assessments
6. Consider aquatic therapy (if available and covered by insurance)
   - Buoyancy reduces joint stress
   - Excellent for strengthening with low pain
7. Re-evaluation at 8 weeks, 16 weeks, and discharge
8. Fall diary (track near-falls, falls, circumstances)
9. Pain management strategies (ice/heat, topical analgesics)
10. Consider tai chi or chair yoga class at senior center (maintenance program after PT discharge)

**Cautions:**
- Monitor blood pressure before/after exercise (orthostatic hypotension risk)
- Watch for signs of overexertion (increased pain, fatigue, dizziness)
- Ensure safety at all times (therapist/caregiver supervision initially)
- Stop exercise if pain exceeds 5/10
- Modify exercises as needed based on pain/tolerance
- Close monitoring for fall events (report immediately)

**Billing Codes:**
- CPT 97110 (Therapeutic exercise) - 30 min
- CPT 97112 (Neuromuscular re-education) - 15 min
- CPT 97116 (Gait training) - 15 min
- CPT 97530 (Therapeutic activities) - 15 min
- ICD-10: Z91.81 (History of falling)
- ICD-10: R26.81 (Unsteadiness on feet)
- ICD-10: M17.0 (Bilateral primary osteoarthritis of knee)
- ICD-10: M62.84 (Sarcopenia)
- ICD-10: E11.40 (Type 2 diabetes mellitus with diabetic neuropathy)
- ICD-10: M80.08 (Age-related osteoporosis with current pathological fracture)

**Discharge Plan:**
- Goal: Independent ambulation with cane, zero falls, Berg Balance >48
- Anticipated discharge: 12-16 weeks
- Transition to community-based exercise program (senior center)
- Maintenance home program (simplified version, 3-4x/week)
- Follow-up with PT PRN (as needed for flare-ups)
- Annual fall risk screening recommended

---

## 📊 COMPARISON TABLE: THREE CASES

| **Category** | **Emma (8 years)** | **Marcus (28 years)** | **Margaret (72 years)** |
|--------------|--------------------|-----------------------|-------------------------|
| **Age** | 8 | 28 | 72 |
| **Primary Diagnosis** | Developmental Coordination Disorder | ACL Reconstruction (6 weeks post-op) | Multiple falls, Osteoarthritis |
| **Movement Quality** | 38/100 (Poor) | 54/100 (Fair) | 32/100 (Poor) |
| **Main Limitation** | Balance & coordination | Quad weakness, proprioception | Balance, strength, fall risk |
| **Balance Test** | 3.2 sec single leg | 12 sec single leg (surgical side) | Unable to perform single leg |
| **Gait Speed** | Normal for age but clumsy | Antalgic gait, limp | 0.62 m/s (severely impaired) |
| **Pain Level** | 0-1/10 (minimal) | 4/10 (surgical knee) | 6-7/10 (bilateral knees) |
| **Fall Risk** | Moderate (frequent falls during play) | Low (post-surgical precautions) | HIGH (2 falls in past month) |
| **Therapy Frequency** | 1x/week + daily HEP | 2x/week + daily HEP | 3x/week + daily HEP |
| **Therapy Duration** | 8 weeks | 12-24 weeks (phased return to sport) | 12-16 weeks |
| **Expected Outcome (8 weeks)** | 55-65/100 quality score | 70-75/100 quality score | 42-48/100 quality score |
| **Long-term Goal** | Age-appropriate coordination | Full return to competitive soccer | Zero falls, maintain independence |
| **Prognosis** | Good (developmental plasticity) | Excellent (young, healthy, motivated) | Fair-Good (multiple barriers) |
| **Key Exercises** | Balance training, coordination | Quad strengthening, proprioception | Balance, strength, functional mobility |
| **Unique Considerations** | School accommodations, parent involvement | Return-to-sport protocol, graft protection | Home safety, fall prevention, multiple comorbidities |

---

## 📝 USING THESE PROFILES IN PhysioMotion

### **How to Test with These Patients:**

**1. Patient Intake:**
- Use the demographics section to fill out patient intake form
- Enter all medical history information
- Document medications, allergies, functional limitations

**2. Assessment:**
- Select appropriate camera (laptop, phone, or Femto Mega)
- Perform tests as described in each profile
- System should capture 33 joints (MediaPipe) or 32 joints (Femto Mega)
- Record movements as specified (duration, repetitions)

**3. Expected System Outputs:**

**For Emma (Pediatric):**
```json
{
  "movement_quality_score": 38,
  "deficiencies": [
    {
      "area": "Balance control",
      "severity": "severe",
      "description": "Single leg balance 70% below age norms"
    },
    {
      "area": "Ankle mobility",
      "severity": "moderate",
      "description": "Dorsiflexion limited to 10° (50% reduction)"
    },
    {
      "area": "Hip and knee strength",
      "severity": "moderate",
      "description": "Unable to maintain proper squat form"
    }
  ],
  "recommendations": [
    "Balance training with support",
    "Ankle stretching exercises",
    "Strengthening program for lower extremities",
    "Proprioception activities (fun, game-based)"
  ]
}
```

**For Marcus (Young Adult Post-Surgical):**
```json
{
  "movement_quality_score": 54,
  "asymmetry_index": 35,
  "deficiencies": [
    {
      "area": "Left quadriceps strength",
      "severity": "severe",
      "description": "3 cm atrophy, MMT 3+/5, 60% deficit vs. right"
    },
    {
      "area": "Proprioception - left knee",
      "severity": "moderate-severe",
      "description": "Balance 60% worse than right, increased re-injury risk"
    },
    {
      "area": "Movement symmetry",
      "severity": "moderate",
      "description": "Weight shift 70/30, step length 24% asymmetry"
    }
  ],
  "recommendations": [
    "Progressive quad strengthening (SLR, TKE, wall sits)",
    "Balance and proprioception training",
    "Gait retraining for symmetry",
    "Hip strengthening to prevent valgus",
    "Progress ROM to full flexion"
  ]
}
```

**For Margaret (Elderly):**
```json
{
  "movement_quality_score": 32,
  "fall_risk_score": "HIGH",
  "deficiencies": [
    {
      "area": "Lower extremity strength",
      "severity": "severe",
      "description": "Quad 3/5, hip abductors 3/5, unable to rise from chair without arms"
    },
    {
      "area": "Balance - static and dynamic",
      "severity": "severe",
      "description": "Berg 38/56 (high fall risk), unable to single leg stance"
    },
    {
      "area": "Gait dysfunction",
      "severity": "moderate-severe",
      "description": "Gait speed 0.62 m/s, shuffling, widened base, high fall risk"
    },
    {
      "area": "Sensory deficits",
      "severity": "moderate",
      "description": "Diabetic neuropathy, impaired proprioception in feet"
    }
  ],
  "recommendations": [
    "PRIORITY: Balance exercises with support (prevent falls)",
    "Chair sit-to-stand progression",
    "Lower extremity strengthening",
    "Gait training (increase speed, step length)",
    "Home safety evaluation (URGENT)",
    "Fall prevention education",
    "Medical alert system (already has - good)"
  ],
  "safety_alerts": [
    "High fall risk - use standby assistance",
    "Home modifications required",
    "Medication review recommended"
  ]
}
```

---

## 🎯 CONCLUSION

These three comprehensive mock patient profiles represent:

1. **Pediatric case (Emma):** Developmental disorder, coordination challenges, building foundational skills
2. **Young adult case (Marcus):** Post-surgical rehabilitation, return to high-level athletics, asymmetry correction
3. **Elderly case (Margaret):** Fall risk, multiple comorbidities, maintaining independence

Each profile includes:
- ✅ Complete demographics and medical history
- ✅ Detailed assessment data (joint angles, quality scores, functional tests)
- ✅ Expected system outputs (biomechanical analysis)
- ✅ Comprehensive exercise prescriptions
- ✅ Expected outcomes at multiple time points
- ✅ Clinical notes and billing codes

**Use these profiles to:**
- Test the PhysioMotion system end-to-end
- Validate assessment accuracy
- Verify exercise recommendations
- Demonstrate system capabilities to stakeholders
- Train new users on the platform

---

**Document Version:** 1.0  
**Date:** January 31, 2026  
**Author:** PhysioMotion Development Team  
**Status:** Ready for system testing ✅
