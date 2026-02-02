# 📊 Medical Outcome Data Format

## Complete Example Output from Movement Analysis

This document shows the exact format and structure of data you'll receive from the real-time joint analysis system.

---

## 🎯 Complete Analysis Response

### **API Endpoint:**
```
PUT /api/tests/:id/analyze
```

### **Request Body:**
```json
{
  "skeleton_data": {
    "timestamp": 1698765432000,
    "landmarks": {
      "left_shoulder": { "x": 0.45, "y": 0.35, "z": -0.2, "visibility": 0.98 },
      "right_shoulder": { "x": 0.55, "y": 0.35, "z": -0.2, "visibility": 0.98 },
      "left_elbow": { "x": 0.50, "y": 0.45, "z": -0.15, "visibility": 0.95 },
      "right_elbow": { "x": 0.60, "y": 0.45, "z": -0.15, "visibility": 0.95 },
      "left_wrist": { "x": 0.55, "y": 0.55, "z": -0.1, "visibility": 0.92 },
      "right_wrist": { "x": 0.65, "y": 0.55, "z": -0.1, "visibility": 0.92 },
      "left_hip": { "x": 0.47, "y": 0.60, "z": -0.18, "visibility": 0.97 },
      "right_hip": { "x": 0.53, "y": 0.60, "z": -0.18, "visibility": 0.97 },
      "left_knee": { "x": 0.48, "y": 0.75, "z": -0.20, "visibility": 0.96 },
      "right_knee": { "x": 0.52, "y": 0.75, "z": -0.20, "visibility": 0.96 },
      "left_ankle": { "x": 0.48, "y": 0.88, "z": -0.22, "visibility": 0.94 },
      "right_ankle": { "x": 0.52, "y": 0.88, "z": -0.22, "visibility": 0.94 },
      "left_heel": { "x": 0.47, "y": 0.90, "z": -0.23, "visibility": 0.91 },
      "right_heel": { "x": 0.53, "y": 0.90, "z": -0.23, "visibility": 0.91 },
      "left_foot_index": { "x": 0.49, "y": 0.90, "z": -0.20, "visibility": 0.89 },
      "right_foot_index": { "x": 0.55, "y": 0.90, "z": -0.20, "visibility": 0.89 }
    }
  }
}
```

### **Response Structure:**
```json
{
  "success": true,
  "data": {
    "analysis_id": 789,
    "analysis": {
      "joint_angles": [...],
      "movement_quality_score": 72,
      "detected_compensations": [...],
      "recommendations": [...],
      "deficiencies": [...]
    }
  }
}
```

---

## 📐 Joint Angles Data

### **Structure:**
```typescript
interface JointAngle {
  joint_name: string;
  left_angle?: number;
  right_angle?: number;
  normal_range: [number, number];
  status: 'normal' | 'limited' | 'excessive';
  bilateral_difference?: number;
}
```

### **Example Output:**

```json
{
  "joint_angles": [
    {
      "joint_name": "Left Shoulder Flexion",
      "left_angle": 145.3,
      "normal_range": [0, 180],
      "status": "limited",
      "bilateral_difference": 12.7
    },
    {
      "joint_name": "Right Shoulder Flexion",
      "right_angle": 158.0,
      "normal_range": [0, 180],
      "status": "normal",
      "bilateral_difference": 12.7
    },
    {
      "joint_name": "Left Elbow Flexion",
      "left_angle": 139.2,
      "normal_range": [0, 150],
      "status": "normal",
      "bilateral_difference": 5.4
    },
    {
      "joint_name": "Right Elbow Flexion",
      "right_angle": 133.8,
      "normal_range": [0, 150],
      "status": "normal",
      "bilateral_difference": 5.4
    },
    {
      "joint_name": "Left Hip Flexion",
      "left_angle": 85.6,
      "normal_range": [0, 120],
      "status": "limited",
      "bilateral_difference": 8.2
    },
    {
      "joint_name": "Right Hip Flexion",
      "right_angle": 93.8,
      "normal_range": [0, 120],
      "status": "normal",
      "bilateral_difference": 8.2
    },
    {
      "joint_name": "Left Knee Flexion",
      "left_angle": 112.4,
      "normal_range": [0, 135],
      "status": "limited",
      "bilateral_difference": 14.3
    },
    {
      "joint_name": "Right Knee Flexion",
      "right_angle": 126.7,
      "normal_range": [0, 135],
      "status": "normal",
      "bilateral_difference": 14.3
    },
    {
      "joint_name": "Left Ankle Dorsiflexion",
      "left_angle": 82.1,
      "normal_range": [70, 110],
      "status": "limited",
      "bilateral_difference": 3.8
    },
    {
      "joint_name": "Right Ankle Dorsiflexion",
      "right_angle": 85.9,
      "normal_range": [70, 110],
      "status": "normal",
      "bilateral_difference": 3.8
    }
  ]
}
```

### **Joint Status Interpretation:**

| Status | Meaning | Action |
|--------|---------|--------|
| **normal** | Within expected range | Maintain current function |
| **limited** | Below normal range | Mobility/flexibility work needed |
| **excessive** | Above normal range | Stability/control work needed |

---

## 🎯 Movement Quality Score

### **Calculation Method:**
```
Score = 100
  - (10 points × number of limited joints)
  - (5 points × number of excessive joints)
  - (8 points × number of significant asymmetries >10°)
  - (7 points × number of compensations)

Final Score: 0-100
```

### **Example:**
```json
{
  "movement_quality_score": 72,
  "interpretation": "Fair movement quality"
}
```

### **Score Ranges:**

| Score | Quality | Clinical Interpretation | Treatment Intensity |
|-------|---------|------------------------|-------------------|
| **90-100** | Excellent | Minimal dysfunction | Preventive/maintenance |
| **80-89** | Good | Minor limitations | Light therapeutic exercise |
| **70-79** | Fair | Moderate limitations | Active therapeutic program |
| **60-69** | Poor | Significant limitations | Intensive therapy required |
| **<60** | Very Poor | Severe limitations | Comprehensive rehabilitation |

---

## ⚠️ Detected Compensations

### **Format:**
```typescript
string[] // Array of compensation pattern descriptions
```

### **Example Output:**
```json
{
  "detected_compensations": [
    "Left knee valgus detected - knee tracking inward",
    "Excessive forward trunk lean - core weakness or hip mobility limitation",
    "Left heel lifting - ankle dorsiflexion limitation",
    "Shoulder height asymmetry - possible lateral trunk lean or shoulder dysfunction",
    "Pelvic obliquity - one hip higher than the other"
  ]
}
```

### **Compensation Categories:**

#### **1. Knee Compensations:**
- Knee valgus (knee caving in)
- Knee varus (knee bowing out)
- Asymmetric knee tracking

#### **2. Trunk Compensations:**
- Excessive forward lean
- Lateral trunk lean
- Excessive lordosis
- Loss of neutral spine

#### **3. Ankle Compensations:**
- Heel lifting
- Excessive pronation
- Supination
- Asymmetric weight distribution

#### **4. Shoulder/Upper Body Compensations:**
- Shoulder elevation
- Shoulder height asymmetry
- Scapular winging
- Forward head posture

#### **5. Pelvic Compensations:**
- Pelvic obliquity
- Anterior pelvic tilt
- Posterior pelvic tilt
- Lateral pelvic shift

---

## 🩺 Deficiencies Report

### **Structure:**
```typescript
interface Deficiency {
  area: string;
  severity: 'mild' | 'moderate' | 'severe';
  description: string;
  recommended_exercises: number[];
}
```

### **Example Output:**

```json
{
  "deficiencies": [
    {
      "area": "Right Shoulder ROM",
      "severity": "moderate",
      "description": "Limited overhead shoulder flexion with 35° deficit. This affects daily activities requiring overhead reach such as placing items on high shelves, swimming, and overhead pressing movements.",
      "recommended_exercises": [11, 12]
    },
    {
      "area": "Hip Flexion",
      "severity": "moderate",
      "description": "Limited hip flexion range of motion bilaterally. This affects squat depth and can cause excessive forward trunk lean. Typical causes include hip flexor tightness, anterior hip impingement, or posterior chain restrictions.",
      "recommended_exercises": [1, 3]
    },
    {
      "area": "Ankle Dorsiflexion",
      "severity": "moderate",
      "description": "Limited ankle dorsiflexion range of motion. This can lead to heel lifting during squats and compensation patterns up the kinetic chain including knee valgus and forward trunk lean.",
      "recommended_exercises": [3]
    },
    {
      "area": "Lower Extremity Stability",
      "severity": "moderate",
      "description": "Instability and poor movement control in lower extremity. Knee valgus and balance deficits detected. This increases injury risk particularly for ACL injuries and patellofemoral pain syndrome.",
      "recommended_exercises": [4, 5]
    },
    {
      "area": "Core Stability",
      "severity": "moderate",
      "description": "Weak core stability causing compensatory trunk movements and poor postural control. This affects all movement patterns and increases risk of low back pain.",
      "recommended_exercises": [5, 13, 14]
    },
    {
      "area": "Bilateral Asymmetry",
      "severity": "moderate",
      "description": "Significant differences between left and right sides detected. Asymmetries found in: shoulder, knee. This increases injury risk and affects functional performance. Left side shows 15% deficit compared to right in shoulder flexion and 12% deficit in knee flexion.",
      "recommended_exercises": [7, 8]
    }
  ]
}
```

### **Severity Definitions:**

| Severity | ROM Deficit | Asymmetry | Impact | Priority |
|----------|------------|-----------|---------|----------|
| **Mild** | 0-15% | <10° | Minimal functional limitation | Low-Medium |
| **Moderate** | 15-35% | 10-20° | Moderate functional limitation | Medium-High |
| **Severe** | >35% | >20° | Significant functional limitation | High-Critical |

---

## 💡 AI Recommendations

### **Format:**
```typescript
string[] // Array of clinical recommendation strings
```

### **Example Output:**

```json
{
  "recommendations": [
    "Continue with prescribed exercise program focusing on identified deficiencies",
    "Multiple compensation patterns detected - recommend reducing exercise complexity until fundamental movement patterns improve",
    "Perform daily mobility work for 10-15 minutes focusing on identified ROM limitations",
    "Focus on single-leg balance and stability exercises to address asymmetry",
    "Progress ankle mobility work before advancing squat depth",
    "Incorporate core stabilization exercises in all movement patterns",
    "Re-assess movement quality in 2-3 weeks to track progress"
  ]
}
```

### **Recommendation Categories:**

#### **1. Exercise Intensity:**
- Reduce complexity
- Maintain current level
- Progress to next level
- Regress to fundamentals

#### **2. Frequency:**
- Daily mobility work
- 3x per week strength
- 5x per week stability
- Twice-weekly supervised

#### **3. Focus Areas:**
- ROM limitations
- Stability deficits
- Asymmetry correction
- Compensation patterns

#### **4. Timeline:**
- Immediate (within 1 week)
- Short-term (2-4 weeks)
- Medium-term (4-8 weeks)
- Long-term (8-12 weeks)

---

## 🏥 Complete Medical Note Format

### **SOAP Note Structure:**

```json
{
  "medical_note": {
    "subjective": {
      "chief_complaint": "Right shoulder pain and limited ROM after rotator cuff repair",
      "pain_level": 4,
      "activity_level": "light",
      "assessment_reason": "post_surgery",
      "goals": "Return to recreational tennis and overhead activities"
    },
    "objective": {
      "tests_completed": 4,
      "movement_quality_score": 72,
      "joint_measurements": {
        "right_shoulder_flexion": {
          "measured": 145.3,
          "expected": 180,
          "deficit": 34.7
        },
        "left_hip_flexion": {
          "measured": 85.6,
          "expected": 120,
          "deficit": 34.4
        },
        "left_knee_flexion": {
          "measured": 112.4,
          "expected": 135,
          "deficit": 22.6
        }
      },
      "compensations": [
        "Left knee valgus detected",
        "Excessive forward trunk lean",
        "Left heel lifting",
        "Shoulder height asymmetry",
        "Pelvic obliquity"
      ],
      "bilateral_asymmetries": {
        "shoulder": 12.7,
        "knee": 14.3
      }
    },
    "assessment": {
      "summary": "Patient demonstrates FAIR movement quality (72/100) with 6 significant deficiencies identified.",
      "primary_findings": [
        "Right Shoulder ROM - Moderate (35° flexion deficit)",
        "Left Hip Mobility - Moderate (34° flexion deficit)",
        "Lower Extremity Stability - Moderate (multiple compensations)"
      ],
      "functional_impact": "Moderate functional limitations present. Patient would benefit from targeted therapeutic exercise program focusing on shoulder and hip mobility, core stability, and lower extremity control.",
      "prognosis": "Good with adherence to exercise program. Expected timeline: 6-8 weeks for significant improvement."
    },
    "plan": {
      "treatment_approach": "Comprehensive therapeutic exercise program",
      "prescribed_exercises": [
        {
          "name": "Shoulder Wall Slides",
          "sets": 3,
          "reps": 15,
          "frequency": "5x per week",
          "target": "Right shoulder ROM"
        },
        {
          "name": "Hip Flexor Stretches",
          "sets": 3,
          "duration": "30 seconds each",
          "frequency": "5x per week",
          "target": "Hip mobility"
        },
        {
          "name": "Single Leg Balance",
          "sets": 3,
          "duration": "30 seconds each leg",
          "frequency": "3x per week",
          "target": "Lower extremity stability"
        },
        {
          "name": "Plank Hold",
          "sets": 3,
          "duration": "30 seconds",
          "frequency": "3x per week",
          "target": "Core stability"
        }
      ],
      "frequency": "3-5x per week for 6 weeks",
      "remote_monitoring": {
        "enabled": true,
        "tracking": "Daily exercise compliance via mobile app",
        "feedback": "Real-time form analysis",
        "reporting": "Weekly progress reports"
      },
      "reassessment": "Schedule follow-up assessment in 4 weeks",
      "patient_education": [
        "Proper exercise technique demonstrated",
        "Pain monitoring guidelines provided",
        "Progression criteria explained",
        "Warning signs for stopping exercise reviewed"
      ],
      "cpt_codes": ["97163", "97110", "97112", "98975", "98977"]
    }
  }
}
```

---

## 📊 Database Storage Format

### **1. Movement Test Record:**

```sql
-- movement_tests table
INSERT INTO movement_tests (
  id,
  assessment_id,
  test_name,
  test_category,
  test_status,
  skeleton_data,
  completed_at
) VALUES (
  456,
  123,
  'Functional Movement Screen',
  'mobility',
  'completed',
  '{
    "timestamp": 1698765432000,
    "landmarks": { ... }
  }',
  '2025-10-20 14:30:00'
);
```

### **2. Movement Analysis Record:**

```sql
-- movement_analysis table
INSERT INTO movement_analysis (
  id,
  test_id,
  joint_angles,
  deficiencies,
  movement_quality_score,
  ai_recommendations,
  ai_confidence_score,
  analyzed_at
) VALUES (
  789,
  456,
  '[
    {
      "joint_name": "Left Shoulder Flexion",
      "left_angle": 145.3,
      "status": "limited"
    },
    ...
  ]',
  '[
    {
      "area": "Right Shoulder ROM",
      "severity": "moderate",
      "description": "..."
    },
    ...
  ]',
  72,
  '[
    "Continue with prescribed exercise program...",
    ...
  ]',
  0.92,
  '2025-10-20 14:30:05'
);
```

---

## 🎨 Frontend Display Format

### **Results Screen HTML Structure:**

```html
<!-- Movement Quality Score -->
<div class="quality-score-card">
  <h3>Movement Quality Score</h3>
  <div class="score-display">72</div>
  <div class="score-label">out of 100</div>
  <div class="score-bar">
    <div class="score-fill" style="width: 72%"></div>
  </div>
  <p class="score-interpretation">Fair Movement Quality</p>
</div>

<!-- Deficiencies List -->
<div class="deficiencies-section">
  <h3>Detected Deficiencies (6)</h3>

  <div class="deficiency-card moderate">
    <div class="deficiency-header">
      <span class="severity-badge moderate">MODERATE</span>
      <h4>Right Shoulder ROM</h4>
    </div>
    <p class="deficiency-description">
      Limited overhead shoulder flexion with 35° deficit...
    </p>
    <div class="deficiency-metrics">
      <span>Measured: 145.3°</span>
      <span>Expected: 180°</span>
      <span>Deficit: 34.7°</span>
    </div>
  </div>

  <!-- More deficiency cards... -->
</div>

<!-- Recommendations -->
<div class="recommendations-section">
  <h3>AI Recommendations (7)</h3>
  <ul class="recommendations-list">
    <li>Continue with prescribed exercise program...</li>
    <li>Perform daily mobility work for 10-15 minutes...</li>
    <!-- More recommendations... -->
  </ul>
</div>
```

---

## 📱 Mobile App JSON Response

### **For Remote Patient Monitoring:**

```json
{
  "patient_id": 123,
  "assessment_id": 456,
  "test_id": 789,
  "analysis_id": 789,
  "timestamp": "2025-10-20T14:30:05Z",
  "summary": {
    "quality_score": 72,
    "quality_level": "fair",
    "total_deficiencies": 6,
    "severe_deficiencies": 0,
    "moderate_deficiencies": 6,
    "mild_deficiencies": 0
  },
  "key_findings": [
    "Right Shoulder ROM deficit: 35°",
    "Left Hip mobility limitation: 34°",
    "Multiple compensation patterns detected"
  ],
  "exercise_program": {
    "total_exercises": 4,
    "weekly_frequency": "3-5 sessions",
    "duration_weeks": 6,
    "exercises": [...]
  },
  "next_steps": [
    "Start prescribed exercises today",
    "Track daily in mobile app",
    "Report pain >5/10 immediately",
    "Re-assess in 4 weeks"
  ]
}
```

---

## 🔗 API Response Examples

### **Success Response:**

```json
{
  "success": true,
  "data": {
    "analysis_id": 789,
    "analysis": {
      "joint_angles": [...],
      "movement_quality_score": 72,
      "detected_compensations": [...],
      "recommendations": [...],
      "deficiencies": [...]
    }
  },
  "timestamp": "2025-10-20T14:30:05Z"
}
```

### **Error Response:**

```json
{
  "success": false,
  "error": "Invalid skeleton data: missing required landmarks",
  "error_code": "INVALID_SKELETON_DATA",
  "details": {
    "missing_landmarks": ["left_shoulder", "right_hip"]
  },
  "timestamp": "2025-10-20T14:30:05Z"
}
```

---

## 📋 Summary Table

| Component | Data Type | Example Value | Clinical Use |
|-----------|-----------|---------------|--------------|
| **Joint Angles** | Array | 145.3° | ROM assessment |
| **Quality Score** | Number | 72/100 | Overall function |
| **Compensations** | Array | 5 patterns | Movement dysfunction |
| **Deficiencies** | Array | 6 findings | Treatment targets |
| **Recommendations** | Array | 7 items | Clinical guidance |
| **Severity** | Enum | moderate | Priority level |
| **Asymmetry** | Number | 12.7° | Left-right difference |

---

**This format is what you'll receive from the analysis system for creating medical reports, exercise prescriptions, and treatment plans!** 📊
