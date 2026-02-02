-- Comprehensive SOAP notes and billing documentation for mock patients
-- Complete medical documentation for each case

-- Emma Rodriguez SOAP Note
INSERT INTO assessments (
  patient_id, clinician_id, assessment_date, assessment_type, status,
  subjective_notes, objective_findings, assessment_notes, treatment_plan
) VALUES (
  2, 1, '2026-01-31 10:45:00', 'follow_up', 'reviewed',
  'SUBJECTIVE: Emma is an 8-year-old female with developmental coordination disorder (DCD) presenting for initial physical therapy evaluation. Parents report frequent falls during play activities, difficulty with ball skills, and inability to ride a bicycle without training wheels. No pain complaints. Parents note she avoids sports activities and prefers sedentary play. Has been receiving occupational therapy at school 2x/week for past 6 months with some improvement. No previous physical therapy. Parents concerned about her motor development compared to peers.',
  'OBJECTIVE: Height: 127cm (50th percentile), Weight: 25.5kg (50th percentile). Vital signs stable. Gait: Wide-based, frequent step-offs during heel-to-toe walk. Balance: Single leg stance 3.2s right, 2.8s left (normal >10s for age). Squat: Limited depth with 75° hip flexion, 85° knee flexion, 10° ankle dorsiflexion. Forward reach: 15cm (normal 20-25cm). Coordination: Difficulty with rapid alternating movements. Functional Mobility: Requires assistance for complex movements.',
  'ASSESSMENT: Developmental Coordination Disorder (DCD) with significant functional limitations. Movement Quality Score: 38/100 (Poor). Primary deficits: 1) Severe balance deficit (70% below age norms), 2) Limited lower extremity flexibility (ankle dorsiflexion 50% reduction), 3) Poor postural control with excessive trunk lean, 4) Reduced proprioception with visual dependence. Prognosis: Good with consistent therapy and home program.',
  'PLAN: 1) Begin 8-week comprehensive PT program 3x/week focusing on balance training, strengthening, and coordination. 2) Home exercise program with parent education. 3) School-based recommendations for PE accommodations. 4) Re-evaluation in 4 weeks to assess progress. Goals: Improve single leg balance to 7-10 seconds, increase ankle dorsiflexion to 15-18°, achieve age-appropriate coordination for bicycle riding. CPT Codes: 97110 (therapeutic exercise), 97112 (neuromuscular reeducation), 97530 (therapeutic activities).'
);

-- Marcus Thompson SOAP Note
INSERT INTO assessments (
  patient_id, clinician_id, assessment_date, assessment_type, status,
  subjective_notes, objective_findings, assessment_notes, treatment_plan
) VALUES (
  3, 1, '2026-01-31 14:45:00', 'follow_up', 'reviewed',
  'SUBJECTIVE: Marcus is a 28-year-old male status post ACL reconstruction with hamstring autograft 6 weeks ago. Reports knee feels unstable, especially when walking on uneven surfaces. Having difficulty with exercises prescribed by surgeon. Semi-professional soccer player, very motivated to return to sport. Denies sharp pain but reports aching after activity. Using ibuprofen 600mg as needed. Wife assisting with exercises at home.',
  'OBJECTIVE: Surgical incision well-healed, no signs of infection. Girth measurements: Left thigh 52cm vs Right 55cm (-3cm atrophy), Left knee 38cm vs Right 40cm (-2cm). ROM: Left knee flexion 0-115° (limited) vs Right 0-140°. Strength: Left quad 3+/5 vs Right 5/5. Balance: Single leg stance: Right 30s, Left 12s (60% deficit). Gait: Antalgic pattern with 24% step length asymmetry. Ligamentous integrity: Stable to Lachman and pivot shift.',
  'ASSESSMENT: Status post ACL reconstruction, 6 weeks post-op. Movement Quality Score: 54/100 (Fair - appropriate for this stage). Primary issues: 1) Severe quadriceps weakness with 3cm atrophy, 2) Reduced proprioception (60% balance deficit on surgical side), 3) Movement asymmetry with antalgic gait, 4) Limited knee flexion (20° deficit). Healing appropriately for timeline. No signs of complications.',
  'PLAN: 1) Continue Phase 1 rehabilitation 2x/week for weeks 6-12. 2) Emphasize quadriceps strengthening and proprioception training. 3) Stationary bike for ROM. 4) Progress to functional activities as tolerated. 5) Re-assess at 12 weeks for progression to Phase 2. Return-to-sport timeline: 6-9 months with proper progression. CPT Codes: 97110 (therapeutic exercise), 97112 (neuromuscular reeducation), 97001 (PT evaluation).'
);

-- Margaret Chen SOAP Note
INSERT INTO assessments (
  patient_id, clinician_id, assessment_date, assessment_type, status,
  subjective_notes, objective_findings, assessment_notes, treatment_plan
) VALUES (
  4, 1, '2026-01-31 16:30:00', 'follow_up', 'reviewed',
  'SUBJECTIVE: Margaret is a 72-year-old female with multiple comorbidities presenting with significant fall risk. Reports falling twice in past month - once in bathroom, once tripping over rug. Lives alone in senior community, very fearful of falling again. Knee pain limits daily activities, especially stairs. Using cane inconsistently. Reports difficulty getting up from low chairs and prolonged standing. Family concerned about safety.',
  'OBJECTIVE: Vital signs: BP 142/85, HR 78, O2 sat 98%. Berg Balance Scale: 38/56 (HIGH FALL RISK). TUG: 18.5s (HIGH FALL RISK). 30-sec Chair Stand: 6 reps (POOR). Gait speed: 0.62 m/s (SEVERELY IMPAIRED). Strength: Quads 3/5, Hip abductors 3/5. ROM: Knee extension -9° deficit bilateral, ankle dorsiflexion 6°. Sensation: Diminished in feet (neuropathy). Functional: Requires assistance for stairs, transfers slow and painful.',
  'ASSESSMENT: HIGH FALL RISK - 12/12 STEADI risk factors present. Multiple comorbidities complicating rehab. Movement Quality Score: 32/100 (Poor). Primary deficits: 1) Severe lower extremity weakness (quad strength 50% below normal), 2) Multi-factorial balance impairment, 3) Gait dysfunction with shuffling pattern, 4) Chronic pain limiting activity. Complex case requiring multidisciplinary approach.',
  'PLAN: 1) Priority: Fall prevention - 3x/week supervised PT for 12 weeks. 2) Home safety assessment and modifications. 3) Balance training with stable support initially. 4) Strengthening program focusing on quads and hip abductors. 5) Gait training with appropriate assistive device. 6) Family education and emergency planning. 7) Re-assess fall risk monthly. Goals: Achieve zero falls, improve gait speed to >0.8 m/s, increase Berg score to >45. CPT Codes: 97110 (therapeutic exercise), 97112 (neuromuscular reeducation), 97530 (therapeutic activities), 97001 (PT evaluation).'
);

-- Billing codes insertion
INSERT INTO assessments (
  patient_id, clinician_id, assessment_date, assessment_type, status,
  assessment_notes, treatment_plan
) VALUES
-- Emma Rodriguez billing codes
(2, 1, '2026-01-31 11:00:00', 'follow_up', 'completed',
'BILLING DOCUMENTATION - Emma Rodriguez\nICD-10 Diagnoses: F82 (Specific developmental disorders of motor function), M62.81 (Muscle weakness - generalized), R27.8 (Other lack of coordination)\nCPT Codes: 97001 (PT evaluation - 45 min), 97110 (Therapeutic exercise - 15 min units), 97112 (Neuromuscular reeducation - 15 min units), 97530 (Therapeutic activities - 15 min units)\nBilling Units: 97001 x1, 97110 x8, 97112 x6, 97530 x4\nTotal Billable Time: 8 hours over 4 weeks\nInsurance: Blue Cross Blue Shield - Pre-authorization obtained\nModifiers: None required\nDocumentation: All functional limitations documented with objective measurements and goals.',
'PROGNOSIS: Good with consistent therapy. Expected improvement in balance and coordination over 8-week treatment period.'),

-- Marcus Thompson billing codes
(3, 1, '2026-01-31 15:00:00', 'follow_up', 'completed',
'BILLING DOCUMENTATION - Marcus Thompson\nICD-10 Diagnoses: M23.92 (Other internal derangements of knee - unspecified), S83.519A (Sprain of anterior cruciate ligament of knee - initial encounter), M62.531 (Muscle wasting and atrophy - quadriceps)\nCPT Codes: 97001 (PT evaluation - 45 min), 97110 (Therapeutic exercise - 15 min units), 97112 (Neuromuscular reeducation - 15 min units), 97140 (Manual therapy - 15 min units)\nBilling Units: 97001 x1, 97110 x12, 97112 x8, 97140 x4\nTotal Billable Time: 12 hours over 6 weeks\nInsurance: Aetna PPO - No pre-authorization required\nModifiers: GP (services delivered by PT)\nDocumentation: Post-surgical status clearly documented with objective strength and ROM measurements, functional goals established.',
'PROGNOSIS: Excellent with appropriate rehabilitation progression. Expected return to sport at 6-9 months with proper adherence to program.'),

-- Margaret Chen billing codes
(4, 1, '2026-01-31 17:00:00', 'follow_up', 'completed',
'BILLING DOCUMENTATION - Margaret Chen\nICD-10 Diagnoses: Z91.81 (History of falling), M17.9 (Osteoarthritis of knee - unspecified), M62.50 (Muscle wasting and atrophy - unspecified), R26.89 (Other abnormalities of gait and mobility), R29.6 (Repeated falls)\nCPT Codes: 97001 (PT evaluation - 45 min), 97110 (Therapeutic exercise - 15 min units), 97112 (Neuromuscular reeducation - 15 min units), 97530 (Therapeutic activities - 15 min units), 97750 (Physical performance test - 30 min)\nBilling Units: 97001 x1, 97110 x16, 97112 x12, 97530 x8, 97750 x2\nTotal Billable Time: 16 hours over 12 weeks\nInsurance: Medicare Part B + AARP Supplement - Authorization on file\nModifiers: GP (services delivered by PT), KX (exceeded therapy cap - documented medical necessity)\nDocumentation: Comprehensive fall risk assessment with multiple standardized tests, functional limitations clearly documented, medical necessity established.',
'PROGNOSIS: Fair to good with consistent participation. Multiple factors may limit progress, but improvement in fall risk factors expected with adherence to program.');