#!/usr/bin/env python3
"""
Boxer3D RAG Knowledge Base — lightweight embedded RAG for MSK assessment
Loads ROM norms, exercises, protocols, and clinical guidelines into memory.
Provides context injection for AI analysis prompts.
"""
import json, os

# ─── ROM Normative Data (compiled from AIMS MSK RAG) ───
ROM_NORMS = {
    "shoulder": {
        "flexion": {"min": 150, "max": 180, "avg": 161, "position": "sitting/standing"},
        "extension": {"min": 45, "max": 60, "avg": 54, "position": "prone"},
        "abduction": {"min": 150, "max": 180, "avg": 163, "position": "sitting"},
        "external_rotation": {"min": 80, "max": 90, "avg": 86, "position": "supine"},
        "internal_rotation": {"min": 60, "max": 70, "avg": 65, "position": "supine"}
    },
    "elbow": {
        "flexion": {"min": 130, "max": 150, "avg": 143, "position": "supine"},
        "extension": {"min": 0, "max": 10, "avg": 3, "position": "supine"}
    },
    "hip": {
        "flexion": {"min": 110, "max": 130, "avg": 120, "position": "supine"},
        "extension": {"min": 10, "max": 30, "avg": 16, "position": "prone"},
        "abduction": {"min": 30, "max": 50, "avg": 40, "position": "supine"},
        "external_rotation": {"min": 40, "max": 60, "avg": 47, "position": "sitting"},
        "internal_rotation": {"min": 30, "max": 45, "avg": 36, "position": "sitting"}
    },
    "knee": {
        "flexion": {"min": 120, "max": 150, "avg": 134, "position": "supine"},
        "extension": {"min": 0, "max": 10, "avg": 2, "position": "supine"}
    },
    "ankle": {
        "dorsiflexion": {"min": 10, "max": 20, "avg": 14, "position": "sitting"},
        "plantarflexion": {"min": 30, "max": 50, "avg": 37, "position": "sitting"}
    },
    "cervical": {
        "flexion": {"min": 40, "max": 60, "avg": 50, "position": "sitting"},
        "extension": {"min": 40, "max": 75, "avg": 58, "position": "sitting"},
        "rotation": {"min": 60, "max": 80, "avg": 70, "position": "sitting"},
        "lateral_flexion": {"min": 30, "max": 45, "avg": 37, "position": "sitting"}
    },
    "lumbar": {
        "flexion": {"min": 40, "max": 60, "avg": 50, "position": "standing"},
        "extension": {"min": 15, "max": 30, "avg": 23, "position": "standing"},
        "lateral_flexion": {"min": 15, "max": 25, "avg": 20, "position": "standing"}
    }
}

# ─── Exercise Library (compiled from AIMS MSK RAG) ───
EXERCISES = [
    {"name": "Pendulum Exercises (Codman)", "joint": "shoulder", "sets": 3, "reps": 15,
     "phase": "acute", "difficulty": "beginner", "cpt": "97110"},
    {"name": "Wall Slides", "joint": "shoulder", "sets": 3, "reps": 10,
     "phase": "recovery", "difficulty": "beginner", "cpt": "97110"},
    {"name": "Prone I,T,Y", "joint": "shoulder", "sets": 3, "reps": 8,
     "phase": "strengthening", "difficulty": "intermediate", "cpt": "97110"},
    {"name": "Quad Sets", "joint": "knee", "sets": 3, "reps": 15,
     "phase": "acute", "difficulty": "beginner", "cpt": "97110"},
    {"name": "Straight Leg Raises", "joint": "hip", "sets": 3, "reps": 10,
     "phase": "recovery", "difficulty": "beginner", "cpt": "97110"},
    {"name": "Clamshells", "joint": "hip", "sets": 3, "reps": 12,
     "phase": "strengthening", "difficulty": "intermediate", "cpt": "97110"},
    {"name": "Ankle Pumps", "joint": "ankle", "sets": 2, "reps": 20,
     "phase": "acute", "difficulty": "beginner", "cpt": "97110"},
    {"name": "Chin Tucks", "joint": "cervical", "sets": 3, "reps": 10,
     "phase": "recovery", "difficulty": "beginner", "cpt": "97110"},
    {"name": "Cat-Cow Stretch", "joint": "lumbar", "sets": 2, "reps": 10,
     "phase": "acute", "difficulty": "beginner", "cpt": "97112"},
    {"name": "Dead Bug", "joint": "lumbar", "sets": 3, "reps": 8,
     "phase": "strengthening", "difficulty": "intermediate", "cpt": "97110"},
    {"name": "Hip Hinge", "joint": "hip", "sets": 3, "reps": 10,
     "phase": "recovery", "difficulty": "intermediate", "cpt": "97530"},
    {"name": "Standing Calf Raises", "joint": "ankle", "sets": 3, "reps": 12,
     "phase": "strengthening", "difficulty": "intermediate", "cpt": "97110"},
]

# ─── Clinical Tests ───
SPECIAL_TESTS = {
    "shoulder": [
        {"name": "Empty Can (Supraspinatus)", "positive_finding": "pain/weakness", "sensitivity": 0.68, "specificity": 0.74},
        {"name": "Neer Impingement", "positive_finding": "pain at 90-120°", "sensitivity": 0.72, "specificity": 0.60},
        {"name": "Hawkins-Kennedy", "positive_finding": "pain on internal rotation", "sensitivity": 0.74, "specificity": 0.57},
        {"name": "Drop Arm Test", "positive_finding": "unable to lower arm slowly", "sensitivity": 0.27, "specificity": 0.88},
        {"name": "Apprehension Test", "positive_finding": "fear of dislocation", "sensitivity": 0.53, "specificity": 0.86}
    ],
    "knee": [
        {"name": "McMurray's Test", "positive_finding": "click/pain on rotation", "sensitivity": 0.55, "specificity": 0.77},
        {"name": "Lachman Test", "positive_finding": "excessive anterior translation", "sensitivity": 0.85, "specificity": 0.94},
        {"name": "Patellar Grind", "positive_finding": "crepitus/pain", "sensitivity": 0.42, "specificity": 0.71}
    ],
    "lumbar": [
        {"name": "Straight Leg Raise", "positive_finding": "pain 30-70°", "sensitivity": 0.72, "specificity": 0.66},
        {"name": "Slump Test", "positive_finding": "neural tension signs", "sensitivity": 0.68, "specificity": 0.74}
    ],
    "hip": [
        {"name": "FADIR (Impingement)", "positive_finding": "groin pain", "sensitivity": 0.78, "specificity": 0.58},
        {"name": "FABER (Patrick's)", "positive_finding": "posterior pain", "sensitivity": 0.60, "specificity": 0.70}
    ],
    "cervical": [
        {"name": "Spurling's Test", "positive_finding": "radicular pain", "sensitivity": 0.50, "specificity": 0.86},
        {"name": "Sharp-Purser", "positive_finding": "clunk/subluxation", "sensitivity": 0.69, "specificity": 0.96}
    ]
}

def build_rag_context(joint_angles: dict, abnormalities: list) -> str:
    """Build RAG context string for AI analysis prompt."""
    lines = ["## MSK Clinical Knowledge Base\n"]

    # Add ROM norms for affected joints
    affected_joints = set()
    for abn in abnormalities:
        for joint_name in ROM_NORMS:
            if joint_name in abn.lower():
                affected_joints.add(joint_name)

    if affected_joints:
        lines.append("### ROM Normative Data (relevant joints):")
        for joint in sorted(affected_joints):
            norms = ROM_NORMS[joint]
            for movement, data in norms.items():
                lines.append(f"  {joint} {movement}: normal {data['min']}°-{data['max']}° (avg {data['avg']}°)")
        lines.append("")

    # Add relevant special tests
    lines.append("### Relevant Special Tests:")
    for joint in sorted(affected_joints):
        if joint in SPECIAL_TESTS:
            for test in SPECIAL_TESTS[joint]:
                lines.append(f"  {test['name']} (Sn={test['sensitivity']:.2f}, Sp={test['specificity']:.2f})")
    lines.append("")

    # Add exercise recommendations based on phase
    joint_exercises = []
    for e in EXERCISES:
        if e["joint"] in affected_joints:
            joint_exercises.append(e)

    if joint_exercises:
        lines.append("### Evidence-Based Exercise Prescriptions:")
        for phase in ["acute", "recovery", "strengthening"]:
            phase_ex = [e for e in joint_exercises if e["phase"] == phase]
            if phase_ex:
                lines.append(f"  {phase.upper()} Phase:")
                for e in phase_ex:
                    lines.append(f"    - {e['name']}: {e['sets']}x{e['reps']}, CPT {e['cpt']}")
        lines.append("")

    return "\n".join(lines)


def build_rag_prompt(joint_angles: dict, abnormalities: list) -> str:
    """Build a complete clinical analysis prompt with RAG context."""
    context = build_rag_context(joint_angles, abnormalities)

    prompt = f"""You are a board-certified physical medicine and rehabilitation specialist reviewing a real-time 3D movement assessment.

{context}

### Patient Assessment Data:
Joint angles detected: {json.dumps(joint_angles, indent=2)}
Abnormalities found: {json.dumps(abnormalities, indent=2)}

Provide your clinical analysis in this exact format:
SUMMARY: (2-3 sentence clinical summary based on joint angle measurements)
FINDINGS: (what the measurements indicate — which joints are impaired)
DIFFERENTIAL: (possible diagnoses based on movement patterns)
RECOMMENDATIONS: (specific exercises from the knowledge base above)
ICD10: (relevant ICD-10 codes for the findings)
CPT: (relevant CPT billing codes)
CONFIDENCE: (0.0-1.0 based on data completeness)
"""
    return prompt


if __name__ == "__main__":
    # Test
    test_angles = {"shoulder_flexion": 45, "shoulder_abduction": 30}
    test_abnormalities = ["Left shoulder flexion limited to 45° (normal 150-180)", "Left shoulder abduction limited"]
    print(build_rag_prompt(test_angles, test_abnormalities))
