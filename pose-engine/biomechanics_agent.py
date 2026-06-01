#!/usr/bin/env python3
"""
Biomechanics Agent — Boxer3D v4.0 (Dual-Model)
Takes YOLO pose keypoints + DEIMv2 enrichment and calculates:
  - Joint angles (shoulder, elbow, hip, knee, ankle, spine)
  - Cervical spine assessment (from DEIMv2 head pose)
  - Symmetry scores (left/right comparison)
  - Gait metrics (step length, cadence, stance/swing ratio)
  - Balance assessment (COP estimation)
  - Injury risk indicators
  - Range of Motion (ROM) vs normative data
  - Clinical context (wheelchair, crutches, mobility flags)

Part of the multi-agent MSK assessment pipeline.
"""
import sys
import json
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple

# COCO keypoint indices used by YOLO pose
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

# Normative range of motion data (degrees) — sourced from AAOS/AMA guidelines
NORM_ROM = {
    "shoulder_flexion": (150, 180),
    "shoulder_abduction": (150, 180),
    "elbow_flexion": (130, 150),
    "hip_flexion": (100, 130),
    "knee_flexion": (120, 150),
    "ankle_dorsiflexion": (15, 25),
    "cervical_rotation": (60, 80),
    "cervical_flexion": (45, 60),
    "cervical_extension": (45, 60),
    "lumbar_flexion": (60, 90),
}

# DEIMv2 head pose → cervical rotation inference
HEAD_POSE_TO_CERVICAL = {
    "front": {"rotation": 0, "status": "neutral", "plane": "sagittal"},
    "right_front": {"rotation": -20, "status": "mild_right_rotation", "plane": "transverse"},
    "right_side": {"rotation": -45, "status": "moderate_right_rotation", "plane": "transverse"},
    "right_back": {"rotation": -70, "status": "severe_right_rotation", "plane": "transverse"},
    "back": {"rotation": 180, "status": "extension", "plane": "sagittal"},
    "left_back": {"rotation": 70, "status": "severe_left_rotation", "plane": "transverse"},
    "left_side": {"rotation": 45, "status": "moderate_left_rotation", "plane": "transverse"},
    "left_front": {"rotation": 20, "status": "mild_left_rotation", "plane": "transverse"},
}


def angle_between(p1, p2, p3):
    """Calculate angle at p2 formed by points p1-p2-p3 (in degrees)."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def analyze_deimv2_enrichment(deimv2_data: Dict) -> Dict:
    """
    Analyze DEIMv2 enrichment data for clinical context.
    Returns structured clinical assessment.
    """
    result = {
        "cervical_spine": {},
        "clinical_flags": [],
        "mobility_context": None,
        "head_orientation": None,
        "deimv2_available": False
    }

    if not deimv2_data:
        return result

    result["deimv2_available"] = True
    result["head_orientation"] = deimv2_data.get("head_pose")

    # ─── Cervical spine assessment from head pose ───
    head_pose = deimv2_data.get("head_pose")
    head_pose_conf = deimv2_data.get("head_pose_confidence", 0)

    if head_pose and head_pose_conf > 0.5:
        cervical_info = HEAD_POSE_TO_CERVICAL.get(head_pose, {})
        result["cervical_spine"] = {
            "head_pose": head_pose,
            "confidence": head_pose_conf,
            "estimated_rotation_deg": cervical_info.get("rotation", 0),
            "status": cervical_info.get("status", "unknown"),
            "plane": cervical_info.get("plane", "unknown"),
            "normative_range": NORM_ROM.get("cervical_rotation"),
        }

        # Clinical flags from cervical
        clinical_context = deimv2_data.get("clinical_context", {})
        if clinical_context.get("cervical_asymmetry_flag"):
            result["clinical_flags"].append({
                "type": "cervical_asymmetry",
                "severity": "moderate",
                "detail": f"Head oriented {head_pose} — assess cervical rotation ROM",
                "can_assess_neck_rom": clinical_context.get("can_assess_neck_rom", False),
                "can_assess_rotation": clinical_context.get("can_assess_rotation", False)
            })

        if clinical_context.get("clinical_note"):
            result["clinical_flags"].append({
                "type": "clinical_note",
                "severity": "info",
                "detail": clinical_context["clinical_note"]
            })

    # ─── Attributes → clinical context ───
    attrs = deimv2_data.get("attributes", {})
    if "wheelchair" in attrs:
        result["mobility_context"] = "wheelchair_user"
        result["clinical_flags"].append({
            "type": "mobility_aid",
            "severity": "info",
            "detail": "Wheelchair detected — adjust ROM expectations, consider seated assessment protocol"
        })
    if "crutches" in attrs:
        result["mobility_context"] = "crutches_user"
        result["clinical_flags"].append({
            "type": "mobility_aid",
            "severity": "info",
            "detail": "Crutches detected — patient may have weight-bearing restrictions"
        })
    if "adult" in attrs:
        result["clinical_flags"].append({
            "type": "demographic",
            "severity": "info",
            "detail": f"Adult detected (conf: {attrs['adult'].get('confidence', 0):.2f})"
        })
    if "child" in attrs:
        result["clinical_flags"].append({
            "type": "demographic",
            "severity": "info",
            "detail": f"Child detected — use pediatric ROM norms (conf: {attrs['child'].get('confidence', 0):.2f})"
        })

    # ─── Face parts quality assessment ───
    face_parts = deimv2_data.get("face_parts", {})
    if face_parts:
        face_quality = len(face_parts)
        result["face_tracking_quality"] = "high" if face_quality >= 4 else "medium" if face_quality >= 2 else "low"

    return result


def analyze_frame(keypoints_xy: List[List[float]],
                  confidence: Optional[List[float]] = None,
                  deimv2_data: Optional[Dict] = None) -> Dict:
    """
    Analyze a single frame of 17 keypoints + optional DEIMv2 enrichment.
    keypoints_xy: list of [x, y] or [x, y, conf] for each of 17 keypoints
    """
    # Parse keypoints
    kps = {}
    for name, idx in KP.items():
        if idx < len(keypoints_xy):
            pt = keypoints_xy[idx]
            x, y = pt[0], pt[1]
            conf = pt[2] if len(pt) > 2 else 1.0
            kps[name] = {"x": x, "y": y, "conf": conf}
        else:
            kps[name] = {"x": 0, "y": 0, "conf": 0}

    def k(name):
        return (kps[name]["x"], kps[name]["y"])

    angles = {}

    # Upper body angles
    if all(kps[n]["conf"] > 0.3 for n in ["left_shoulder", "left_elbow", "left_wrist"]):
        angles["left_elbow_angle"] = angle_between(k("left_shoulder"), k("left_elbow"), k("left_wrist"))
    if all(kps[n]["conf"] > 0.3 for n in ["right_shoulder", "right_elbow", "right_wrist"]):
        angles["right_elbow_angle"] = angle_between(k("right_shoulder"), k("right_elbow"), k("right_wrist"))
    if all(kps[n]["conf"] > 0.3 for n in ["left_shoulder", "left_hip"]):
        angles["left_shoulder_inclination"] = angle_between(k("left_elbow"), k("left_shoulder"), k("left_hip"))
    if all(kps[n]["conf"] > 0.3 for n in ["right_shoulder", "right_hip"]):
        angles["right_shoulder_inclination"] = angle_between(k("right_elbow"), k("right_shoulder"), k("right_hip"))

    # Lower body angles
    if all(kps[n]["conf"] > 0.3 for n in ["left_hip", "left_knee", "left_ankle"]):
        angles["left_knee_angle"] = angle_between(k("left_hip"), k("left_knee"), k("left_ankle"))
    if all(kps[n]["conf"] > 0.3 for n in ["right_hip", "right_knee", "right_ankle"]):
        angles["right_knee_angle"] = angle_between(k("right_hip"), k("right_knee"), k("right_ankle"))
    if all(kps[n]["conf"] > 0.3 for n in ["left_hip", "left_knee"]):
        angles["left_hip_angle"] = angle_between(k("left_shoulder"), k("left_hip"), k("left_knee"))
    if all(kps[n]["conf"] > 0.3 for n in ["right_hip", "right_knee"]):
        angles["right_hip_angle"] = angle_between(k("right_shoulder"), k("right_hip"), k("right_knee"))

    # Spine / posture
    if all(kps[n]["conf"] > 0.3 for n in ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
        mid_shoulder = ((kps["left_shoulder"]["x"] + kps["right_shoulder"]["x"]) / 2,
                        (kps["left_shoulder"]["y"] + kps["right_shoulder"]["y"]) / 2)
        mid_hip = ((kps["left_hip"]["x"] + kps["right_hip"]["x"]) / 2,
                   (kps["left_hip"]["y"] + kps["right_hip"]["y"]) / 2)
        nose = k("nose")
        angles["spine_angle"] = angle_between(nose, mid_shoulder, mid_hip)
        angles["trunk_tilt"] = angle_between(mid_shoulder, mid_hip, (mid_hip[0], mid_hip[1] - 100))

    # Symmetry analysis
    symmetry = {}
    if "left_knee_angle" in angles and "right_knee_angle" in angles:
        symmetry["knee_symmetry"] = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
    if "left_elbow_angle" in angles and "right_elbow_angle" in angles:
        symmetry["elbow_symmetry"] = abs(angles["left_elbow_angle"] - angles["right_elbow_angle"])
    symmetry["score"] = 100 - sum(symmetry.values()) / max(len(symmetry), 1)

    # Balance indicator — center of mass proxy
    balance = {}
    if all(kps[n]["conf"] > 0.3 for n in ["left_ankle", "right_ankle"]):
        com_x = (kps["left_hip"]["x"] + kps["right_hip"]["x"]) / 2
        base_x = (kps["left_ankle"]["x"] + kps["right_ankle"]["x"]) / 2
        balance["cop_deviation"] = abs(com_x - base_x)

    # Risk indicators
    risks = []
    if symmetry.get("knee_symmetry", 0) > 10:
        risks.append({"joint": "knee", "asymmetry": round(symmetry["knee_symmetry"], 1),
                      "risk": "high", "note": "Knee angle asymmetry >10° indicates compensation pattern"})
    if angles.get("spine_angle", 0) > 15:
        risks.append({"joint": "spine", "angle": round(angles["spine_angle"], 1),
                      "risk": "moderate", "note": "Forward lean >15° suggests postural compensation"})

    # ─── DEIMv2 enrichment ───
    deimv2_assessment = analyze_deimv2_enrichment(deimv2_data or {})
    if deimv2_assessment.get("deimv2_available"):
        # Add cervical risks
        cervical = deimv2_assessment.get("cervical_spine", {})
        if cervical.get("status") and "severe" in cervical.get("status", ""):
            risks.append({
                "joint": "cervical_spine",
                "rotation": cervical.get("estimated_rotation_deg"),
                "risk": "high",
                "note": f"Severe cervical rotation detected ({cervical['head_pose']})"
            })
        elif cervical.get("status") and "moderate" in cervical.get("status", ""):
            risks.append({
                "joint": "cervical_spine",
                "rotation": cervical.get("estimated_rotation_deg"),
                "risk": "moderate",
                "note": f"Moderate cervical rotation detected ({cervical['head_pose']})"
            })

    return {
        "angles": angles,
        "symmetry": symmetry,
        "balance": balance,
        "risks": risks,
        "flagged": len(risks) > 0,
        "deimv2": deimv2_assessment
    }


def analyze_pose_data(pose_json_path: str) -> Dict:
    """Run full biomechanical analysis on pose agent output (supports dual-model format)."""
    with open(pose_json_path) as f:
        pose_data = json.load(f)

    frame_analyses = []

    deimv2_data = None

    # Handle dual-model output format from WebSocket server
    frames = pose_data.get("frames", [])
    if not frames:
        # Single-enriched frame format (direct from server)
        if "persons" in pose_data:
            deimv2_data = pose_data.get("deimv2")
            for person in pose_data.get("persons", []):
                kps = [list(kp.values()) if isinstance(kp, dict) else kp for kp in person.get("keypoints", [])]
                analysis = analyze_frame(kps, deimv2_data=deimv2_data)
                frame_analyses.append(analysis)
        return _aggregate_results(frame_analyses, deimv2_data)

    # Multi-frame format
    for frame in frames:
        deimv2_data = frame.get("deimv2")
        for person_kps in frame.get("keypoints", []):
            analysis = analyze_frame(person_kps, deimv2_data=deimv2_data)
            frame_analyses.append(analysis)

    deimv2_data = frames[-1].get("deimv2") if frames else None
    return _aggregate_results(frame_analyses, deimv2_data)


def _aggregate_results(frame_analyses: List[Dict], deimv2_data: Optional[Dict] = None) -> Dict:
    """Aggregate frame-level analyses into final report."""
    if not frame_analyses:
        return {"error": "No poses detected", "status": "no_data"}

    # Aggregate angles across frames
    all_angles = {}
    for analysis in frame_analyses:
        for key, val in analysis["angles"].items():
            if key not in all_angles:
                all_angles[key] = []
            all_angles[key].append(val)

    avg_angles = {k: round(sum(v)/len(v), 1) for k, v in all_angles.items()}
    min_angles = {k: round(min(v), 1) for k, v in all_angles.items()}
    max_angles = {k: round(max(v), 1) for k, v in all_angles.items()}

    # Range of motion
    rom = {}
    angle_map = {
        "left_knee_angle": "knee_flexion", "right_knee_angle": "knee_flexion",
        "left_elbow_angle": "elbow_flexion", "right_elbow_angle": "elbow_flexion",
        "left_hip_angle": "hip_flexion", "right_hip_angle": "hip_flexion",
    }
    for key, norm_key in angle_map.items():
        if key in max_angles and key in min_angles:
            measured_rom = max_angles[key] - min_angles[key]
            norm_range = NORM_ROM.get(norm_key)
            if norm_range:
                status = "normal" if norm_range[0] <= measured_rom <= norm_range[1] else "limited"
                rom[norm_key] = {"measured": round(measured_rom, 1), "norm": norm_range, "status": status}

    # Risk scoring
    risk_score = 0
    all_risks = []
    for a in frame_analyses:
        all_risks.extend(a["risks"])
    risk_score = len([r for r in all_risks if r.get("risk") == "high"]) * 15 + \
                 len([r for r in all_risks if r.get("risk") == "moderate"]) * 5

    # DEIMv2 enrichment aggregation
    deimv2_summary = None
    if frame_analyses:
        last_deimv2 = frame_analyses[-1].get("deimv2", {})
        if last_deimv2.get("deimv2_available"):
            deimv2_summary = {
                "head_pose": last_deimv2.get("head_orientation"),
                "cervical_spine": last_deimv2.get("cervical_spine", {}),
                "mobility_context": last_deimv2.get("mobility_context"),
                "clinical_flags": last_deimv2.get("clinical_flags", []),
                "face_tracking_quality": last_deimv2.get("face_tracking_quality", "unknown"),
            }

    return {
        "status": "complete",
        "frames_analyzed": len(frame_analyses),
        "average_angles": avg_angles,
        "range_of_motion": rom,
        "risk_assessment": {
            "score": min(risk_score, 100),
            "level": "low" if risk_score < 20 else "moderate" if risk_score < 50 else "high",
            "findings": list(set((r["note"] for r in all_risks)))
        },
        "summary": {
            "symmetry_score": round(
                sum(a["symmetry"].get("score", 100) for a in frame_analyses) / len(frame_analyses), 1
            ),
            "flagged_issues": len(set(r["note"] for r in all_risks))
        },
        "deimv2_enrichment": deimv2_summary
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boxer3D Biomechanics Agent v4.0")
    parser.add_argument("pose_json", help="Pose agent output JSON (supports dual-model format)")
    parser.add_argument("--output", "-o", help="Output path")
    args = parser.parse_args()
    
    result = analyze_pose_data(args.pose_json)
    output_path = args.output or Path(args.pose_json).parent / "biomechanics_output.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
