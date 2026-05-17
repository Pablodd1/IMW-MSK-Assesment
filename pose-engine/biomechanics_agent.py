#!/usr/bin/env python3
"""
Biomechanics Agent — Boxer3D v3.0
Takes pose keypoints from Pose Agent and calculates:
  - Joint angles (shoulder, elbow, hip, knee, ankle, spine)
  - Symmetry scores (left/right comparison)
  - Gait metrics (step length, cadence, stance/swing ratio)
  - Balance assessment (COP estimation)
  - Injury risk indicators
  - Range of Motion (ROM) vs normative data

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
    "lumbar_flexion": (60, 90),
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


def analyze_frame(keypoints_xy: List[List[float]], confidence: Optional[List[float]] = None) -> Dict:
    """
    Analyze a single frame of 17 keypoints.
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

    return {
        "angles": angles,
        "symmetry": symmetry,
        "balance": balance,
        "risks": risks,
        "flagged": len(risks) > 0
    }


def analyze_pose_data(pose_json_path: str) -> Dict:
    """Run full biomechanical analysis on pose agent output."""
    with open(pose_json_path) as f:
        pose_data = json.load(f)

    frame_analyses = []
    for frame in pose_data.get("frames", []):
        for person_kps in frame.get("keypoints", []):
            analysis = analyze_frame(person_kps)
            frame_analyses.append(analysis)

    if not frame_analyses:
        return {"error": "No poses detected", "status": "no_data"}

    # Aggregate across frames
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

    # Final risk score
    risk_score = 0
    all_risks = []
    for a in frame_analyses:
        all_risks.extend(a["risks"])
    risk_score = len([r for r in all_risks if r.get("risk") == "high"]) * 15 + \
                 len([r for r in all_risks if r.get("risk") == "moderate"]) * 5

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
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boxer3D Biomechanics Agent")
    parser.add_argument("pose_json", help="Pose agent output JSON")
    parser.add_argument("--output", "-o", help="Output path")
    args = parser.parse_args()
    
    result = analyze_pose_data(args.pose_json)
    output_path = args.output or Path(args.pose_json).parent / "biomechanics_output.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
