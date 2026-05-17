#!/usr/bin/env python3
"""
Pose Agent — Boxer3D v3.0
Runs YOLO26 pose estimation on video/images and extracts 3D keypoints.
Part of the multi-agent MSK assessment pipeline.
"""
import sys
import json
import numpy as np
from pathlib import Path

def run_pose_estimation(input_path, output_dir=".", conf_thresh=0.5, model_type="yolo11n-pose"):
    """
    Run YOLO pose estimation.
    Returns structured keypoint data for downstream agents.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_type)
    except ImportError:
        print(json.dumps({"error": "ultralytics not installed", "resolution": "pip install ultralytics"}))
        sys.exit(1)

    results = model(input_path, conf=conf_thresh, save=False, verbose=False)
    
    frames_data = []
    for r in results:
        if r.keypoints is None:
            continue
        frame = {
            "frame": r.boxes is not None and len(r.boxes) > 0,
            "num_people": len(r.keypoints.data) if r.keypoints.data is not None else 0,
            "keypoints": []
        }
        if r.keypoints.data is not None:
            for person_kps in r.keypoints.data:
                kp_list = person_kps.cpu().numpy().tolist()
                frame["keypoints"].append(kp_list)
        frames_data.append(frame)
    
    # Save output
    out = {
        "model": model_type,
        "conf_threshold": conf_thresh,
        "total_frames": len(results),
        "frames": frames_data,
        "summary": {
            "max_people_detected": max((f["num_people"] for f in frames_data), default=0),
            "total_detections": sum(f["num_people"] for f in frames_data)
        }
    }
    output_path = Path(output_dir) / "pose_output.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    
    print(json.dumps({"status": "complete", "output": str(output_path), "summary": out["summary"]}))
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boxer3D Pose Agent")
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--model", "-m", default="yolo11n-pose", help="YOLO model name (use yolo11n-pose for speed, yolo11x-pose for accuracy)")
    args = parser.parse_args()
    run_pose_estimation(args.input, args.output_dir, args.conf, args.model)
