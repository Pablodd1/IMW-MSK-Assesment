#!/usr/bin/env python3
"""
Boxer3D Multi-Agent Orchestrator
Coordinates: Pose Agent → Biomechanics Agent → Report Agent
"""
import sys
import json
import os
import subprocess
from pathlib import Path

POSE_AGENT = os.path.join(os.path.dirname(__file__), "pose_agent.py")
BIOMECH_AGENT = os.path.join(os.path.dirname(__file__), "biomechanics_agent.py")
REPORT_AGENT = os.path.join(os.path.dirname(__file__), "report_agent.py")


def run_agent(script, *args):
    """Run a Python agent script and return its output."""
    result = subprocess.run(
        [sys.executable, script] + list(args),
        capture_output=True, text=True, cwd=os.path.dirname(script)
    )
    if result.returncode != 0:
        print(f"Agent {script} failed: {result.stderr}", file=sys.stderr)
        return None
    return result.stdout


def orchestrate(input_path, output_dir=".", patient_name="Patient", model="yolo11n-pose"):
    """Run the full Boxer3D pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Boxer3D Pipeline Starting")
    print(f"   Input: {input_path}")
    print(f"   Model: {model}")
    print(f"   Patient: {patient_name}")
    print()
    
    # Step 1: Pose Agent
    print("1️⃣  Pose Agent — detecting keypoints...")
    step1_log = run_agent(POSE_AGENT, input_path, "-o", str(output_dir), "-m", model)
    if step1_log is None:
        print("❌ Pose Agent failed")
        return False
    print(f"   ✅ Pose detection complete → {output_dir}/pose_output.json")
    print()
    
    # Step 2: Biomechanics Agent
    print("2️⃣  Biomechanics Agent — analyzing movement...")
    step2_log = run_agent(BIOMECH_AGENT, str(output_dir / "pose_output.json"), "-o", str(output_dir / "biomechanics_output.json"))
    if step2_log is None:
        print("❌ Biomechanics Agent failed")
        return False
    print(f"   ✅ Biomechanics analysis complete → {output_dir}/biomechanics_output.json")
    print()
    
    # Step 3: Report Agent
    print("3️⃣  Report Agent — generating clinical summary...")
    step3_log = run_agent(REPORT_AGENT, str(output_dir / "biomechanics_output.json"),
                          "-o", str(output_dir / "report_output.json"), "-p", patient_name)
    if step3_log is None:
        print("❌ Report Agent failed")
        return False
    print(f"   ✅ Report generated → {output_dir}/report_output.json")
    print(f"   ✅ Patient summary → {output_dir}/patient_summary.txt")
    print()
    
    print("✅ BOXER3D PIPELINE COMPLETE")
    print(f"   All outputs in: {output_dir}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boxer3D Multi-Agent Orchestrator")
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--patient", "-p", default="Patient", help="Patient name")
    parser.add_argument("--model", "-m", default="yolo11n-pose",
                        choices=["yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
                                 "yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose"],
                        help="YOLO model for pose estimation (n=nano to x=extra large)")
    args = parser.parse_args()
    orchestrate(args.input, args.output_dir, args.patient, args.model)
