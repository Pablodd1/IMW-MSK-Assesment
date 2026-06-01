#!/usr/bin/env python3
"""
Boxer3D Multi-Agent Orchestrator v4.0 — Dual-Model
Coordinates: DEIMv2 Agent → Pose Agent → Biomechanics Agent → Report Agent

New v4.0 pipeline:
  1. DEIMv2 Agent — head pose, person attributes, face parts, mobility context
  2. Pose Agent — 17-keypoint skeleton via YOLO11-pose
  3. Biomechanics Agent — joint angles + cervical assessment (enriched)
  4. Report Agent — structured clinical summary with dual-model context
"""
import sys
import json
import os
import subprocess
from pathlib import Path

POSE_AGENT = os.path.join(os.path.dirname(__file__), "pose_agent.py")
DEIMV2_AGENT = os.path.join(os.path.dirname(__file__), "deimv2_agent.py")
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


def orchestrate(input_path, output_dir=".", patient_name="Patient", model="yolo11n-pose",
                deimv2_model=None, enable_deimv2=False):
    """Run the full Boxer3D pipeline with optional DEIMv2 enrichment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*55}")
    print(f"  Boxer3D Pipeline v4.0 — Dual-Model")
    print(f"  Input: {input_path}")
    print(f"  YOLO Model: {model}")
    print(f"  DEIMv2: {'✅ Enabled' if enable_deimv2 else '❌ Disabled'}")
    print(f"  Patient: {patient_name}")
    print(f"{'='*55}\n")
    
    # Step 0: DEIMv2 Agent (if enabled)
    deimv2_output = None
    if enable_deimv2 and deimv2_model:
        print("0️⃣  DEIMv2 Agent — head pose + attributes...")
        deimv2_cmd = [
            sys.executable, DEIMV2_AGENT, input_path,
            "--model", deimv2_model
        ]
        result = subprocess.run(deimv2_cmd, capture_output=True, text=True,
                                cwd=os.path.dirname(DEIMV2_AGENT))
        if result.returncode == 0:
            print("   ✅ DEIMv2 analysis complete")
            deimv2_output = result.stdout
        else:
            print(f"   ⚠️  DEIMv2 failed (continuing without): {result.stderr[:200]}")
        print()
    
    # Step 1: Pose Agent
    print("1️⃣  Pose Agent — detecting keypoints...")
    step1_log = run_agent(POSE_AGENT, input_path, "-o", str(output_dir), "-m", model)
    if step1_log is None:
        print("❌ Pose Agent failed")
        return False
    print(f"   ✅ Pose detection complete → {output_dir}/pose_output.json")
    print()
    
    # Step 2: Biomechanics Agent (now accepts DEIMv2-enriched data)
    print("2️⃣  Biomechanics Agent — analyzing movement...")
    step2_log = run_agent(BIOMECH_AGENT, str(output_dir / "pose_output.json"),
                          "-o", str(output_dir / "biomechanics_output.json"))
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
    
    print(f"{'='*55}")
    print(f"  ✅ BOXER3D PIPELINE COMPLETE")
    print(f"  Models used: {model}" + (" + deimv2-wholebody34" if enable_deimv2 else ""))
    print(f"  All outputs in: {output_dir}")
    print(f"{'='*55}\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boxer3D Multi-Agent Orchestrator v4.0")
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--patient", "-p", default="Patient", help="Patient name")
    parser.add_argument("--model", "-m", default="yolo11n-pose",
                        choices=["yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose"],
                        help="YOLO model for pose estimation")
    parser.add_argument("--deimv2", action="store_true", help="Enable DEIMv2 enrichment")
    parser.add_argument("--deimv2-model", default="/home/jasme/IMW-MSK-Assesment/models/deimv2_wholebody34_680q.onnx",
                        help="DEIMv2 ONNX model path")
    parser.add_argument("--dual", action="store_true", help="Shortcut: enable dual-model mode")
    args = parser.parse_args()
    
    enable_deimv2 = args.deimv2 or args.dual
    
    orchestrate(
        args.input,
        args.output_dir,
        args.patient,
        args.model,
        deimv2_model=args.deimv2_model,
        enable_deimv2=enable_deimv2
    )
