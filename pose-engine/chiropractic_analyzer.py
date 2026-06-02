"""
Chiropractic Analyzer — Postural analysis, leg length discrepancy,
pelvic tilt, and forward head posture for chiropractic assessment.
"""
import numpy as np
from typing import Dict, List, Optional


class ChiropracticAnalyzer:
    """
    Postural analysis engine for chiropractic clinical decision support.
    Uses COCO body keypoints (0-16) from pose estimation.
    """

    def __init__(self):
        self.reference_frames: List[Dict[int, Dict]] = []
        self.baseline = None

    def calibrate_from_t_pose(self, kpts: Dict[int, Dict]):
        """Capture patient's neutral T-pose as postural baseline."""
        self.reference_frames.append(kpts)
        if len(self.reference_frames) >= 3:
            self._compute_baseline()

    def _compute_baseline(self):
        """Average reference frames to establish neutral posture."""
        # Simplified: could use Procrustes alignment here
        self.baseline = self.reference_frames[-1]

    def analyze_plumb_line(self, kpts: Dict[int, Dict]) -> Dict:
        """
        Compares patient posture to vertical plumb line.
        Returns lateral deviations at ear, shoulder, hip, knee, ankle.
        """
        # Left-side landmarks (ear=3, shoulder=5, hip=11, knee=13, ankle=15)
        indices = [3, 5, 11, 13, 15]
        names = ["ear", "shoulder", "hip", "knee", "ankle"]
        points = [kpts.get(i) for i in indices]

        if not all(points):
            return {"error": "Insufficient landmarks for plumb line"}

        ankle_x = points[-1]["x"]
        deviations = []
        for p, name in zip(points, names):
            dev_cm = (p["x"] - ankle_x) * 200  # approximate cm for ~2m subject
            deviations.append({"level": name, "deviation_cm": round(dev_cm, 1)})

        # Forward head posture: ear anterior to shoulder
        fhp_cm = (points[0]["x"] - points[1]["x"]) * 200

        # Sway pattern
        shoulder_dev = deviations[1]["deviation_cm"]
        hip_dev = deviations[2]["deviation_cm"]
        pattern = "C-shaped" if shoulder_dev * hip_dev > 0 else "S-shaped"

        return {
            "plumb_line_deviations": deviations,
            "forward_head_posture_cm": round(fhp_cm, 1),
            "lateral_sway_pattern": pattern,
        }

    def leg_length_discrepancy(self, kpts: Dict[int, Dict]) -> Dict:
        """
        Estimates apparent leg length from iliac crest to ankle.
        """
        lh, rh = kpts.get(11), kpts.get(12)
        la, ra = kpts.get(15), kpts.get(16)

        if not all([lh, rh, la, ra]):
            return {"error": "Missing hip or ankle landmarks"}

        left_len = abs(la["y"] - lh["y"])
        right_len = abs(ra["y"] - rh["y"])
        disc_norm = abs(left_len - right_len)
        disc_cm = disc_norm * 90  # assume ~90cm leg

        sig = (
            "significant" if disc_cm > 0.5
            else "mild" if disc_cm > 0.2
            else "none"
        )

        return {
            "left_leg_apparent": round(left_len, 4),
            "right_leg_apparent": round(right_len, 4),
            "discrepancy_normalized": round(disc_norm, 4),
            "discrepancy_cm": round(disc_cm, 1),
            "clinical_significance": sig,
        }

    def pelvic_tilt(self, kpts: Dict[int, Dict]) -> Dict:
        """
        Anterior / posterior pelvic tilt from hip-hip line angle.
        """
        lh, rh = kpts.get(11), kpts.get(12)
        if not lh or not rh:
            return {"error": "Missing hip landmarks"}

        angle = float(np.degrees(np.arctan2(rh["y"] - lh["y"], rh["x"] - lh["x"])))
        # Normalize: standing neutral ~0 deg; positive = anterior tilt visual
        interpretation = (
            "anterior_tilt" if angle > 5
            else "posterior_tilt" if angle < -5
            else "neutral"
        )

        return {
            "pelvic_obliquity_deg": round(angle, 1),
            "interpretation": interpretation,
        }

    def full_postural_screen(self, kpts: Dict[int, Dict]) -> Dict:
        """Run all chiropractic screens and return unified report."""
        return {
            "plumb_line": self.analyze_plumb_line(kpts),
            "leg_length": self.leg_length_discrepancy(kpts),
            "pelvic_tilt": self.pelvic_tilt(kpts),
        }
