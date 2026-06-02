"""
FMS Engine — Automated Functional Movement Screen scoring via pose landmarks.
Only Deep Squat and ASLR are reliable from single anterior camera.
Other tests require sagittal view or multi-camera.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import IntEnum


class FMSScore(IntEnum):
    ZERO = 0   # Pain
    ONE = 1    # Cannot perform
    TWO = 2    # Compensated perform
    THREE = 3  # Clean


@dataclass
class FMSResult:
    test_name: str
    score: FMSScore
    criteria_met: Dict[str, bool] = field(default_factory=dict)
    compensations: List[str] = field(default_factory=list)
    raw_measurements: Dict = field(default_factory=dict)
    notes: str = ""


class FMSEngine:
    """
    Automated FMS scoring using 3D pose landmarks (normalized 0-1, z in meters).
    """

    def __init__(self, depth_available: bool = False):
        self.depth = depth_available

    @staticmethod
    def _angle(a, b, c) -> float:
        """Angle at b formed by a-b-c (degrees)."""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        norm = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norm == 0:
            return 0.0
        cos = np.dot(ba, bc) / norm
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    @staticmethod
    def _verticality(top, bot) -> float:
        """Deviation from vertical in degrees (0 = perfectly vertical)."""
        dx = top[0] - bot[0]
        dy = top[1] - bot[1]
        if dy == 0:
            return 90.0
        return float(np.degrees(np.arctan2(abs(dx), abs(dy))))

    def score_deep_squat(self, kpts: Dict[int, Dict]) -> FMSResult:
        """
        Criteria for 3/3:
          1. Femur below horizontal (hip y > knee y in normalized coords)
          2. Torso parallel with tibia (within 15 degrees)
          3. No knee valgus (knees track over toes)
          4. Heels remain down (best with depth or side view)
        """
        ls, rs = kpts.get(5), kpts.get(6)
        lh, rh = kpts.get(11), kpts.get(12)
        lk, rk = kpts.get(13), kpts.get(14)
        la, ra = kpts.get(15), kpts.get(16)

        if not all([ls, rs, lh, rh, lk, rk, la, ra]):
            return FMSResult("deep_squat", FMSScore.ONE,
                             compensations=["Missing landmarks"],
                             notes="Insufficient visibility for scoring")

        # 1. Squat depth: hip below knee (y increases downward in normalized coords)
        hip_y = (lh["y"] + rh["y"]) / 2
        knee_y = (lk["y"] + rk["y"]) / 2
        femur_below_horizontal = hip_y > knee_y

        # 2. Torso-tibia parallel
        mid_shoulder = ((ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2)
        mid_hip = ((lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2)
        torso_v = self._verticality(mid_shoulder, mid_hip)

        mid_knee = ((lk["x"] + rk["x"]) / 2, (lk["y"] + rk["y"]) / 2)
        mid_ankle = ((la["x"] + ra["x"]) / 2, (la["y"] + ra["y"]) / 2)
        tibia_v = self._verticality(mid_knee, mid_ankle)

        torso_tibia_parallel = abs(torso_v - tibia_v) < 15.0

        # 3. Knee valgus: knee x inside ankle x
        knee_valgus = (lk["x"] < la["x"]) or (rk["x"] > ra["x"])

        # 4. Heel lift — impossible from anterior camera alone; flag as unknown
        heel_lift = False
        heel_note = "Heel position requires side view or depth floor plane"

        criteria = {
            "femur_below_horizontal": femur_below_horizontal,
            "torso_tibia_parallel": torso_tibia_parallel,
            "no_knee_valgus": not knee_valgus,
            "heels_down": not heel_lift  # always true from anterior view
        }

        compensations = []
        if not femur_below_horizontal:
            compensations.append("Limited squat depth")
        if not torso_tibia_parallel:
            compensations.append("Torso lean or tibia angle deviation")
        if knee_valgus:
            compensations.append("Knee valgus detected")

        if all(criteria.values()):
            score = FMSScore.THREE
        elif femur_below_horizontal:
            score = FMSScore.TWO
        else:
            score = FMSScore.ONE

        return FMSResult(
            test_name="deep_squat",
            score=score,
            criteria_met=criteria,
            compensations=compensations,
            raw_measurements={
                "torso_angle_deg": round(torso_v, 1),
                "tibia_angle_deg": round(tibia_v, 1),
                "hip_y": round(hip_y, 4),
                "knee_y": round(knee_y, 4),
            },
            notes=heel_note
        )

    def score_aslr(self, kpts: Dict[int, Dict], side: str = "left") -> FMSResult:
        """
        Active Straight-Leg Raise.
        Criteria:
          - Raised leg stays straight (knee angle > 160 degrees)
          - Ankle crosses mid-thigh of down leg
        """
        hip_idx = 11 if side == "left" else 12
        knee_idx = 13 if side == "left" else 14
        ankle_idx = 15 if side == "left" else 16
        opp_hip_idx = 12 if side == "left" else 11

        hip = kpts.get(hip_idx)
        knee = kpts.get(knee_idx)
        ankle = kpts.get(ankle_idx)
        opp_hip = kpts.get(opp_hip_idx)

        if not all([hip, knee, ankle, opp_hip]):
            return FMSResult(f"aslr_{side}", FMSScore.ONE,
                             compensations=["Missing landmarks"])

        # Knee angle: hip-knee-ankle
        knee_angle = self._angle(
            (hip["x"], hip["y"]),
            (knee["x"], knee["y"]),
            (ankle["x"], ankle["y"])
        )
        leg_straight = knee_angle > 160.0

        # Ankle above opposite hip (higher in frame = smaller y)
        ankle_above_opp_hip = ankle["y"] < opp_hip["y"]

        criteria = {
            "leg_straight": leg_straight,
            "ankle_above_mid_thigh": ankle_above_opp_hip,
        }

        if all(criteria.values()):
            score = FMSScore.THREE
        elif leg_straight:
            score = FMSScore.TWO
        else:
            score = FMSScore.ONE

        return FMSResult(
            test_name=f"aslr_{side}",
            score=score,
            criteria_met=criteria,
            compensations=[] if score == FMSScore.THREE else ["Compensatory movement detected"],
            raw_measurements={"knee_angle_deg": round(knee_angle, 1)},
            notes="Anterior view; external rotation not assessable"
        )

    def score_shoulder_mobility(self, kpts: Dict[int, Dict]) -> FMSResult:
        """
        Requires hand keypoints (DWPose 133-keypoint).
        Without hands: fallback to wrist proximity behind back.
        """
        lw = kpts.get(9)
        rw = kpts.get(10)
        if not lw or not rw:
            return FMSResult("shoulder_mobility", FMSScore.ONE,
                             compensations=["Missing wrist landmarks"],
                             notes="DWPose hand keypoints required for precise scoring")
        # Simplified: fists within 15cm (normalized ~0.08) behind back
        dist = np.linalg.norm(np.array([lw["x"], lw["y"]]) - np.array([rw["x"], rw["y"]]))
        score = FMSScore.THREE if dist < 0.08 else FMSScore.TWO if dist < 0.15 else FMSScore.ONE
        return FMSResult(
            test_name="shoulder_mobility",
            score=score,
            raw_measurements={"inter_wrist_dist": round(dist, 4)},
            notes="Wrist-only fallback; use DWPose hand keypoints for true scoring"
        )

    def run_battery(self, kpts: Dict[int, Dict]) -> Dict[str, FMSResult]:
        """Run all automatable tests from anterior view."""
        return {
            "deep_squat": self.score_deep_squat(kpts),
            "aslr_left": self.score_aslr(kpts, "left"),
            "aslr_right": self.score_aslr(kpts, "right"),
            "shoulder_mobility": self.score_shoulder_mobility(kpts),
        }
