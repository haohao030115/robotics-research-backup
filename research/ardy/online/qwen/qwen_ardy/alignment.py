from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentGate:
    aligned: bool
    direction: str
    distance_m: float
    reason: str


def check_pregrasp(delta_world_m, has_hand_contact: bool) -> AlignmentGate:
    """Hard safety gate; VLM may recommend but cannot override this result."""
    delta = np.asarray(delta_world_m, dtype=float)
    distance = float(np.linalg.norm(delta))
    # Calibrated against the Dex3 grasp-center proxy.  The proxy lies inside the
    # palm, so successful capture has a nonzero center distance.
    precise_free_space = distance <= .035 and abs(delta[2]) <= .025
    calibrated_contact_pose = distance <= .075 and abs(delta[2]) <= .045 and has_hand_contact
    aligned = precise_free_space or calibrated_contact_pose
    if aligned:
        return AlignmentGate(True, "none", distance, "inside_calibrated_capture_envelope")
    axis = int(np.argmax(np.abs(delta)))
    if axis == 0: direction = "forward" if delta[0] > 0 else "backward"
    elif axis == 1: direction = "left" if delta[1] > 0 else "right"
    else: direction = "up" if delta[2] > 0 else "down"
    return AlignmentGate(False, direction, distance, "geometry_or_contact_gate_failed")
