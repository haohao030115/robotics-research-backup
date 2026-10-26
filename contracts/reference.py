"""Canonical reference boundary used by producer adapters and execution.

This mirrors the existing canonical schema without importing the existing
runtime package. Validation is intentionally strict and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


class ReferenceValidationError(ValueError):
    """Raised when a producer artifact cannot cross the canonical boundary."""


@dataclass(frozen=True)
class CanonicalReference:
    time_s: np.ndarray
    body_ref_q: np.ndarray
    body_ref_dq: np.ndarray
    root_ref_pos: np.ndarray
    root_ref_quat: np.ndarray
    hand_ref_q: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    hand_ref_dq: np.ndarray | None = None

    def validate(self) -> "CanonicalReference":
        arrays = {
            "time_s": self.time_s,
            "body_ref_q": self.body_ref_q,
            "body_ref_dq": self.body_ref_dq,
            "root_ref_pos": self.root_ref_pos,
            "root_ref_quat": self.root_ref_quat,
            "hand_ref_q": self.hand_ref_q,
        }
        if self.hand_ref_dq is not None:
            arrays["hand_ref_dq"] = self.hand_ref_dq
        n = len(self.time_s)
        expected = {
            "time_s": (n,),
            "body_ref_q": (n, 29),
            "body_ref_dq": (n, 29),
            "root_ref_pos": (n, 3),
            "root_ref_quat": (n, 4),
            "hand_ref_q": (n, 14),
        }
        if self.hand_ref_dq is not None:
            expected["hand_ref_dq"] = (n, 14)
        if n < 7:
            raise ReferenceValidationError("at least seven reference frames are required")
        for name, value in arrays.items():
            value = np.asarray(value)
            if value.shape != expected[name] or value.dtype.kind not in "fiu":
                raise ReferenceValidationError(f"{name}: expected {expected[name]}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ReferenceValidationError(f"{name}: non-finite value")
        if not np.all(np.diff(self.time_s) > 0):
            raise ReferenceValidationError("time_s must strictly increase")
        hz = self.metadata.get("reference_hz")
        if hz != 50 or not np.isclose(self.time_s[0], 0.0) or not np.allclose(
            np.diff(self.time_s), 1 / 50, rtol=0, atol=1e-7
        ):
            raise ReferenceValidationError("reference_hz/time_s must describe 50 Hz starting at zero")
        if self.metadata.get("quaternion_convention", self.metadata.get("root_quat_order")) != "wxyz":
            raise ReferenceValidationError("root quaternion convention must be wxyz")
        if self.metadata.get("root_pose_frame", "world") != "world":
            raise ReferenceValidationError("root_pose_frame must be world")
        if not np.allclose(np.linalg.norm(self.root_ref_quat, axis=1), 1, atol=1e-5, rtol=0):
            raise ReferenceValidationError("root quaternions must be unit length")
        if not np.allclose(self.hand_ref_q[0], 0, atol=1e-10, rtol=0):
            raise ReferenceValidationError("initial hand reference must be zero")
        if self.metadata.get("producer_type") not in {"kimodo", "ardy", "rule_based"}:
            raise ReferenceValidationError("producer_type must be kimodo, ardy or rule_based")
        if self.metadata.get("task_arm", "right") not in {"left", "right"}:
            raise ReferenceValidationError("task_arm must be left or right")
        return self


def load_reference(folder: str | Path) -> CanonicalReference:
    """Load a producer's canonical artifact without running physics."""

    folder = Path(folder)
    with np.load(folder / "reference.npz", allow_pickle=False) as values:
        arrays = {name: values[name] for name in values.files}
    metadata = json.loads((folder / "metadata.json").read_text())
    return CanonicalReference(metadata=metadata, **{name: arrays[name] for name in (
        "time_s", "body_ref_q", "body_ref_dq", "root_ref_pos", "root_ref_quat", "hand_ref_q"
    )}, hand_ref_dq=arrays.get("hand_ref_dq")).validate()
