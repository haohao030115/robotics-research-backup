"""Model-plan contract shared by VA rollout adapters."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PolicyPlan:
    body_tokens: np.ndarray
    hand_ref: np.ndarray
    start_tick: int
    phase_logits: np.ndarray | None = None
    closure_progress: float | None = None

    def validate(self) -> "PolicyPlan":
        if np.asarray(self.body_tokens).shape != (10, 64):
            raise ValueError("body_tokens must have shape [10,64]")
        if np.asarray(self.hand_ref).shape != (50, 14):
            raise ValueError("hand_ref must have shape [50,14]")
        if not np.isfinite(self.body_tokens).all() or not np.isfinite(self.hand_ref).all():
            raise ValueError("policy plan contains non-finite values")
        if int(self.start_tick) != self.start_tick or self.start_tick < 0:
            raise ValueError("start_tick must be a non-negative integer")
        return self
