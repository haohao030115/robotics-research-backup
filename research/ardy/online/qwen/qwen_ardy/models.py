from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator


@dataclass
class EpisodeSummary:
    attempt: int
    success: bool
    block_world_initial: list[float]
    block_world_final: list[float]
    closest_grasp_center_world: list[float]
    closest_delta_world: list[float]
    minimum_hand_block_distance_m: float
    block_lift_max_m: float
    block_lift_final_m: float
    final_contacts: int
    minimum_root_height_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlannerDecision(BaseModel):
    assessment: str = Field(max_length=500)
    strategy: Literal[
        "retry", "move_closer", "move_farther", "move_left", "move_right",
        "move_higher", "move_lower", "stabilize", "success",
    ]
    semantic_action: Literal[
        "approach_and_grasp", "careful_reach", "stabilize_then_reach", "lift_and_hold"
    ] = "careful_reach"
    constraint_delta_world_m: list[float] = Field(min_length=3, max_length=3)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("constraint_delta_world_m")
    @classmethod
    def finite_and_small(cls, value: list[float]) -> list[float]:
        a = np.asarray(value, dtype=float)
        if not np.isfinite(a).all():
            raise ValueError("constraint delta must be finite")
        return np.clip(a, -0.08, 0.08).round(5).tolist()


SEMANTIC_PROMPTS = {
    "approach_and_grasp": (
        "A person starts standing with both arms relaxed at their sides, takes a small "
        "step forward, reaches with the right hand to grasp a block on a table, holds it "
        "firmly, and slowly lifts the block while the left arm remains relaxed at their side."
    ),
    "careful_reach": (
        "A person starts standing with both arms relaxed at their sides, carefully reaches "
        "forward with the right hand, securely grasps a small block on a table, and slowly "
        "lifts and holds it while keeping the body balanced."
    ),
    "stabilize_then_reach": (
        "A person stands steadily with both arms relaxed, balances the body, then slowly "
        "reaches with the right hand to grasp a small block and gently lifts it."
    ),
    "lift_and_hold": (
        "A person firmly holds a small block in the right hand, slowly lifts it from the "
        "table, and keeps it steady while maintaining balance."
    ),
}


def semantic_prompt(decision: PlannerDecision) -> str:
    """Return a vetted ARDY-style phrase; numeric state never enters the prompt."""
    return SEMANTIC_PROMPTS[decision.semantic_action]
