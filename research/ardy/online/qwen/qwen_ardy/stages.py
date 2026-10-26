from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    START = "start"
    RAISE_HAND = "raise_hand"
    LOWER_TO_BLOCK = "lower_to_block"
    GRASP = "grasp"
    LIFT_WITH_BLOCK = "lift_with_block"
    DONE = "done"


STAGE_PROMPTS = {
    Stage.START: "A person stands still in a balanced neutral pose with both arms relaxed at their sides.",
    Stage.RAISE_HAND: "A person remains balanced and slowly raises the right hand in preparation to reach forward.",
    Stage.LOWER_TO_BLOCK: "A person carefully moves the open right hand forward and downward toward a small block on a table.",
    Stage.GRASP: "A person keeps the right wrist steady around a small block while maintaining a balanced stance.",
    Stage.LIFT_WITH_BLOCK: "A person securely holds a small block in the right hand and slowly lifts it while keeping the body balanced.",
}


@dataclass(frozen=True)
class LiveState:
    distance_m: float
    vertical_error_m: float
    root_height_m: float
    root_tilt_deg: float
    thumb_contact: bool
    opposing_contact: bool
    block_lift_m: float
    stable_contact_s: float = 0.0


@dataclass(frozen=True)
class GateResult:
    ready: bool
    reason: str


def gate(stage: Stage, state: LiveState) -> GateResult:
    if state.root_height_m < .68 or state.root_tilt_deg > 15:
        return GateResult(False, "robot_unstable")
    if stage == Stage.START:
        return GateResult(True, "stable_start")
    if stage == Stage.RAISE_HAND:
        return GateResult(state.distance_m < .30, "hand_raised" if state.distance_m < .30 else "continue_raise")
    if stage == Stage.LOWER_TO_BLOCK:
        ready = state.distance_m < .035 and abs(state.vertical_error_m) < .025
        return GateResult(ready, "pregrasp_aligned" if ready else "continue_alignment")
    if stage == Stage.GRASP:
        ready = state.thumb_contact and state.opposing_contact and state.stable_contact_s >= .25
        return GateResult(ready, "opposing_contact_stable" if ready else "continue_closing")
    if stage == Stage.LIFT_WITH_BLOCK:
        ready = state.block_lift_m > .08 and state.thumb_contact and state.opposing_contact
        return GateResult(ready, "grasp_lift_success" if ready else "continue_lift")
    return GateResult(True, "done")


def next_stage(stage: Stage) -> Stage:
    order = [Stage.START, Stage.RAISE_HAND, Stage.LOWER_TO_BLOCK,
             Stage.GRASP, Stage.LIFT_WITH_BLOCK, Stage.DONE]
    return order[min(order.index(stage) + 1, len(order) - 1)]

