"""Convert an image-space Qwen plan into a bounded ARDY condition."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .visual_trace import CropTransform, trace_endpoint_to_world, world_point_to_uv


DIRECTION_PROMPTS = {
    "forward": "A person stands balanced and slowly extends their right hand slightly forward.",
    "backward": "A person stands balanced and slowly draws their right hand slightly backward.",
    "left": "A person stands balanced and slowly moves their right hand slightly to the left.",
    "right": "A person stands balanced and slowly moves their right hand slightly to the right.",
    "up": "A person stands balanced and slowly raises their right hand.",
    "down": "A person stands balanced and slowly lowers their right hand.",
    "hold": "A person stands balanced and keeps their raised right hand steady.",
    "lift": "A person stands balanced and slowly raises their right hand while keeping it steady.",
}


def ardy_direction_prompt(delta_world: np.ndarray, next_subtask: str) -> tuple[str, str]:
    """Map geometry to a body-motion phrase that requires no visual grounding."""
    if next_subtask == "lift":
        return "lift", DIRECTION_PROMPTS["lift"]
    if next_subtask == "close_gripper":
        return "hold", DIRECTION_PROMPTS["hold"]
    delta = np.asarray(delta_world, dtype=float)
    axis = int(np.argmax(np.abs(delta)))
    if float(np.linalg.norm(delta)) < .008:
        direction = "hold"
    elif axis == 2:
        direction = "up" if delta[2] > 0 else "down"
    elif axis == 1:
        direction = "left" if delta[1] > 0 else "right"
    else:
        direction = "forward" if delta[0] > 0 else "backward"
    return direction, DIRECTION_PROMPTS[direction]


def build_ardy_condition(model: Any, data: Any,
                         decision: dict, current_hand_world: np.ndarray,
                         target_plane_z_m: float, camera: str = "blocks_overview",
                         image_size: tuple[int, int] = (640, 480),
                         max_step_m: float = .06,
                         crop: CropTransform | None = None) -> dict:
    trace = decision["trace"]
    end_uv = trace["end_uv"]
    if trace.get("coordinate_space") == "crop_normalized":
        if crop is None:
            raise ValueError("crop metadata is required for crop-normalized Visual Trace")
        end_uv = crop.crop_uv_to_full_uv(end_uv)
    start_uv = world_point_to_uv(model, data, camera, current_hand_world, image_size)
    world = trace_endpoint_to_world(
        model, data, camera, end_uv, image_size, target_plane_z_m, start_uv)
    requested = np.asarray(world.endpoint_world_m)
    current = np.asarray(current_hand_world, dtype=float)
    delta = requested - current
    length = float(np.linalg.norm(delta))
    if length > max_step_m:
        requested = current + delta * (max_step_m / length)
    direction, prompt = ardy_direction_prompt(requested-current, decision["next_subtask"])
    # Numeric values are conditions for ARDY, never interpolated into its text.
    return {
        "prompt": prompt,
        "ardy_direction": direction,
        "vlm_action_text": str(decision.get("ardy_prompt", "")),
        "next_subtask": str(decision["next_subtask"]),
        "completed_subtasks": list(decision["completed_subtasks"]),
        "remaining_subtasks": list(decision["remaining_subtasks"]),
        "constraint": {
            "type": "visual_trace_endpoint_world",
            "endpoint_world_m": requested.round(6).tolist(),
            "source_trace": asdict(world),
            "crop": crop.to_dict() if crop else None,
            "max_cartesian_step_m": max_step_m,
        },
    }
