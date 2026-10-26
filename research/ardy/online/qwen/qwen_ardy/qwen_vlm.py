from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from .config import QWEN_BASE_URL, QWEN_MODEL
from .models import EpisodeSummary, PlannerDecision


SYSTEM_PROMPT = """You are the high-level visual planner for a simulated Unitree G1 grasp.
You receive exact MuJoCo world-frame telemetry and optional overview images. Select one safe
semantic strategy and a SMALL Cartesian correction for the next ARDY keyframe constraint.
Positive x/y/z are MuJoCo world axes. closest_delta_world is block minus grasp-center, so a
reasonable correction usually follows that vector. constraint_delta_world_m is the ABSOLUTE
offset from the block for the next grasp-center constraint, not an accumulated motion command.
Never output joints, torques, code, or a
numeric natural-language motion prompt. The numeric correction is applied only as a keyframe
constraint. Hand closure is contact-driven and MUST NOT be scheduled by time. Success requires
final lift > 0.08 m and two opposing finger contact groups.
Return one JSON object only with: assessment, strategy, semantic_action,
constraint_delta_world_m (three numbers, each within [-0.08,0.08]), confidence."""


def _image_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


class QwenPlanner:
    def __init__(self, api_key: str | None = None, model: str = QWEN_MODEL):
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured")
        self.client = OpenAI(api_key=key, base_url=QWEN_BASE_URL, timeout=90.0)
        self.model = model

    def decide(self, summary: EpisodeSummary, images: Iterable[Path] = ()) -> PlannerDecision:
        content: list[dict] = [{
            "type": "text",
            "text": "Assess this completed attempt and plan the next one. Telemetry:\n" +
                    json.dumps(summary.to_dict(), ensure_ascii=False),
        }]
        for path in list(images)[:3]:
            if path.exists() and path.stat().st_size:
                content.append({"type": "image_url", "image_url": {"url": _image_url(path)}})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = response.choices[0].message.content or "{}"
        return PlannerDecision.model_validate_json(raw)


def heuristic_decision(summary: EpisodeSummary) -> PlannerDecision:
    if summary.success:
        return PlannerDecision(
            assessment="Physical success criteria passed.", strategy="success",
            semantic_action="lift_and_hold", constraint_delta_world_m=[0, 0, 0], confidence=1,
        )
    delta = summary.closest_delta_world
    labels = ["move_closer", "move_left", "move_higher"]
    axis = max(range(3), key=lambda i: abs(delta[i]))
    strategy = labels[axis]
    if axis == 0 and delta[0] < 0: strategy = "move_farther"
    if axis == 1: strategy = "move_left" if delta[1] > 0 else "move_right"
    if axis == 2: strategy = "move_higher" if delta[2] > 0 else "move_lower"
    correction = [max(-0.05, min(0.05, float(v))) for v in delta]
    return PlannerDecision(
        assessment="Geometry fallback follows the measured block-minus-grasp-center error.",
        strategy=strategy, semantic_action="careful_reach",
        constraint_delta_world_m=correction, confidence=0.65,
    )
