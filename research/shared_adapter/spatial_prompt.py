"""Deterministic spatial language used as an opt-in neural condition.

The default producer prompts remain unchanged.  This module only supplies a
single case-derived descriptor when a controlled generation explicitly opts
into ``spatial-conditioned-prompt-v1``.  The descriptor is semantic rather
than a hand-tuned coordinate offset; geometric task constraints remain the
authoritative position condition.
"""

from __future__ import annotations

from typing import Iterable, List


VERSION = "spatial-conditioned-prompt-v1"

_DESCRIPTORS = {
    "center": "The red cube is at the center of the tabletop.",
    "near": "The red cube is on the near side of the tabletop.",
    "far": "The red cube is on the far side of the tabletop.",
    "left": "The red cube is on the left side of the tabletop.",
    "right": "The red cube is on the right side of the tabletop.",
}


def descriptor(case_id: str) -> str:
    """Return the fixed semantic descriptor for a canonical case."""

    try:
        return _DESCRIPTORS[str(case_id)]
    except KeyError as exc:
        raise ValueError(f"unknown canonical spatial case: {case_id!r}") from exc


def condition_prompts(prompts: Iterable[str], case_id: str) -> List[str]:
    """Append one case-derived descriptor to each of five stage prompts."""

    values = [str(prompt).strip() for prompt in prompts]
    if len(values) != 5 or any(not value for value in values):
        raise ValueError("spatial conditioning requires five non-empty prompts")
    suffix = descriptor(case_id)
    return [f"{value} {suffix}" for value in values]

