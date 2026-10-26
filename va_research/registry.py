"""Explicit registry for existing VA families and their entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class ModelFamily:
    name: str
    implementation: str
    input_contract: str
    output_contract: str
    training_entry: str
    evaluation_entry: str
    rollout_entry: str | None


FAMILIES = {
    # V2/V3 remain historical compatibility families. Their validated scripts
    # are retained in va_training until their dependency graph is migrated as
    # a unit; no current experiments command dispatches them.
    "v2": ModelFamily("v2", "va_research.models.v2.token_v2", "token-v2 dataset", "token + hand chunk", "va_research/models/v2/token_v2.py", "va_training/scripts/evaluate_token_v2.py", "va_training/scripts/run_token_closed_loop.py"),
    "v3": ModelFamily("v3", "va_research.models.v3.model", "RGB + 96D state", "2 body tokens + 2 hand refs", "va_research/models/v3/training.py", "va_training/scripts/evaluate_token_v3.py", "va_training/scripts/run_token_v3_closed_loop.py"),
    "v4": ModelFamily("v4", "va_research.models.v4.model", "RGB 128 + body68 + hand28", "body 10x64 + hand 50x14 + phase/progress", "va_research.models.v4.training", "va_research.models.v4.evaluation", "va_research.models.v4.rollout"),
    "v4_1": ModelFamily("v4_1", "va_research.models.v4_1.model", "V4-compatible", "V4 + readiness", "va_research.models.v4_1.train_mixed", "va_research.models.v4.evaluation", "va_research.models.v4_1.oracle_rollout"),
    "bcz_hand": ModelFamily("bcz_hand", "va_research.models.v4_0_bcz_hand.model", "V4-compatible", "V4 + closure intent", "va_research.models.v4_0_bcz_hand.train_full", "va_research.models.v4.evaluation", "va_research.models.v4.one_shot_rollout"),
    "oracle": ModelFamily("oracle", "va_research.models.v4_oracle", "V4-compatible", "privileged hand diagnostic", "diagnostic only", "diagnostic only", "va_research.models.v4_oracle.rollout"),
}


def get_family(name: str) -> ModelFamily:
    try:
        return FAMILIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown VA family: {name}; choose {sorted(FAMILIES)}") from exc
