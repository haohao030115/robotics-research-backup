"""Command construction for the existing Kimodo workflow.

Command construction is pure and testable. Execution remains opt-in through
the orchestration layer so native/MUSA work cannot start accidentally.
"""

from __future__ import annotations

from pathlib import Path
import sys


def build_commands(
    *,
    project_root: str | Path,
    prompt: str,
    output: str | Path,
    seed: int | None = None,
    case: str | None = None,
    adapter: str = "legacy",
    hand_profile: str = "default",
    preserve_generated_body: bool = False,
    preserve_body_weight: float = 1.0,
    body_preservation_policy: str = "legacy",
    opposition_centering: bool = False,
    collision_aware_hand: bool = False,
    collision_hand_min_clearance_m: float = 0.001,
    collision_hand_max_relief_rad: float = 0.30,
) -> dict[str, list[str]]:
    root = Path(project_root).resolve()
    output = Path(output).resolve()
    task = [
        sys.executable,
        "-B",
        str(root / "research/kimodo/generation/prepare_grasp_task.py"),
        "--out",
        str(output / "task"),
        "--case",
        case or "center",
        "--seed",
        str(seed or 0),
    ]
    generation = [
        sys.executable,
        "-B",
        str(root / "research/kimodo/generation/generate_grasp_motion.py"),
        "--task",
        str(output / "task"),
        "--out",
        str(output / "generation"),
        # Formal Kimodo runs used physical MUSA device 1.  Keep this explicit
        # so seeded generation is comparable with the validated batch.
        "--gpu",
        "1",
        "--segmented",
    ]
    if adapter not in {"legacy", "unified"}:
        raise ValueError("adapter must be legacy or unified")
    if body_preservation_policy not in {"legacy", "bounded"}:
        raise ValueError("body_preservation_policy must be legacy or bounded")
    if not 0 <= preserve_body_weight <= 1:
        raise ValueError("preserve_body_weight must be in [0,1]")
    formal_config = root / "workspace/kunqi/task9_formal40_20261008/kimodo" / "trials" / f"seed_{seed or 0}" / "config" / "frozen_config.json"
    formal_calibration = formal_config.with_name("development_calibration.json")
    if not formal_config.is_file():
        formal_config = root / "workspace/kunqi/task9_formal40_20261008/kimodo_frozen_method_source/config/frozen_config.json"
        formal_calibration = formal_config.with_name("development_calibration.json")
    export = [
        sys.executable,
        "-B",
        str(root / "research/kimodo/export/export_kimodo_reference.py"),
        "--task",
        str(output / "task"),
        "--generation",
        str(output / "generation"),
        "--out",
        str(output / "reference"),
        "--adapt",
        "neutral-world-wrist",
        "--clearance",
        "0.03",
        "--contact-refine",
        "--approach-refine",
        "--adapter",
        adapter,
        "--hand-profile",
        hand_profile,
    ]
    if preserve_generated_body:
        export.append("--preserve-generated-body")
    export += ["--preserve-body-weight", str(preserve_body_weight),
               "--body-preservation-policy", body_preservation_policy]
    if opposition_centering:
        export.append("--opposition-centering")
    if collision_aware_hand:
        if collision_hand_min_clearance_m < 0 or not 0 < collision_hand_max_relief_rad <= 0.5:
            raise ValueError("invalid collision hand projection bounds")
        export.extend(["--collision-aware-hand",
                       "--collision-hand-min-clearance-m", str(collision_hand_min_clearance_m),
                       "--collision-hand-max-relief-rad", str(collision_hand_max_relief_rad)])
    # The recorded calibration is explicitly tied to the center/seed-1 task;
    # never silently reuse it for spatial evaluation positions.
    if case in (None, "center") and formal_calibration.is_file() and (seed or 0) == 1:
        export[export.index("--adapter"):export.index("--adapter")] = ["--tracking-calibration", str(formal_calibration)]
    qualification = [
        sys.executable,
        "-B",
        str(root / "research/kimodo/qualification/replay_validate_kimodo.py"),
        "--candidate",
        str(output / "reference"),
        "--out",
        str(output / "qualification"),
        "--run",
        "--gpu",
        "1",
    ]
    return {"task": task, "generation": generation, "export": export, "qualification": qualification}
