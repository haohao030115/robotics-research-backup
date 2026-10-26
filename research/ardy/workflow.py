"""Command construction for the existing ARDY workflow."""

from __future__ import annotations

from pathlib import Path
import json


def build_commands(
    *,
    project_root: str | Path,
    config: str | Path,
    output: str | Path,
    seed: int,
    case: str,
    gpu: int,
    adapter: str = "legacy",
    hand_profile: str = "ardy",
    preserve_generated_body: bool = False,
    preserve_body_weight: float = 1.0,
    body_preservation_policy: str = "legacy",
    opposition_centering: bool = False,
    collision_aware_hand: bool = False,
    collision_hand_min_clearance_m: float = 0.001,
    collision_hand_max_relief_rad: float = 0.30,
    eef_lift_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    eef_lift_offset_window_s: tuple[float, float] = (10.0, 11.0),
) -> dict[str, list[str]]:
    root = Path(project_root).resolve()
    output = Path(output).resolve()
    code = root / "research/ardy/qualification"
    # The validated ARDY deployment reserves MUSA devices 4-7. Keep the
    # historical CLI default (0) usable by mapping it to the first ARDY
    # device; explicit 4-7 selections remain unchanged.
    ardy_gpu = gpu if gpu in (4, 5, 6, 7) else 4
    pipeline = [
        # Keep the historical executable spelling so manifests remain
        # comparable across the host and the va-train container.
        "python3",
        "-B",
        str(code / "run_pipeline.py"),
        "--out",
        str(output),
        "--config",
        str(Path(config).resolve()),
        "--seed",
        str(seed),
        "--case",
        case,
        "--gpu",
        str(ardy_gpu),
    ]
    if adapter not in {"legacy", "unified"}:
        raise ValueError("adapter must be legacy or unified")
    if body_preservation_policy not in {"legacy", "bounded"}:
        raise ValueError("body_preservation_policy must be legacy or bounded")
    if not 0 <= preserve_body_weight <= 1:
        raise ValueError("preserve_body_weight must be in [0,1]")
    if len(eef_lift_offset_m) != 3 or any(not isinstance(x, (int, float)) for x in eef_lift_offset_m):
        raise ValueError("eef_lift_offset_m must contain three numeric values")
    if sum(float(x) ** 2 for x in eef_lift_offset_m) ** 0.5 > 0.10:
        raise ValueError("eef_lift_offset_m norm must be <= 0.10 m")
    if len(eef_lift_offset_window_s) != 2 or not (0 <= float(eef_lift_offset_window_s[0]) < float(eef_lift_offset_window_s[1])):
        raise ValueError("eef_lift_offset_window_s must satisfy 0 <= start < end")
    pipeline.extend(["--adapter", adapter, "--hand-profile", hand_profile])
    if preserve_generated_body:
        pipeline.append("--preserve-generated-body")
    pipeline.extend(["--preserve-body-weight", str(preserve_body_weight),
                     "--body-preservation-policy", body_preservation_policy,
                     "--eef-lift-offset-m", *map(str, eef_lift_offset_m),
                     "--eef-lift-offset-window-s", *map(str, eef_lift_offset_window_s)])
    if opposition_centering:
        pipeline.append("--opposition-centering")
    if collision_aware_hand:
        if collision_hand_min_clearance_m < 0 or not 0 < collision_hand_max_relief_rad <= 0.5:
            raise ValueError("invalid collision hand projection bounds")
        pipeline.extend(["--collision-aware-hand",
                         "--collision-hand-min-clearance-m", str(collision_hand_min_clearance_m),
                         "--collision-hand-max-relief-rad", str(collision_hand_max_relief_rad)])
    # Formal center runs use the measured seed-0 tracking calibration for all
    # controlled seeds.  Keep it in the JSON config so the adapter records the
    # evidence hash and rejects mismatched case/source files.
    try:
        cfg = json.loads(Path(config).read_text())
        calibration = cfg.get("adapter", {}).get("calibration")
    except (OSError, json.JSONDecodeError):
        calibration = None
    if calibration:
        pipeline.extend(["--calibration", str(Path(calibration).resolve())])
    # ``run_pipeline.py`` owns the ordered generation -> adaptation -> replay
    # -> verification transaction.  Exposing its individual internal stages
    # here caused the runner to execute a generate-only pass and then attempt
    # to reuse the same output directory.  Keep one authoritative stage and
    # let the pipeline record its own sub-stage provenance.
    # Keep the historical key for callers that only inspect command paths;
    # the orchestration layer selects the single authoritative pipeline key.
    return {"pipeline": pipeline, "full_pipeline": pipeline}
