#!/usr/bin/env python3
"""Smoke test: move the simulated block once and verify all goals follow it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .block_pose_keyframes import build_constraints, extract_demo_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--ardy-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--demo-scene", type=Path, required=True)
    parser.add_argument("--demo-simulation", type=Path, required=True)
    parser.add_argument("--demo-result", type=Path, required=True)
    parser.add_argument("--lateral-shift", type=float, default=0.04)
    args = parser.parse_args()

    template = extract_demo_template(args.demo_scene, args.demo_simulation, args.demo_result)
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    address = model.joint("wood_block_free").qposadr[0]
    original_qpos = data.qpos[address:address + 7].copy()
    shifted_qpos = original_qpos.copy()
    shifted_qpos[1] += args.lateral_shift

    original = build_constraints(
        args.scene, args.ardy_repo, args.output_root / "original", template,
        initial_block_qpos=original_qpos,
    )
    # The block is moved exactly once before the task snapshot.  The planner is
    # not told a target; it independently reads the new body pose from MuJoCo.
    shifted = build_constraints(
        args.scene, args.ardy_repo, args.output_root / "shifted", template,
        initial_block_qpos=shifted_qpos,
    )
    block_delta = np.asarray(shifted["block_pose_read_once"]["position"]) - np.asarray(
        original["block_pose_read_once"]["position"]
    )
    goal_delta = np.asarray(shifted["stage_positions_world"]) - np.asarray(
        original["stage_positions_world"]
    )
    expected = np.array([0.0, args.lateral_shift, 0.0])
    max_error = float(np.max(np.abs(goal_delta - expected)))
    report = {
        "pass": bool(np.allclose(block_delta, expected, atol=1e-6) and max_error < 1e-6),
        "lateral_shift_m": args.lateral_shift,
        "block_pose_delta_m": block_delta.tolist(),
        "stage_goal_deltas_m": goal_delta.tolist(),
        "max_goal_translation_error_m": max_error,
        "original_block_was_modified_after_snapshot": original["block_was_modified_after_snapshot"],
        "shifted_block_was_modified_after_snapshot": shifted["block_was_modified_after_snapshot"],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
