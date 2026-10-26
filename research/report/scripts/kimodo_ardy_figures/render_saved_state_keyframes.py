#!/usr/bin/env python3
"""Render verified keyframes from recorded MuJoCo states without advancing physics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_keyframes(rows: list[dict]) -> dict[str, dict]:
    playing = [row for row in rows if bool(row["playing"])]
    by_frame = {int(row["frame"]): row for row in playing}
    ordered = [by_frame[key] for key in sorted(by_frame) if key >= 0]
    if not ordered:
        raise ValueError("no playback rows")

    contacts = [
        row
        for row in ordered
        if float(row["thumb_force_n"]) > 1e-4
        and float(row["opposing_finger_force_n"]) > 1e-4
    ]
    if not contacts:
        raise ValueError("no grasp contact")
    grasp = contacts[0]
    approach_target = min(450, int(grasp["frame"]) - 75)
    approach = min(ordered, key=lambda row: abs(int(row["frame"]) - approach_target))

    lifted = [
        row
        for row in ordered
        if int(row["frame"]) > int(grasp["frame"])
        and float(row["lift_m"]) >= 0.10
    ]
    if not lifted:
        raise ValueError("no 0.10 m lift crossing")
    lift = lifted[0]

    qualified = [row for row in ordered if bool(row["qualified"])]
    if not qualified:
        raise ValueError("no qualified hold")
    hold = qualified[-1]

    selected = {"approach": approach, "grasp": grasp, "lift": lift, "hold": hold}
    if len({int(row["frame"]) for row in selected.values()}) != 4:
        raise ValueError("events are not distinct")
    return selected


def _strict_contact_counts(result: dict) -> tuple[int, int]:
    report = result.get("original_shared_evaluator_report", {})
    return (
        int(report.get("robot_table_contact_steps", result.get("robot_table_contact_steps", 0))),
        int(report.get("robot_self_contact_steps", result.get("robot_self_contact_steps", 0))),
    )


def _validate_run(run: Path, provenance: dict, result: dict) -> None:
    if not result.get("physical_success") or not result.get("expert_valid"):
        raise ValueError("representative run must be physical_success=true and expert_valid=true")
    table_steps, self_steps = _strict_contact_counts(result)
    if table_steps or self_steps:
        raise ValueError(f"representative run has contact violations: table={table_steps}, self={self_steps}")
    for key in ("scene", "robot"):
        path = Path(provenance[key])
        if digest(path) != provenance[key + "_sha256"]:
            raise RuntimeError(f"{key} hash changed since recording")
    builder = Path(provenance["builder"])
    recorded = provenance.get("audit_hashes", {}).get(str(builder))
    if recorded and digest(builder) != recorded:
        raise RuntimeError("scene builder hash changed since recording")
    for name in ("simulation.npz", "physics_steps.json"):
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)


def render_run(run: Path, output: Path, producer: str, camera_spec: dict) -> None:
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    physics = json.loads((run / "physics_steps.json").read_text(encoding="utf-8"))
    _validate_run(run, provenance, result)
    selected = select_keyframes(physics)

    os.environ.setdefault(
        "SONIC_REPO",
        str(Path(os.environ.get("RESEARCH_ROOT", Path(__file__).resolve().parents[3])) / "GR00T-WholeBodyControl"),
    )
    sys.path.insert(0, str(Path(provenance["builder"]).parent))
    import project_paths

    project_paths.default_robot_xml = lambda: Path(provenance["robot"])
    import mujoco
    import numpy as np
    from PIL import Image
    from scene import load_scene

    model, data = load_scene(
        supported=False,
        robot_path=provenance["robot"],
        scene_path=provenance["scene"],
    )
    recorded = np.load(run / "simulation.npz", allow_pickle=False)
    qpos, qvel = recorded["qpos"], recorded["qvel"]
    reference_frames = recorded["reference_frame"].astype(int)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or qvel.shape != (len(qpos), model.nv):
        raise ValueError("recorded state/model shape mismatch")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("non-finite recorded state")

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = camera_spec["lookat"]
    camera.distance = float(camera_spec["distance"])
    camera.azimuth = float(camera_spec["azimuth"])
    camera.elevation = float(camera_spec["elevation"])

    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.geomgroup[0] = 0
    output.mkdir(parents=True)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for label, row in selected.items():
            frame = int(row["frame"])
            matches = np.flatnonzero(reference_frames == frame)
            if len(matches) != 1:
                raise ValueError(f"reference frame {frame} has {len(matches)} recorded states")
            index = int(matches[0])
            data.qpos[:] = qpos[index]
            data.qvel[:] = qvel[index]
            data.time = float(recorded["sim_time"][index])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            Image.fromarray(renderer.render()).save(output / f"{label}.png")

    by_frame = {int(row["frame"]): row for row in physics if bool(row["playing"]) and int(row["frame"]) >= 0}
    trace_rows = [by_frame[key] for key in sorted(by_frame)]
    fields = [
        "sim_time", "frame", "lift_m", "thumb_force_n", "opposing_finger_force_n",
        "cube_table_contact", "qualified", "hold_seconds",
    ]
    t0 = float(trace_rows[0]["sim_time"])
    with (output / "physical_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in trace_rows:
            item = {field: row[field] for field in fields}
            item["sim_time"] = float(item["sim_time"]) - t0
            writer.writerow(item)

    table_steps, self_steps = _strict_contact_counts(result)
    manifest = {
        "producer": producer,
        "scope": "representative_saved_state_replay",
        "run": str(run),
        "source": str(run / "simulation.npz"),
        "source_sha256": digest(run / "simulation.npz"),
        "physics_sha256": digest(run / "physics_steps.json"),
        "result_sha256": digest(run / "result.json"),
        "scene_sha256": digest(Path(provenance["scene"])),
        "robot_sha256": digest(Path(provenance["robot"])),
        "camera": camera_spec,
        "events": {
            label: {"frame": int(row["frame"]), "time_s": float(row["sim_time"]) - t0}
            for label, row in selected.items()
        },
        "result": {
            "physical_success": bool(result["physical_success"]),
            "expert_valid": bool(result["expert_valid"]),
            "max_lift_m": float(result["max_lift_m"]),
            "max_hold_s": float(result["max_continuous_hold_seconds"]),
            "robot_table_contact_steps": table_steps,
            "robot_self_contact_steps": self_steps,
        },
        "mujoco_version": mujoco.__version__,
        "physics_reexecuted": False,
        "grasp_validation_performed": False,
    }
    (output / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--producer", choices=("kimodo", "ardy"), required=True)
    parser.add_argument("--camera-json", required=True)
    args = parser.parse_args()
    render_run(
        args.run_dir.resolve(),
        args.output_dir.resolve(),
        args.producer,
        json.loads(args.camera_json),
    )


if __name__ == "__main__":
    main()
