#!/usr/bin/env python3
"""One-shot block-pose → ARDY constraints → SONIC reference pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .block_pose_keyframes import PROMPT, build_constraints, extract_demo_template


OPEN = np.array([0.0, -0.1, -0.1, 0.15, 0.15, 0.15, 0.15])
CLOSED = np.array([0.0, -0.8, -1.2, 1.2, 1.3, 1.2, 1.3])
HAND_NAMES = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]


def write_hands(output_dir: Path, frames: int, fps: float = 50.0) -> None:
    times = np.arange(frames) / fps
    amount = np.clip((times - 6.0) / 1.0, 0, 1)
    amount = amount * amount * (3 - 2 * amount)
    right = OPEN[None] * (1 - amount[:, None]) + CLOSED[None] * amount[:, None]
    left = np.tile(OPEN, (frames, 1))
    for side, values in (("right", right), ("left", left)):
        header = ",".join(f"{side}_{name}" for name in HAND_NAMES)
        np.savetxt(output_dir / f"{side}_hand_pos.csv", values, delimiter=",", header=header, comments="")
    (output_dir / "hand_plan.json").write_text(json.dumps({
        "source": "fuyuhan successful demo schedule", "fps": fps, "frames": frames,
        "close_start_s": 6.0, "close_end_s": 7.0,
        "right_open": OPEN.tolist(), "right_closed": CLOSED.tolist(),
    }, indent=2) + "\n")


def reference_quality(output_dir: Path, scene: Path, ardy_repo: Path, plan: dict) -> dict:
    qpos = np.loadtxt(output_dir / "ardy_motion.csv", delimiter=",", ndmin=2)
    names = [
        joint.get("name")
        for joint in ET.parse(ardy_repo / "ardy/assets/skeletons/g1skel34/xml/g1.xml").findall(".//worldbody//joint")
        if joint.get("type") != "free"
    ]
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    joint_q = np.array([model.joint(name).qposadr[0] for name in names])
    wrist = model.body("right_wrist_yaw_link").id
    offset = np.asarray(plan["grasp_offset"])
    centers = []
    for pose in qpos:
        data.qpos[:7] = pose[:7]
        data.qpos[joint_q] = pose[7:]
        mujoco.mj_kinematics(model, data)
        rotation = data.xmat[wrist].reshape(3, 3)
        centers.append(data.xpos[wrist] + rotation @ offset)
    centers = np.asarray(centers)
    frames = np.asarray(plan["dense_frames"], dtype=int)
    goals = np.asarray(plan["dense_world_goals"])
    errors = np.linalg.norm(centers[frames] - goals, axis=1)
    quality = {
        "source": "ARDY generated body; successful-demo block-local keyframes",
        "reach_error_mean_m": float(errors.mean()),
        "reach_error_max_m": float(errors.max()),
        "reach_errors_m": errors.tolist(),
    }
    (output_dir / "reference_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ardy-repo", type=Path, required=True)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--demo-scene", type=Path, required=True)
    parser.add_argument("--demo-simulation", type=Path, required=True)
    parser.add_argument("--demo-result", type=Path, required=True)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template = extract_demo_template(args.demo_scene, args.demo_simulation, args.demo_result)
    plan = build_constraints(args.scene, args.ardy_repo, args.output_dir, template)
    stem = args.output_dir / "ardy_motion"
    if not args.skip_generate:
        environment = os.environ.copy()
        environment.update({
            "MUSA_VISIBLE_DEVICES": args.gpu, "CUDA_VISIBLE_DEVICES": args.gpu,
            "TEXT_ENCODERS_DIR": str(args.ardy_repo / "text_encoders"),
            "TEXT_ENCODER_MODE": "local", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
        subprocess.run([
            sys.executable, str(args.ardy_repo / "scripts/generate.py"), PROMPT,
            "--model", "g1", "--checkpoints_dir", str(args.ardy_repo / "checkpoints"),
            "--duration", "12", "--seed", "0", "--constraints",
            str(args.output_dir / "constraints.json"), "--output", str(stem),
        ], check=True, cwd=args.ardy_repo, env=environment)

    converter = args.sonic_root / "gear_sonic_deploy/reference/convert_ardy.py"
    subprocess.run([
        sys.executable, str(converter), str(stem.with_suffix(".npz")),
        "--ardy-repo", str(args.ardy_repo), "--output-dir", str(args.output_dir),
    ], check=True, cwd=args.sonic_root)
    frame_count = len(np.loadtxt(args.output_dir / "joint_pos.csv", delimiter=",", skiprows=1, ndmin=2))
    write_hands(args.output_dir, frame_count)
    quality = reference_quality(args.output_dir, args.scene, args.ardy_repo, plan)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "block_pose": plan["block_pose_read_once"],
        "prompt": PROMPT, "reference_frames": frame_count,
        "quality": {key: value for key, value in quality.items() if key != "reach_errors_m"},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
