#!/usr/bin/env python3
"""One-shot Qwen routing followed by the fixed fuyuhan grasp sequence."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

from .local_qwen import LocalQwenPlanner, MODEL_PATH


STAGES = [
    ("start", 0, 75,
     "A person starts standing with both arms relaxed at their sides."),
    ("approach", 75, 125,
     "A person takes a small step forward and reaches with the right hand."),
    ("align", 125, 150,
     "A person carefully lowers and aligns the open right hand for grasping."),
    ("close_and_hold", 150, 200,
     "A person holds the right wrist steady and firmly closes the right hand."),
    ("lift", 200, 300,
     "A person securely holds the object and slowly lifts the right hand."),
]
GRASP_OFFSET = np.array([0.125, 0.045, 0.0])


def wrist_manifest(source: Path, scene: Path) -> list[dict]:
    """Read stage wrist targets from the successful ARDY reference itself."""
    qpos = np.loadtxt(source / "ardy_motion.csv", delimiter=",", ndmin=2)
    tree = ET.parse(Path(os.environ.get(
        "ARDY_REPO", str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "workspace/fuyuhan/ardy"))) /
        "ardy/assets/skeletons/g1skel34/xml/g1.xml")
    names = [node.get("name") for node in tree.findall(".//worldbody//joint")
             if node.get("type") != "free"]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qadr = np.array([model.joint(name).qposadr[0] for name in names])
    wrist = model.body("right_wrist_yaw_link").id
    manifest = []
    for name, begin, end, prompt in STAGES:
        frame = min(end - 1, len(qpos) - 1)
        data.qpos[:7] = qpos[frame, :7]
        data.qpos[qadr] = qpos[frame, 7:]
        mujoco.mj_forward(model, data)
        rotation = data.xmat[wrist].reshape(3, 3).copy()
        center = data.xpos[wrist] + rotation @ GRASP_OFFSET
        manifest.append({
            "stage": name, "frames_25hz": [begin, end], "fixed_prompt": prompt,
            "wrist_position_world_m": data.xpos[wrist].round(8).tolist(),
            "grasp_center_world_m": center.round(8).tolist(),
            "wrist_rotation_world": rotation.round(8).tolist(),
            "source": "fuyuhan successful ARDY demo",
        })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-prompt", default="Please grasp the wooden block and lift it.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--skip-qwen", action="store_true",
                        help="Route directly to grasp_block; useful for repeated simulation tests.")
    parser.add_argument("--route-only", action="store_true",
                        help="Run startup routing, write task_route.json, then stop.")
    parser.add_argument("--qwen-model", type=Path, default=MODEL_PATH)
    parser.add_argument("--qwen-device", default="auto")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--robot-x", type=float, default=.24)
    parser.add_argument("--robot-y", type=float, default=-.03)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.scene = args.scene.resolve()
    args.sonic_root = args.sonic_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started = time.monotonic()
    if args.skip_qwen:
        route = {"task": "grasp_block", "confidence": 1.0,
                 "reason": "Qwen bypassed for repeatable simulation test",
                 "model": None}
    else:
        planner = LocalQwenPlanner(args.qwen_model, args.qwen_device)
        route = planner.route_task(args.task_prompt, args.image)
        # Qwen is a startup router only.  Release its weights before launching
        # the real-time SONIC controller; keeping the VLM resident measurably
        # degrades control timing and grasp accuracy on this shared MUSA GPU.
        del planner
        gc.collect()
        if hasattr(torch, "musa") and hasattr(torch.musa, "empty_cache"):
            torch.musa.empty_cache()
    (args.output_dir / "task_route.json").write_text(
        json.dumps({"instruction": args.task_prompt, **route}, indent=2,
                   ensure_ascii=False) + "\n")
    if args.route_only:
        print(json.dumps(route, indent=2, ensure_ascii=False))
        return
    if route["task"] != "grasp_block":
        raise SystemExit("Qwen route is unsupported; no robot motion was started.")

    manifest = wrist_manifest(args.source, args.scene)
    (args.output_dir / "fixed_stage_plan.json").write_text(
        json.dumps({"route": route, "stages": manifest,
                    "qwen_calls_during_motion": 0,
                    "continuous_reference": True}, indent=2,
                   ensure_ascii=False) + "\n")

    command = [
        sys.executable,
        str(args.sonic_root / "gear_sonic_deploy/reference/test_ardy_grasp.py"),
        "--arms-at-sides", "--robot-x", str(args.robot_x),
        "--robot-y", str(args.robot_y), "--hand-kp", "6",
        "--motion-dir", str(args.source.parent),
        "--scene", str(args.scene),
        "--output-dir", str(args.output_dir / "simulation"),
    ]
    if args.viewer:
        command.append("--viewer")
    subprocess.run(command, check=True, cwd=args.sonic_root, env=os.environ.copy())
    result_path = args.output_dir / "simulation/result.json"
    result = json.loads(result_path.read_text())
    summary = {"task": route["task"], "stage_count": len(manifest),
               "qwen_calls_during_motion": 0,
               "elapsed_s": time.monotonic() - started, **result}
    (args.output_dir / "result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
