#!/usr/bin/env python3
"""JSON-lines worker so MuJoCo can keep its LowState heartbeat during VLM inference."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .local_qwen import LocalQwenPlanner
from .visual_trace import CropTransform, crop_for_vlm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="musa:7")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    planner = LocalQwenPlanner(args.model, args.device)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    if not args.no_warmup:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="blocks_overview")
        with tempfile.TemporaryDirectory() as directory:
            warmup = Path(directory) / "warmup.png"
            Image.fromarray(renderer.render()).resize((384, 384)).save(warmup)
            planner.decide_visual_trace(
                warmup, {"contacts": [], "holding": False, "lifted": False})
    print(json.dumps({"event": "ready", "model": args.model.name}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            data.qpos[:] = np.asarray(request["qpos"], dtype=float)
            if "qvel" in request:
                data.qvel[:] = np.asarray(request["qvel"], dtype=float)
            mujoco.mj_forward(model, data)
            image_path = Path(request["image"])
            full_image_path = Path(request.get("full_image", image_path))
            renderer.update_scene(data, camera="blocks_overview")
            Image.fromarray(renderer.render()).save(full_image_path)
            if request.get("crop"):
                crop_for_vlm(full_image_path, CropTransform(**request["crop"]), image_path)
            if request.get("mode") == "visual_trace":
                decision = planner.decide_visual_trace(image_path, request["state"])
            elif request.get("mode") == "rewrite_fixed_stage":
                decision = planner.rewrite_fixed_stage_prompt(
                    image_path, request["stage"], request["fixed_prompt"], request["state"])
            else:
                decision = planner.decide_stage(
                    image_path, request["stage"], request["state"])
            print(json.dumps({"id": request["id"], "ok": True, "decision": decision}), flush=True)
        except Exception as error:
            print(json.dumps({"id": request.get("id"), "ok": False,
                              "error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__": main()
