"""One-shot real-model smoke test for the rolling Visual Trace protocol."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from .trace_bridge import build_ardy_condition
from .visual_trace import (crop_for_vlm, overlay_trace, task_crop,
                           world_point_to_uv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="musa:7")
    parser.add_argument("--requests", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    block = model.body("wood_block").id
    wrist = model.body("right_wrist_yaw_link").id
    center = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ np.array([.125, .045, 0.])
    load_started = time.perf_counter()
    worker = subprocess.Popen([
        os.sys.executable, "-m", "qwen_ardy.local_qwen_server", "--model", str(args.model),
        "--device", args.device, "--scene", str(args.scene)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(args.output / "worker.log").open("w"),
        text=True, bufsize=1)
    ready = json.loads(worker.stdout.readline())
    model_load_s = time.perf_counter() - load_started
    hand_uv = world_point_to_uv(model, data, "blocks_overview", center, (640, 480))
    block_uv = world_point_to_uv(model, data, "blocks_overview", data.xpos[block], (640, 480))
    crop = task_crop(hand_uv, block_uv, (640, 480))
    responses, inference_s = [], []
    for request_id in range(args.requests):
        request = {"id": request_id, "mode": "visual_trace",
                   "image": str(args.output / f"crop_{request_id}.png"),
                   "full_image": str(args.output / f"frame_{request_id}.png"),
                   "crop": crop.to_dict(), "qpos": data.qpos.tolist(),
                   "qvel": data.qvel.tolist(), "state": {
                       "block_world_m": data.xpos[block].tolist(),
                       "grasp_center_world_m": center.tolist(),
                       "contacts": [], "holding": False, "lifted": False}}
        started = time.perf_counter()
        worker.stdin.write(json.dumps(request) + "\n"); worker.stdin.flush()
        responses.append(json.loads(worker.stdout.readline()))
        inference_s.append(time.perf_counter() - started)
    worker.terminate(); worker.wait(10)
    response = responses[-1]
    if not response.get("ok"):
        raise RuntimeError(response)
    decision = response["decision"]
    condition = build_ardy_condition(model, data, decision, center, data.xpos[block][2], crop=crop)
    full_trace = condition["constraint"]["source_trace"]
    overlay_trace(args.output / f"frame_{args.requests-1}.png", full_trace["start_uv"],
                  full_trace["end_uv"], args.output / "visual_trace.png")
    result = {"ready": ready, "model_load_s": model_load_s,
              "inference_s": inference_s, "decision": decision,
              "ardy_condition": condition}
    (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
