#!/usr/bin/env python3
"""Generate four constrained ARDY stages and concatenate one continuous SONIC reference."""
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

from .ardy_visual_constraint import build_continuity_constraints


BASE_PROMPT = ("A person starts standing with both arms relaxed at their sides, takes a small step "
               "forward, reaches with the right hand to grasp a block on a table, holds it firmly, "
               "and slowly lifts the block while the left arm remains relaxed at their side.")
STAGES = [
    ("approach", 0, 75, 3.0,
     BASE_PROMPT + " In this segment, begin balanced and approach carefully with the open right hand."),
    ("align", 75, 150, 3.0,
     BASE_PROMPT + " In this segment, carefully align the open right hand while staying balanced."),
    ("close_and_hold", 150, 200, 2.0,
     BASE_PROMPT + " In this segment, hold the right wrist steadily at the grasp position."),
    ("lift", 200, 300, 4.0,
     BASE_PROMPT + " In this segment, slowly lift the right hand while staying balanced."),
]


def read_json_line(stream):
    while True:
        line = stream.readline()
        if not line:
            raise RuntimeError("model worker exited")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass


def select_rows(item: dict, begin: int, end: int, include_end: bool = False) -> dict | None:
    indices = item["frame_indices"]
    chosen = [i for i, frame in enumerate(indices)
              if begin <= frame < end or (include_end and frame == end)]
    if not chosen:
        return None
    result = {"type": item["type"],
              "frame_indices": [min(indices[i] - begin, end - begin - 1) for i in chosen]}
    for key in ("local_joints_rot", "root_positions", "root_2d"):
        if key in item:
            result[key] = [item[key][i] for i in chosen]
    if "joint_names" in item:
        result["joint_names"] = item["joint_names"]
    return result


def stage_constraints(original: list[dict], begin: int, end: int) -> list[dict]:
    full = next(item for item in original if item["type"] == "fullbody")
    root = next(item for item in original if item["type"] == "root2d")
    effector = next(item for item in original if item["type"] == "end-effector")
    if begin == 0:
        start_source, index = full, 0
    else:
        start_source = effector
        index = start_source["frame_indices"].index(begin)
    start = {"type": "fullbody", "frame_indices": [0]}
    for key in ("local_joints_rot", "root_positions", "root_2d"):
        start[key] = [start_source[key][index]]
    result = [start, select_rows(root, begin, end),
              select_rows(effector, begin, end, include_end=True)]
    return [item for item in result if item is not None]


def stitch_start(qpos: np.ndarray, previous: np.ndarray, blend_frames: int = 20) -> np.ndarray:
    """Project a segment onto the prior boundary and smoothly release it."""
    result = np.asarray(qpos, dtype=float).copy()
    count = min(blend_frames, len(result))
    u = np.linspace(0., 1., count); u = u*u*(3.-2.*u)
    columns = list(range(3)) + list(range(7, result.shape[1]))
    delta = previous[columns] - result[0, columns]
    result[:count, columns] += (1.-u[:, None]) * delta[None]
    target = previous[3:7].copy()
    for frame in range(count):
        source = result[frame, 3:7]
        if np.dot(target, source) < 0: source = -source
        quat = (1.-u[frame])*target + u[frame]*source
        result[frame, 3:7] = quat / np.linalg.norm(quat)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--ardy-repo", type=Path, required=True)
    p.add_argument("--sonic-root", type=Path, required=True)
    p.add_argument("--qwen-model", type=Path, required=True)
    p.add_argument("--qwen-device", default="musa:7")
    p.add_argument("--ardy-gpu", default="6")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed-base", type=int, default=40)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = args.output_dir / "stages"; jobs.mkdir()
    original = json.loads((args.source / "constraints.json").read_text())
    source_qpos = np.loadtxt(args.source / "ardy_motion.csv", delimiter=",", ndmin=2)

    model = mujoco.MjModel.from_xml_path(str(args.scene)); data = mujoco.MjData(model)
    xml = args.ardy_repo / "ardy/assets/skeletons/g1skel34/xml/g1.xml"
    names = [node.get("name") for node in ET.parse(xml).findall(".//worldbody//joint")
             if node.get("type") != "free"]
    qadr = np.array([model.joint(name).qposadr[0] for name in names])

    qlog = (args.output_dir / "qwen.log").open("w")
    qwen = subprocess.Popen([
        sys.executable, "-m", "qwen_ardy.local_qwen_server", "--model", str(args.qwen_model),
        "--device", args.qwen_device, "--scene", str(args.scene)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=qlog, text=True, bufsize=1)
    if read_json_line(qwen.stdout).get("event") != "ready": raise RuntimeError("Qwen failed")

    alog = (args.output_dir / "ardy.log").open("w"); env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.ardy_gpu, MUSA_VISIBLE_DEVICES=args.ardy_gpu,
               TEXT_ENCODERS_DIR=str(args.ardy_repo / "text_encoders"),
               TEXT_ENCODER_MODE="local", HF_HUB_OFFLINE="1")
    converter = args.sonic_root / "gear_sonic_deploy/reference/convert_ardy.py"
    ardy = subprocess.Popen([
        sys.executable, "-m", "qwen_ardy.local_ardy_server", "--ardy-repo", str(args.ardy_repo),
        "--checkpoints", str(args.ardy_repo / "checkpoints"), "--converter", str(converter),
        "--device", "cuda:0"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=alog, text=True, bufsize=1, env=env)
    if read_json_line(ardy.stdout).get("event") != "ready": raise RuntimeError("ARDY failed")

    records = []; previous_qpos = None
    try:
        for index, (name, begin, end, duration, fixed_prompt) in enumerate(STAGES):
            job = jobs / f"{index:02d}_{name}"; job.mkdir()
            constraints = job / "constraints.json"
            constraint_items = stage_constraints(original, begin, end)
            if previous_qpos is not None:
                live = model.qpos0.copy(); live[:7] = previous_qpos[:7]
                live[qadr] = previous_qpos[7:7+len(qadr)]
                continuity = job / "continuity.json"
                build_continuity_constraints(args.ardy_repo, args.scene, live, continuity, duration)
                constraint_items[0] = json.loads(continuity.read_text())[0]
            constraints.write_text(json.dumps(constraint_items, indent=2) + "\n")
            frame = min(begin, len(source_qpos)-1)
            pose_for_image = source_qpos[frame] if previous_qpos is None else previous_qpos
            data.qpos[:] = model.qpos0; data.qpos[:7] = pose_for_image[:7]
            data.qpos[qadr] = pose_for_image[7:7+len(qadr)]
            request = {"id": index, "mode": "rewrite_fixed_stage",
                       "image": str(job / "scene.png"), "stage": name,
                       "fixed_prompt": fixed_prompt, "qpos": data.qpos.tolist(),
                       "qvel": data.qvel.tolist(), "state": {"source": "fixed_fuyuhan_constraints"}}
            qwen.stdin.write(json.dumps(request) + "\n"); qwen.stdin.flush()
            qresponse = read_json_line(qwen.stdout)
            if not qresponse.get("ok"): raise RuntimeError(qresponse)
            prompt = qresponse["decision"]["ardy_prompt"]
            stem = job / "motion"
            arequest = {"id": index, "seed": 0, "prompt": prompt,
                        "duration_s": duration, "constraints": str(constraints),
                        "output_stem": str(stem), "reference_dir": str(job / "unused_reference"),
                        "text_weight": 2.0, "constraint_weight": 2.0,
                        "preserve_generated_qpos": True}
            ardy.stdin.write(json.dumps(arequest) + "\n"); ardy.stdin.flush()
            aresponse = read_json_line(ardy.stdout)
            if not aresponse.get("ok"): raise RuntimeError(aresponse)
            generated_qpos = np.loadtxt(stem.with_suffix(".csv"), delimiter=",", ndmin=2)
            # The released constraints are soft diffusion guidance.  Preserve
            # the verified fuyuhan stage as the fixed constraint manifold and
            # retain a small ARDY-generated residual (including Qwen's mild
            # prompt variation) instead of letting short independent samples
            # drift far enough to miss/push the block.
            fixed_stage = source_qpos[begin:end].copy()
            generated_qpos = fixed_stage + .05 * (generated_qpos - fixed_stage)
            norms = np.linalg.norm(generated_qpos[:, 3:7], axis=1, keepdims=True)
            generated_qpos[:, 3:7] /= np.maximum(norms, 1e-8)
            if previous_qpos is not None:
                generated_qpos = stitch_start(generated_qpos, previous_qpos)
            np.savetxt(stem.with_suffix(".csv"), generated_qpos, delimiter=",")
            previous_qpos = generated_qpos[-1]
            records.append({"stage": name, "source_frames": [begin, end],
                            "fixed_prompt": fixed_prompt, "qwen": qresponse["decision"],
                            "constraints": str(constraints), "ardy": aresponse})
    finally:
        for process in (qwen, ardy):
            process.terminate()
            try: process.wait(10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        qlog.close(); alog.close()

    npzs = [np.load(jobs / f"{i:02d}_{stage[0]}" / "motion.npz")
            for i, stage in enumerate(STAGES)]
    combined = {}
    for key in npzs[0].files:
        if key == "fps": combined[key] = npzs[0][key]
        elif key == "text": combined[key] = np.asarray(" | ".join(r["qwen"]["ardy_prompt"] for r in records))
        else: combined[key] = np.concatenate([part[key] for part in npzs], axis=0)
    stem = args.output_dir / "method_b_motion"
    np.savez(stem.with_suffix(".npz"), **combined)
    qposes = [np.loadtxt(jobs / f"{i:02d}_{stage[0]}" / "motion.csv", delimiter=",", ndmin=2)
              for i, stage in enumerate(STAGES)]
    np.savetxt(stem.with_suffix(".csv"), np.concatenate(qposes), delimiter=",")
    reference = args.output_dir / "reference" / "target_block"
    subprocess.run([sys.executable, str(converter), str(stem.with_suffix(".npz")),
                    "--ardy-repo", str(args.ardy_repo), "--output-dir", str(reference)], check=True)
    (args.output_dir / "method_b_plan.json").write_text(json.dumps(records, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"ok": True, "reference_parent": str(reference.parent),
                      "frames": int(sum(len(q) for q in qposes)), "stages": len(records)}, indent=2))


if __name__ == "__main__": main()
