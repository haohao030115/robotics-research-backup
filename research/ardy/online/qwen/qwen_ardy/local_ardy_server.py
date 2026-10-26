#!/usr/bin/env python3
"""Persistent JSONL ARDY worker; weights and text encoder are loaded once."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-repo", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--model", default="g1")
    parser.add_argument("--device", default=None,
                        help="Torch device after visibility masking; defaults to cuda:0 when available.")
    args = parser.parse_args()
    sys.path.insert(0, str(args.ardy_repo))
    from ardy.constraints import load_constraints_lst
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.model import load_model
    from ardy.model.registry import resolve_model_name
    from ardy.motion_rep.tools import length_to_mask
    from ardy.tools import seed_everything, to_numpy

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = resolve_model_name(args.model, checkpoints_dir=str(args.checkpoints))
    model = load_model(resolved, device=device, checkpoints_dir=str(args.checkpoints))
    converter = MujocoQposConverter(model.skeleton)
    source_joint_names = [joint.get("name") for joint in
        ET.parse(converter.xml_path).findall(".//worldbody//joint")
        if joint.get("type") != "free"]
    right_arm_columns = [7+i for i,name in enumerate(source_joint_names)
                         if name.startswith("right_") and any(
                             part in name for part in ("shoulder", "elbow", "wrist"))]
    # In short hot-loaded manipulation clips, unrelated whole-body sampling can
    # destabilize SONIC.  Keep every non-right-arm joint at the measured stage
    # boundary; ARDY still generates the complete right-arm trajectory under
    # the native end-effector constraint.
    fixed_columns = [7+i for i in range(len(source_joint_names))
                     if 7+i not in right_arm_columns]
    fps = float(model.motion_rep.fps)
    history = ((int(10*fps)-model.gen_horizon_len)//model.num_frames_per_token
               * model.num_frames_per_token)
    print(json.dumps({"event": "ready", "model": resolved, "fps": fps}), flush=True)

    for line in sys.stdin:
        request = {}
        started = time.perf_counter()
        try:
            request = json.loads(line)
            # The old worker reset every segment to seed 0, so automatic
            # continuation repeatedly sampled the same weak motion mode.
            seed_everything(int(request.get("seed", int(request.get("id", 0)) + 2)))
            prompt = str(request["prompt"]); duration = float(request.get("duration_s", 3.))
            frames = int(duration*fps); constraints = load_constraints_lst(
                request["constraints"], model.skeleton)
            lengths = torch.tensor([frames], device=device); mask = length_to_mask(lengths)
            observed, motion_mask = model.motion_rep.create_conditions_from_constraints_batched(
                constraints, lengths, to_normalize=True, device=device)
            with torch.no_grad():
                motion = model([prompt], frames,
                    num_denoising_steps=int(request.get("diffusion_steps", 10)),
                    pad_mask=mask, first_heading_angle=torch.zeros(1, device=device),
                    motion_mask=motion_mask, observed_motion=observed,
                    # Match ARDY's official CLI/demo defaults.  The former 6.0
                    # constraint scale overwhelmed subtle text-driven arm motion.
                    cfg_weight=(float(request.get("text_weight", 2.)),
                                float(request.get("constraint_weight", 2.))),
                    crop_history_length=history)
                output = to_numpy(model.motion_rep.inverse(motion, is_normalized=True))
            sample = {key: (value[0] if hasattr(value, "shape") and
                      len(value.shape) and value.shape[0] == 1 else value)
                      for key, value in output.items()}
            stem = Path(request["output_stem"]); stem.parent.mkdir(parents=True, exist_ok=True)
            arrays = {key: np.asarray(value) for key, value in sample.items()}
            arrays.update(fps=np.asarray(fps), text=np.asarray(prompt))
            np.savez(stem.with_suffix(".npz"), **arrays)
            qpos = converter.dict_to_qpos(output, device)
            preserve_generated = bool(request.get("preserve_generated_qpos", False))
            scaffold_path = request.get("scaffold_qpos_csv")
            if scaffold_path:
                scaffold_all = np.loadtxt(scaffold_path, delimiter=",", ndmin=2)
                begin, end = request["scaffold_frames"]
                scaffold = scaffold_all[int(begin):int(end)]
                count = qpos.shape[-2]
                old_t = np.linspace(0., 1., len(scaffold)); new_t = np.linspace(0., 1., count)
                scaffold = np.stack([np.interp(new_t, old_t, scaffold[:, column])
                                     for column in range(scaffold.shape[1])], axis=1)
                generated = np.asarray(qpos).reshape(count, -1)
                u1 = np.linspace(0., 1., count)[:, None]
                linear = generated[:1]*(1.-u1) + generated[-1:]*u1
                residual = generated - linear
                scale = float(request.get("residual_scale", .15))
                qpos = scaffold + scale*residual
            # Rolling arm corrections must not accumulate ARDY's incidental
            # locomotion.  Anchor horizontal root, legs and waist to the first
            # generated pose; arms remain entirely ARDY-generated.
            anchor = request.get("anchor_qpos")
            # Generated clips without a successful-demo scaffold use a local
            # root frame.  Scaffold clips must retain the demo's absolute root
            # xy; SONIC tracks those coordinates directly across hot switches.
            if not scaffold_path and not preserve_generated:
                qpos[..., 0] = 0.; qpos[..., 1] = 0.
                qpos[..., 2] = (qpos[..., 2][..., :1] if anchor is None else float(anchor[2]))
                qpos[..., 3] = 1.; qpos[..., 4:7] = 0.
            if not scaffold_path and not preserve_generated:
                for column in fixed_columns:
                    if anchor is None: qpos[..., column] = qpos[..., column][..., :1]
                    else: qpos[..., column] = float(anchor[column])
            target = request.get("target_qpos")
            if not preserve_generated and anchor is not None and target is not None:
                projection_scale = float(request.get("endpoint_projection_scale", 1.0))
                count = qpos.shape[-2]
                u = np.linspace(0., 1., count, dtype=qpos.dtype)
                u = (u*u*(3.-2.*u)).reshape(*([1]*(qpos.ndim-2)), count)
                for column in right_arm_columns:
                    # Boundary projection: preserve ARDY's generated curve,
                    # while smoothly correcting its first/last samples to the
                    # measured live pose and the native IK constraint solution.
                    generated = qpos[..., column].copy()
                    start_delta = float(anchor[column]) - generated[..., :1]
                    end_delta = float(target[column]) - generated[..., -1:]
                    qpos[..., column] = generated + (1.-u)*start_delta + u*(projection_scale*end_delta)
            converter.save_csv(qpos, str(stem.with_suffix(".csv")))
            output_dir = Path(request["reference_dir"])
            subprocess.run([sys.executable, str(args.converter), str(stem.with_suffix(".npz")),
                            "--ardy-repo", str(args.ardy_repo), "--output-dir", str(output_dir)],
                           check=True, stdout=subprocess.DEVNULL)
            print(json.dumps({"id": request["id"], "ok": True,
                              "reference_dir": str(output_dir),
                              "elapsed_s": time.perf_counter()-started}), flush=True)
        except Exception as error:
            print(json.dumps({"id": request.get("id"), "ok": False,
                              "error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    main()
