#!/usr/bin/env python3
"""Build precise ARDY keyframe constraints from MuJoCo world coordinates."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import safe_target

ARM = ["waist_pitch_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
       "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
       "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
GRASP_OFFSET = np.array([.125, .045, 0.])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ardy-repo", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--root-x", type=float, default=.25)
    parser.add_argument("--arms-at-sides", action="store_true")
    parser.add_argument("--delta-x", type=float, default=0.)
    parser.add_argument("--delta-y", type=float, default=0.)
    parser.add_argument("--delta-z", type=float, default=0.)
    args = parser.parse_args()
    sys.path.insert(0, str(args.ardy_repo))
    from ardy.skeleton import G1Skeleton34
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.constraints import (FullBodyConstraintSet, Root2DConstraintSet,
                                  EndEffectorConstraintSet, save_constraints_lst)

    skel = G1Skeleton34()
    converter = MujocoQposConverter(skel)
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [j.get("name") for j in ET.parse(converter.xml_path).findall(".//worldbody//joint")
             if j.get("type") != "free"]
    joint_q = np.array([model.joint(n).qposadr[0] for n in names])
    arm_j = np.array([model.joint(n).id for n in ARM])
    arm_q = model.jnt_qposadr[arm_j]
    wrist = model.body("right_wrist_yaw_link").id
    block_joint = model.joint("wood_block_free")
    block_q = int(block_joint.qposadr[0])
    block_world = data.qpos[block_q:block_q+3].copy()
    target = safe_target(block_world, np.array([args.delta_x, args.delta_y, args.delta_z]))
    target[2] = max(target[2], .73)

    nominal = data.qpos.copy()
    for side in ["left", "right"]:
        for name, value in [("hip_pitch", -.1), ("knee", .3), ("ankle_pitch", -.2)]:
            nominal[model.joint(f"{side}_{name}_joint").qposadr[0]] = value
    if args.arms_at_sides:
        for side, roll in [("left", .15), ("right", -.15)]:
            nominal[model.joint(f"{side}_shoulder_roll_joint").qposadr[0]] = roll
            nominal[model.joint(f"{side}_elbow_joint").qposadr[0]] = 1.4

    matrix = converter.mujoco_to_ardy_matrix.numpy()
    axes = converter._mujoco_joint_axis_values_ardy_space.numpy()
    indexes = converter._mujoco_indices_to_ardy_indices.numpy()
    offsets = converter._rot_offsets_f2q.numpy()

    def dictionary(qpos):
        local = np.tile(np.eye(3), (34, 1, 1))
        rotations = Rotation.from_rotvec(axes * qpos[joint_q, None]).as_matrix()
        local[indexes] = np.einsum("nij,njk->nik", offsets[indexes].transpose(0, 2, 1), rotations)
        local[0] = matrix @ Rotation.from_quat(qpos[[4, 5, 6, 3]]).as_matrix() @ matrix.T
        return torch.tensor(local, dtype=torch.float32), torch.tensor(matrix @ qpos[:3], dtype=torch.float32)

    seed = np.array([.2, -.6, -.2, 0, 1, 0, .5, 0.])
    bounds = model.jnt_range[arm_j].T

    def solve(point, initial):
        data.qpos[:] = nominal
        data.qpos[0] = args.root_x
        def residual(q):
            data.qpos[arm_q] = q
            mujoco.mj_kinematics(model, data)
            rotation = data.xmat[wrist].reshape(3, 3)
            center = data.xpos[wrist] + rotation @ GRASP_OFFSET
            return np.r_[5 * (center - point), Rotation.from_matrix(rotation).as_rotvec()]
        fit = least_squares(residual, np.clip(initial, bounds[0]+1e-6, bounds[1]-1e-6),
                            bounds=(bounds[0]+1e-6, bounds[1]-1e-6), max_nfev=180)
        residual(fit.x)
        error = float(np.linalg.norm(data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ GRASP_OFFSET - point))
        if error > .025:
            raise RuntimeError(f"Unreachable world constraint: {error:.3f} m")
        return data.qpos.copy(), fit.x, error

    duration, fps = 12., 25
    count = int(duration * fps)
    frames = np.r_[np.arange(75, count, 5), count-1]
    poses, errors, goals = [], [], []
    def smooth(t, start, end):
        u = np.clip((t-start)/(end-start), 0, 1)
        return u*u*(3-2*u)
    for frame in frames:
        t = frame / fps
        z = .16*(1-smooth(t, 3, 5)) + .16*smooth(t, 8, 10)
        point = target + np.array([0, 0, z])
        pose, seed, error = solve(point, seed)
        poses.append(pose); errors.append(error); goals.append(point.tolist())

    local, roots = zip(*(dictionary(q) for q in poses))
    local, roots = torch.stack(local), torch.stack(roots)
    grots, gpos, _ = skel.fk(local, roots)
    initial_local, initial_root = dictionary(nominal)
    initial_rots, initial_pos, _ = skel.fk(initial_local[None].repeat(2,1,1,1),
                                            initial_root[None].repeat(2,1))
    all_frames = torch.arange(count)
    root_x = np.array([args.root_x*smooth(i/fps, 1, 3) for i in range(count)])
    root2d = torch.tensor(np.stack([np.zeros(count), root_x], axis=1), dtype=torch.float32)
    constraints = [
        FullBodyConstraintSet(skel, torch.tensor([0, 25]), initial_pos, initial_rots),
        Root2DConstraintSet(skel, all_frames, root2d),
        EndEffectorConstraintSet(
            skel, torch.tensor(frames), gpos, grots, None,
            joint_names=["RightHand", "LeftFoot", "RightFoot", "Hips", "LeftHand"],
        ),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_constraints_lst(str(args.output_dir / "constraints.json"), constraints)
    spec = {
        "coordinate_frame": "MuJoCo world, metres",
        "block_world": block_world.tolist(),
        "constraint_delta_world_m": [args.delta_x, args.delta_y, args.delta_z],
        "target_grasp_center_world": target.tolist(),
        "grasp_center_goals_world": goals,
        "ik_errors_m": errors,
        "max_ik_error_m": max(errors),
        "prompt_contains_numeric_coordinates": False,
    }
    (args.output_dir / "constraint_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()

