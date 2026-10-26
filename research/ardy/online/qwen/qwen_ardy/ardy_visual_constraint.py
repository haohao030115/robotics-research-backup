#!/usr/bin/env python3
"""Compile a Visual Trace world endpoint into ARDY's native constraint JSON."""
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

from .ik_reference import GRASP_OFFSET, RIGHT_ARM


def build_continuity_constraints(ardy_repo: Path, scene: Path, qpos: np.ndarray,
                                 output: Path, duration_s: float = 3., fps: int = 25) -> dict:
    """Constrain only the live starting pose; leave the text motion free."""
    sys.path.insert(0, str(ardy_repo))
    from ardy.skeleton import G1Skeleton34
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.constraints import FullBodyConstraintSet, Root2DConstraintSet, save_constraints_lst
    skeleton=G1Skeleton34(); converter=MujocoQposConverter(skeleton)
    model=mujoco.MjModel.from_xml_path(str(scene)); qpos=np.asarray(qpos,float)
    names=[joint.get("name") for joint in ET.parse(converter.xml_path).findall(".//worldbody//joint")
           if joint.get("type")!="free"]
    joint_q=np.array([model.joint(name).qposadr[0] for name in names])
    matrix=converter.mujoco_to_ardy_matrix.numpy(); axes=converter._mujoco_joint_axis_values_ardy_space.numpy()
    indexes=converter._mujoco_indices_to_ardy_indices.numpy(); offsets=converter._rot_offsets_f2q.numpy()
    local=np.tile(np.eye(3),(34,1,1)); rotations=Rotation.from_rotvec(axes*qpos[joint_q,None]).as_matrix()
    local[indexes]=np.einsum("nij,njk->nik",offsets[indexes].transpose(0,2,1),rotations)
    local[0]=matrix@Rotation.from_quat(qpos[[4,5,6,3]]).as_matrix()@matrix.T
    roots=torch.tensor(matrix@qpos[:3],dtype=torch.float32)[None]
    grots,gpos,_=skeleton.fk(torch.tensor(local,dtype=torch.float32)[None],roots)
    count=max(2,int(duration_s*fps))
    constraints=[FullBodyConstraintSet(skeleton,torch.tensor([0]),gpos,grots),
                 Root2DConstraintSet(skeleton,torch.arange(count),torch.zeros((count,2)))]
    output.parent.mkdir(parents=True,exist_ok=True); save_constraints_lst(str(output),constraints)
    result={"type":"live_start_continuity","duration_s":duration_s,"fps":fps}
    output.with_suffix(".meta.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def build_constraints(ardy_repo: Path, scene: Path, qpos: np.ndarray,
                      endpoint_world: np.ndarray, output: Path,
                      duration_s: float = 3., fps: int = 25,
                      seed_qpos: np.ndarray | None = None) -> dict:
    sys.path.insert(0, str(ardy_repo))
    from ardy.skeleton import G1Skeleton34
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.constraints import (EndEffectorConstraintSet, FullBodyConstraintSet,
                                  Root2DConstraintSet, save_constraints_lst)

    skeleton = G1Skeleton34(); converter = MujocoQposConverter(skeleton)
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    qpos = np.asarray(qpos, dtype=float); data.qpos[:] = qpos; mujoco.mj_forward(model, data)
    names = [joint.get("name") for joint in
             ET.parse(converter.xml_path).findall(".//worldbody//joint")
             if joint.get("type") != "free"]
    joint_q = np.array([model.joint(name).qposadr[0] for name in names])
    arm_ids = np.array([model.joint(name).id for name in RIGHT_ARM])
    arm_q = model.jnt_qposadr[arm_ids]; bounds = model.jnt_range[arm_ids].T
    wrist = model.body("right_wrist_yaw_link").id
    start = qpos.copy()
    seed = start if seed_qpos is None else np.asarray(seed_qpos, dtype=float)
    # Preserve the successful demo's grasp-side wrist orientation as an IK
    # seed while the requested centre still comes from the live block pose.
    data.qpos[:] = seed; mujoco.mj_kinematics(model, data)
    desired_rotation = data.xmat[wrist].reshape(3, 3).copy()

    def residual(values):
        data.qpos[:] = start; data.qpos[arm_q] = values; mujoco.mj_kinematics(model, data)
        rotation = data.xmat[wrist].reshape(3, 3)
        centre = data.xpos[wrist] + rotation @ GRASP_OFFSET
        orientation = Rotation.from_matrix(desired_rotation.T @ rotation).as_rotvec()
        return np.r_[16 * (centre-endpoint_world), .025*orientation,
                     .01*(values-seed[arm_q])]

    candidates = [seed[arm_q], start[arm_q], (bounds[0] + bounds[1]) / 2]
    fits = [least_squares(
        residual, np.clip(candidate, bounds[0]+1e-5, bounds[1]-1e-5),
        bounds=(bounds[0]+1e-5, bounds[1]-1e-5), max_nfev=600)
        for candidate in candidates]
    def position_error(candidate):
        residual(candidate.x)
        centre = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ GRASP_OFFSET
        return float(np.linalg.norm(centre-endpoint_world))
    fit = min(fits, key=position_error)
    residual(fit.x); solved = data.qpos.copy()
    solved_centre = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ GRASP_OFFSET
    error = float(np.linalg.norm(solved_centre-endpoint_world))
    if not np.isfinite(error):
        raise RuntimeError("Visual Trace IK produced a non-finite result")

    matrix = converter.mujoco_to_ardy_matrix.numpy()
    axes = converter._mujoco_joint_axis_values_ardy_space.numpy()
    indexes = converter._mujoco_indices_to_ardy_indices.numpy()
    offsets = converter._rot_offsets_f2q.numpy()

    def dictionary(pose):
        local = np.tile(np.eye(3), (34, 1, 1))
        rotations = Rotation.from_rotvec(axes * pose[joint_q, None]).as_matrix()
        local[indexes] = np.einsum("nij,njk->nik", offsets[indexes].transpose(0,2,1), rotations)
        local[0] = matrix @ Rotation.from_quat(pose[[4,5,6,3]]).as_matrix() @ matrix.T
        return (torch.tensor(local, dtype=torch.float32),
                torch.tensor(matrix @ pose[:3], dtype=torch.float32))

    count = max(2, int(duration_s * fps)); final = count - 1
    frames = np.unique(np.r_[np.arange(0, count, 5), final]).astype(int)
    poses = []
    for frame in frames:
        u = frame / max(1, final); u = u*u*(3.-2.*u)
        pose = start.copy(); pose[arm_q] = (1.-u)*start[arm_q] + u*fit.x
        poses.append(pose)
    local, roots = zip(*(dictionary(pose) for pose in poses))
    local, roots = torch.stack(local), torch.stack(roots)
    global_rot, global_pos, _ = skeleton.fk(local, roots)
    constraints = [
        FullBodyConstraintSet(skeleton, torch.tensor([0]),
                              global_pos[:1], global_rot[:1]),
        Root2DConstraintSet(skeleton, torch.arange(count),
                            torch.zeros((count, 2), dtype=torch.float32)),
        EndEffectorConstraintSet(
            skeleton, torch.tensor(frames), global_pos, global_rot, None,
            joint_names=["RightHand", "LeftFoot", "RightFoot", "Hips", "LeftHand"]),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    save_constraints_lst(str(output), constraints)
    metadata = {"endpoint_world_m": np.asarray(endpoint_world).tolist(),
                "solved_grasp_center_world_m": solved_centre.tolist(),
                "ik_error_m": error, "duration_s": duration_s, "fps": fps,
                "right_arm_only_exact": error <= .02,
                "dense_constraint_frames": frames.tolist(),
                "solved_mujoco_qpos": solved.tolist()}
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ardy-repo", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--qpos", type=Path, help=".npy live MuJoCo qpos; defaults to scene qpos0")
    parser.add_argument("--endpoint", nargs=3, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3.)
    args = parser.parse_args()
    qpos = (np.load(args.qpos) if args.qpos else
            mujoco.MjModel.from_xml_path(str(args.scene)).qpos0.copy())
    result = build_constraints(args.ardy_repo, args.scene, qpos,
                               np.asarray(args.endpoint), args.output, args.duration)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
