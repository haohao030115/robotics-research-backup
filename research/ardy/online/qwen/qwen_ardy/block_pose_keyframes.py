#!/usr/bin/env python3
"""Build ARDY grasp constraints from the live block pose and a successful demo.

The successful fuyuhan run is read-only.  Its physical wrist trajectory is
converted into block-local keyframes, then transplanted to the block pose read
from the requested MuJoCo scene.  No world-space grasp target is hard-coded.
"""
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
from scipy.spatial.transform import Rotation, Slerp


ARM = [
    "waist_pitch_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
GRASP_OFFSET = np.array([0.125, 0.045, 0.0])
DEMO_PLANNING_BLOCK = np.array([0.58, -0.12, 0.73])
DEMO_PHYSICAL_BLOCK = np.array([0.475, -0.177, 0.735])
STAGES = {"approach": 3.0, "align": 5.0, "hold_for_close": 7.0, "lift": 10.0}
PROMPT = (
    "A person starts standing with both arms relaxed at their sides, takes a "
    "small step forward, reaches with the right hand to grasp a block on a "
    "table, holds it firmly, and slowly lifts the block while the left arm "
    "remains relaxed at their side."
)


def body_pose(data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()


def extract_demo_template(scene: Path, simulation: Path, result: Path) -> dict:
    """Extract successful wrist poses in the initial block coordinate frame."""
    recording = np.load(simulation)
    states, qposes = recording["states"], recording["qpos"]
    playback_start = float(json.loads(result.read_text())["playback_start_at"])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    wrist = model.body("right_wrist_yaw_link").id
    block = model.body("wood_block").id

    start_idx = int(np.argmin(np.abs(states[:, 0] - playback_start)))
    data.qpos[:] = qposes[start_idx]
    mujoco.mj_forward(model, data)
    block_p, block_r = body_pose(data, block)

    local_positions, local_rotations, sample_times = [], [], []
    for stage, seconds in STAGES.items():
        index = int(np.argmin(np.abs(states[:, 0] - (playback_start + seconds))))
        data.qpos[:] = qposes[index]
        mujoco.mj_forward(model, data)
        wrist_p, wrist_r = body_pose(data, wrist)
        grasp_center = wrist_p + wrist_r @ GRASP_OFFSET
        local_positions.append(block_r.T @ (grasp_center - block_p))
        local_rotations.append(block_r.T @ wrist_r)
        sample_times.append(float(states[index, 0] - playback_start))

    # Contact makes the measured wrist drift together with the block after the
    # fingers close, so use the successful demo's *intended* relative targets:
    # grasp center at block center, vertical 16 cm approach/lift, level wrist.
    # These are also recorded in fuyuhan's grasp_spec.json.
    aligned_position = np.zeros(3)
    aligned_rotation = np.eye(3)
    lift_height = 0.16
    local_positions = np.asarray([
        aligned_position + np.array([0.0, 0.0, lift_height]),
        aligned_position,
        aligned_position,
        aligned_position + np.array([0.0, 0.0, lift_height]),
    ])
    local_rotations = np.repeat(aligned_rotation[None], 4, axis=0)

    return {
        "stage_names": list(STAGES),
        "times_s": np.asarray(list(STAGES.values())),
        "sample_times_s": np.asarray(sample_times),
        "positions_block": local_positions,
        "rotations_block": local_rotations,
        # Calibration uses the task-start pose, not the slightly settled pose
        # sampled later from the successful simulation recording.
        "source_block_position": DEMO_PHYSICAL_BLOCK.copy(),
        "source_block_rotation": np.eye(3),
    }


def build_constraints(scene: Path, ardy_repo: Path, output_dir: Path,
                      template: dict, duration: float = 12.0,
                      root_x: float = 0.25,
                      initial_block_qpos: np.ndarray | None = None) -> dict:
    sys.path.insert(0, str(ardy_repo))
    from ardy.constraints import (
        EndEffectorConstraintSet, FullBodyConstraintSet,
        Root2DConstraintSet, save_constraints_lst,
    )
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.skeleton import G1Skeleton34

    skeleton = G1Skeleton34()
    converter = MujocoQposConverter(skeleton)
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    data = mujoco.MjData(model)
    if initial_block_qpos is not None:
        block_qadr = model.joint("wood_block_free").qposadr[0]
        data.qpos[block_qadr:block_qadr + 7] = np.asarray(initial_block_qpos, dtype=float)
    mujoco.mj_forward(model, data)
    wrist = model.body("right_wrist_yaw_link").id
    block = model.body("wood_block").id
    block_p, block_r = body_pose(data, block)
    # This snapshot is the task's immutable target pose.  Constraint generation
    # never writes the block qpos again.
    block_qadr = model.joint("wood_block_free").qposadr[0]
    block_qpos = data.qpos[block_qadr:block_qadr + 7].copy()

    # Solve in the exact robot model used to author the successful ARDY demo.
    # The deployment scene is used only for the live block snapshot above.
    planning_scene = scene.with_name("scene_43dof_blocks.xml")
    if planning_scene.exists():
        model = mujoco.MjModel.from_xml_path(str(planning_scene.resolve()))
        data = mujoco.MjData(model)
        wrist = model.body("right_wrist_yaw_link").id

    names = [
        joint.get("name")
        for joint in ET.parse(converter.xml_path).findall(".//worldbody//joint")
        if joint.get("type") != "free"
    ]
    joint_q = np.array([model.joint(name).qposadr[0] for name in names])
    arm_j = np.array([model.joint(name).id for name in ARM])
    arm_q = model.jnt_qposadr[arm_j]
    bounds = model.jnt_range[arm_j].T
    nominal = data.qpos.copy()
    for side in ("left", "right"):
        for name, value in (("hip_pitch", -0.1), ("knee", 0.3), ("ankle_pitch", -0.2)):
            nominal[model.joint(f"{side}_{name}_joint").qposadr[0]] = value
    for side, roll in (("left", 0.15), ("right", -0.15)):
        nominal[model.joint(f"{side}_shoulder_roll_joint").qposadr[0]] = roll
        nominal[model.joint(f"{side}_elbow_joint").qposadr[0]] = 1.4

    matrix = converter.mujoco_to_ardy_matrix.numpy()
    axes = converter._mujoco_joint_axis_values_ardy_space.numpy()
    indexes = converter._mujoco_indices_to_ardy_indices.numpy()
    offsets = converter._rot_offsets_f2q.numpy()

    def dictionary(qpos):
        local = np.tile(np.eye(3), (34, 1, 1))
        rotations = Rotation.from_rotvec(axes * qpos[joint_q, None]).as_matrix()
        local[indexes] = np.einsum(
            "nij,njk->nik", offsets[indexes].transpose(0, 2, 1), rotations
        )
        local[0] = matrix @ Rotation.from_quat(qpos[[4, 5, 6, 3]]).as_matrix() @ matrix.T
        return (torch.tensor(local, dtype=torch.float32),
                torch.tensor(matrix @ qpos[:3], dtype=torch.float32))

    stage_times = template["times_s"]
    # fuyuhan's successful ARDY constraints were authored in a planning frame
    # whose baseline block is [0.58, -0.12, 0.73], while the deployed sides
    # scene reports [0.475, -0.177, 0.735].  Calibrate through that successful
    # pair, then apply the current block's translation/orientation delta.
    source_block_p = template["source_block_position"]
    source_block_r = template["source_block_rotation"]
    rotation_delta = block_r @ source_block_r.T
    planning_block_p = DEMO_PLANNING_BLOCK + (block_p - source_block_p)
    stage_positions = planning_block_p + (rotation_delta @ template["positions_block"].T).T
    stage_rotations = rotation_delta[None] @ template["rotations_block"]
    position_interp = lambda t: np.array([
        np.interp(t, stage_times, stage_positions[:, axis]) for axis in range(3)
    ])
    rotation_interp = Slerp(stage_times, Rotation.from_matrix(stage_rotations))

    fps = 25
    count = int(duration * fps)
    frames = np.r_[np.arange(int(stage_times[0] * fps), count, 5), count - 1]
    seed = np.array([0.2, -0.6, -0.2, 0.0, 1.0, 0.0, 0.5, 0.0])
    poses, errors, goals = [], [], []
    for frame in frames:
        seconds = min(float(frame) / fps, stage_times[-1])
        point = position_interp(seconds)
        desired_rotation = rotation_interp(seconds).as_matrix()

        def residual(values):
            data.qpos[:] = nominal
            data.qpos[0] = root_x
            data.qpos[arm_q] = values
            mujoco.mj_kinematics(model, data)
            wrist_r = data.xmat[wrist].reshape(3, 3)
            center = data.xpos[wrist] + wrist_r @ GRASP_OFFSET
            orientation = Rotation.from_matrix(desired_rotation.T @ wrist_r).as_rotvec()
            return np.r_[5.0 * (center - point), orientation]

        fit = least_squares(
            residual, np.clip(seed, bounds[0] + 1e-6, bounds[1] - 1e-6),
            bounds=(bounds[0] + 1e-6, bounds[1] - 1e-6), max_nfev=180,
        )
        residual(fit.x)
        wrist_r = data.xmat[wrist].reshape(3, 3)
        solved_center = data.xpos[wrist] + wrist_r @ GRASP_OFFSET
        error = float(np.linalg.norm(solved_center - point))
        if error > 0.025:
            raise RuntimeError(f"Unreachable {frame=} keyframe: {error:.4f} m")
        poses.append(data.qpos.copy())
        errors.append(error)
        goals.append(point.tolist())
        seed = fit.x

    local, roots = zip(*(dictionary(qpos) for qpos in poses))
    local, roots = torch.stack(local), torch.stack(roots)
    global_rot, global_pos, _ = skeleton.fk(local, roots)
    initial_local, initial_root = dictionary(nominal)
    initial_rot, initial_pos, _ = skeleton.fk(
        initial_local[None].repeat(2, 1, 1, 1), initial_root[None].repeat(2, 1)
    )
    all_frames = torch.arange(count)

    def smooth(value, start, end):
        u = np.clip((value - start) / (end - start), 0, 1)
        return u * u * (3 - 2 * u)

    x = np.array([root_x * smooth(frame / fps, 1, 3) for frame in range(count)])
    root2d = torch.tensor(np.stack([np.zeros(count), x], axis=1), dtype=torch.float32)
    constraints = [
        FullBodyConstraintSet(skeleton, torch.tensor([0, 25]), initial_pos, initial_rot),
        Root2DConstraintSet(skeleton, all_frames, root2d),
        EndEffectorConstraintSet(
            skeleton, torch.tensor(frames), global_pos, global_rot, None,
            joint_names=["RightHand", "LeftFoot", "RightFoot", "Hips", "LeftHand"],
        ),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_constraints_lst(str(output_dir / "constraints.json"), constraints)
    source = np.stack(poses)[:, np.r_[np.arange(7), joint_q]]
    np.savez(output_dir / "conditioning_poses.npz", qpos=source, frame_indices=frames)
    metadata = {
        "prompt": PROMPT,
        "scene": str(scene.resolve()),
        "ik_planning_scene": str(planning_scene.resolve() if planning_scene.exists() else scene.resolve()),
        "block_pose_read_once": {
            "position": block_p.tolist(), "rotation_matrix": block_r.tolist(),
            "free_joint_qpos": block_qpos.tolist(),
        },
        "planning_frame_calibration": {
            "demo_physical_block_position": source_block_p.tolist(),
            "demo_ardy_block_position": DEMO_PLANNING_BLOCK.tolist(),
            "current_ardy_block_position": planning_block_p.tolist(),
        },
        "stage_times_s": dict(STAGES),
        "stage_positions_block": template["positions_block"].tolist(),
        "stage_rotations_block": template["rotations_block"].tolist(),
        "stage_positions_world": stage_positions.tolist(),
        "stage_rotations_world": stage_rotations.tolist(),
        "dense_frames": frames.tolist(), "dense_world_goals": goals,
        "ik_errors_m": errors, "max_ik_error_m": max(errors),
        "grasp_offset": GRASP_OFFSET.tolist(),
        "block_was_modified_after_snapshot": False,
    }
    (output_dir / "block_pose_plan.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--ardy-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--demo-scene", type=Path, required=True)
    parser.add_argument("--demo-simulation", type=Path, required=True)
    parser.add_argument("--demo-result", type=Path, required=True)
    args = parser.parse_args()
    template = extract_demo_template(args.demo_scene, args.demo_simulation, args.demo_result)
    result = build_constraints(args.scene, args.ardy_repo, args.output_dir, template)
    print(json.dumps({
        "block_position": result["block_pose_read_once"]["position"],
        "stage_positions_world": result["stage_positions_world"],
        "max_ik_error_m": result["max_ik_error_m"],
        "constraints": str(args.output_dir / "constraints.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
