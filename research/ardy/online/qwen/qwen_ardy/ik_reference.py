"""Create a short SONIC reference from the *current* MuJoCo pose.

This is the numerical constraint layer beneath Qwen's semantic instruction.  It
does not close the hand; it only moves the open right-hand grasp centre toward
the live block pose.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .reference_runtime import FORBIDDEN_HAND_FILES


RIGHT_ARM = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
GRASP_OFFSET = np.array([.125, .045, 0.0])


def _last_csv_row(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows)
        last = None
        for last in rows:
            pass
    if last is None:
        raise ValueError(f"empty reference CSV: {path}")
    return header, last


def publish_cartesian_correction(source: Path, reference_root: Path, name: str,
                                 model: mujoco.MjModel, data: mujoco.MjData,
                                 body_joint_ids: np.ndarray,
                                 target_world: np.ndarray, prompt: str,
                                 frames: int = 70, dt: float = .02) -> tuple[Path, np.ndarray, np.ndarray]:
    """Solve live-pose IK and atomically publish an open-hand correction clip."""
    q0 = data.qpos.copy()
    arm_ids = np.array([model.joint(n).id for n in RIGHT_ARM], dtype=int)
    arm_qadr = model.jnt_qposadr[arm_ids]
    bounds = model.jnt_range[arm_ids].T
    wrist = model.body("right_wrist_yaw_link").id
    seed = q0[arm_qadr].copy()
    initial_rotation = data.xmat[wrist].reshape(3, 3).copy()
    target_world = np.asarray(target_world, dtype=float)

    def residual(q):
        data.qpos[:] = q0
        data.qpos[arm_qadr] = q
        mujoco.mj_kinematics(model, data)
        rotation = data.xmat[wrist].reshape(3, 3)
        center = data.xpos[wrist] + rotation @ GRASP_OFFSET
        rotation_error = Rotation.from_matrix(initial_rotation.T @ rotation).as_rotvec()
        return np.r_[12.0 * (center - target_world), .25 * rotation_error,
                     .035 * (q - seed)]

    fit = least_squares(residual, np.clip(seed, bounds[0] + 1e-5, bounds[1] - 1e-5),
                        bounds=(bounds[0] + 1e-5, bounds[1] - 1e-5), max_nfev=240)
    residual(fit.x)
    solved_center = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ GRASP_OFFSET
    error = float(np.linalg.norm(solved_center - target_world))
    data.qpos[:] = q0
    mujoco.mj_kinematics(model, data)
    if error > .018:
        raise RuntimeError(f"live IK target is unreachable: {error:.4f} m")

    body_joint_ids = np.asarray(body_joint_ids, dtype=int)
    body_qadr = model.jnt_qposadr[body_joint_ids]
    start = q0[body_qadr].copy()
    goal = start.copy()
    index_by_id = {int(joint_id): i for i, joint_id in enumerate(body_joint_ids)}
    for joint_id, value in zip(arm_ids, fit.x):
        goal[index_by_id[int(joint_id)]] = value
    u = np.linspace(0., 1., frames)
    u = u * u * (3. - 2. * u)
    joint_pos = start[None] + u[:, None] * (goal - start)[None]
    joint_vel = np.gradient(joint_pos, dt, axis=0)

    final = reference_root / name
    temporary = reference_root / f".{name}.publishing-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for item in source.iterdir():
        if not item.is_file() or item.name in FORBIDDEN_HAND_FILES:
            continue
        target = temporary / item.name
        if item.name in ("joint_pos.csv", "joint_vel.csv"):
            header, _ = _last_csv_row(item)
            values = joint_pos if item.name == "joint_pos.csv" else joint_vel
            with target.open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(header); writer.writerows(values)
        elif item.suffix == ".csv":
            header, last = _last_csv_row(item)
            with target.open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(header)
                writer.writerows([last] * frames)
        else:
            shutil.copy2(item, target)
    (temporary / "stage.json").write_text(json.dumps({
        "stage": name, "prompt": prompt, "hand_control": "contact_driven",
        "constraint": {"type": "live_cartesian_ik", "target_world_m": target_world.tolist(),
                       "solved_center_world_m": solved_center.tolist(), "ik_error_m": error},
    }, indent=2) + "\n")
    if final.exists():
        shutil.rmtree(final)
    temporary.rename(final)
    arm_motor_indexes = np.array([index_by_id[int(joint_id)] for joint_id in arm_ids], dtype=int)
    return final, goal, arm_motor_indexes
