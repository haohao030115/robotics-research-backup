from __future__ import annotations

import numpy as np


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def world_to_robot(point_world: np.ndarray, root_pos: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_wxyz_to_matrix(root_quat_wxyz).T @ (np.asarray(point_world) - np.asarray(root_pos))


def robot_to_world(point_robot: np.ndarray, root_pos: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray(root_pos) + quat_wxyz_to_matrix(root_quat_wxyz) @ np.asarray(point_robot)


def safe_target(block_world: np.ndarray, delta_world: np.ndarray) -> np.ndarray:
    block = np.asarray(block_world, dtype=float)
    delta = np.clip(np.asarray(delta_world, dtype=float), -0.08, 0.08)
    target = block + delta
    if not (0.25 <= target[0] <= 0.80 and -0.45 <= target[1] <= 0.30 and 0.68 <= target[2] <= 1.05):
        raise ValueError(f"target outside configured tabletop workspace: {target.tolist()}")
    return target

