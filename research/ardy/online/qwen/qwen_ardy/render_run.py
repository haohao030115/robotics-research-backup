#!/usr/bin/env python3
"""Render a recorded rolling-validation MuJoCo trajectory to MP4."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--stride", type=int, default=8,
        help="Render every Nth recorded state (interactive trajectories are high-rate).",
    )
    args = parser.parse_args()
    trajectory = np.load(args.run / "trajectory.npz")["qpos"]
    model = mujoco.MjModel.from_xml_path(str(args.scene)); data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 360, 480)
    output = args.run / "rolling_grasp.mp4"
    # PyAV in the deployment image mishandles MuJoCo RGB frames; force FFmpeg.
    with imageio.get_writer(
        output, format="FFMPEG", fps=args.fps, codec="libx264", quality=8
    ) as writer:
        for qpos in trajectory[:: max(1, args.stride)]:
            data.qpos[:] = qpos; mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="blocks_overview")
            writer.append_data(renderer.render())
    renderer.close(); print(output)


if __name__ == "__main__":
    main()
