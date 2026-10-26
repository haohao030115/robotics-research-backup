#!/usr/bin/env python3
"""Independent noVNC monitor for the online Method B simulation.

The controller process must never block on Qt/GL rendering: SONIC treats a
short LowState gap as a safety fault.  This process owns its own MuJoCo model
and only visualizes atomic qpos snapshots written by the simulator.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    title = "MuJoCo live - Method B"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 720)
    last_mtime = -1
    try:
        while True:
            try:
                mtime = args.state_file.stat().st_mtime_ns
                if mtime == last_mtime:
                    cv2.waitKey(1)
                    time.sleep(.02)
                    continue
                last_mtime = mtime
                with np.load(args.state_file, allow_pickle=False) as snapshot:
                    qpos = snapshot["qpos"].copy()
                    stage = int(snapshot["stage"])
                    phase = str(snapshot["phase"])
                    sim_time = float(snapshot["sim_time"])
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="blocks_overview")
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                label = f"stage={stage}  phase={phase}  sim={sim_time:.2f}s"
                cv2.putText(frame, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            .75, (40, 255, 40), 2, cv2.LINE_AA)
                cv2.imshow(title, frame)
                cv2.waitKey(1)
            except (FileNotFoundError, PermissionError, ValueError, OSError):
                time.sleep(.02)
    finally:
        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
