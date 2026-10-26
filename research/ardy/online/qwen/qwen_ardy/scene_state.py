from __future__ import annotations

import argparse
import json

import mujoco

from .config import Paths


def inspect_scene(scene=None) -> dict:
    paths = Paths.discover()
    scene = scene or paths.scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    block_joint = model.joint("wood_block_free")
    q = int(block_joint.qposadr[0])
    cameras = [model.camera(i).name for i in range(model.ncam)]
    return {
        "scene": str(scene),
        "block_world_position_m": data.qpos[q:q+3].round(6).tolist(),
        "block_world_quaternion_wxyz": data.qpos[q+3:q+7].round(6).tolist(),
        "cameras": cameras,
        "coordinates": "MuJoCo world frame, metres; quaternion order wxyz",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    args = parser.parse_args()
    print(json.dumps(inspect_scene(args.scene), indent=2))


if __name__ == "__main__":
    main()

