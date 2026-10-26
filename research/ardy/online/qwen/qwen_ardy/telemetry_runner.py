#!/usr/bin/env python3
"""SONIC grasp runner with live MuJoCo world-state telemetry."""
from __future__ import annotations

import argparse
import json
import os
import pty
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--robot-x", type=float, default=.24)
    parser.add_argument("--robot-y", type=float, default=-.03)
    parser.add_argument("--hand-kp", type=float, default=6.)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--arms-at-sides", action="store_true")
    args = parser.parse_args()
    root = Path(os.environ.get("SONIC_ROOT", str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "GR00T-WholeBodyControl")))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    config = SimLoopConfig().load_wbc_yaml()
    config["ROBOT_SCENE"] = args.scene
    config["ENABLE_ELASTIC_BAND"] = False
    runner = BaseSimulator(config, onscreen=args.viewer)
    env = runner.sim_env
    env.elastic_band = None
    env.mj_data.qpos[0:2] = [args.robot_x, args.robot_y]
    if args.arms_at_sides:
        config["DEFAULT_DOF_ANGLES"] = np.array(config["DEFAULT_DOF_ANGLES"]).copy()
        pose = {"left_shoulder_roll_joint": .15, "right_shoulder_roll_joint": -.15,
                "left_elbow_joint": 1.4, "right_elbow_joint": 1.4}
        for name, value in pose.items():
            index = int(np.flatnonzero(env.body_joint_index == env.mj_model.joint(name).id)[0])
            config["DEFAULT_DOF_ANGLES"][index] = value
    env.mj_data.qpos[env.body_joint_index + env.qpos_offset - 1] = config["DEFAULT_DOF_ANGLES"]
    for i in range(env.num_body_dof):
        cmd = env.unitree_bridge.low_cmd.motor_cmd[i]
        cmd.q = config["DEFAULT_DOF_ANGLES"][i]
        cmd.kp, cmd.kd = config["MOTOR_KP"][i], config["MOTOR_KD"][i]
    for commands in [env.unitree_bridge.left_hand_cmd, env.unitree_bridge.right_hand_cmd]:
        for cmd in commands.motor_cmd:
            cmd.kp, cmd.kd = args.hand_kp, .1
    mujoco.mj_forward(env.mj_model, env.mj_data)
    model, data = env.mj_model, env.mj_data
    if env.viewer:
        env.viewer.cam.lookat[:] = [.35, 0, .85]
        env.viewer.cam.distance, env.viewer.cam.azimuth, env.viewer.cam.elevation = 2., 135, -20
    wrist = model.body("right_wrist_yaw_link").id
    block = model.body("wood_block").id
    block_geom = model.geom("wood_block_geom").id
    master, slave = pty.openpty()
    controller = [
        "./target/release/g1_deploy_onnx_ref", "lo", "policy/release/model_decoder.onnx",
        str(args.motion_dir.resolve()), "--obs-config", "policy/release/observation_config.yaml",
        "--encoder-file", "policy/release/model_encoder.onnx", "--input-type", "keyboard",
        "--policy-precision", "32", "--disable-crc-check", "--enable-csv-logs",
        "--logs-dir", str((args.output_dir / "controller").resolve()),
    ]
    controller_log = (args.output_dir / "controller.log").open("w")
    child = subprocess.Popen(controller, stdin=slave, stdout=controller_log,
                             stderr=subprocess.STDOUT, cwd=root / "gear_sonic_deploy")
    os.close(slave)
    telemetry = (args.output_dir / "telemetry.jsonl").open("w", buffering=1)
    states, qpos, qvel = [], [], []
    ready_at = None
    sent_start = sent_play = reset_objects = False
    start = next_tick = time.monotonic()
    initial_block_z = None
    print(f"Controller pid: {child.pid}; live telemetry: {telemetry.name}", flush=True)
    try:
        for step in range(int(150 / env.sim_dt)):
            t = step * env.sim_dt
            if ready_at is None and step % 20 == 0 and "Init Done" in Path(controller_log.name).read_text():
                ready_at = t; print(f"Controller ready at {t:.3f}s", flush=True)
            phase = t-ready_at if ready_at is not None else -100
            if phase >= .1 and not sent_start:
                os.write(master, b"]"); sent_start = True
            if phase >= 1.5 and not reset_objects:
                for name in ["wood_block_free", "red_block_free"]:
                    joint = model.joint(name).id
                    qa, va = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
                    data.qpos[qa:qa+7] = model.qpos0[qa:qa+7]; data.qvel[va:va+6] = 0
                reset_objects = True
            if phase >= 2 and not sent_play:
                os.write(master, b"t"); sent_play = True
            # The bridge refreshes hand commands continuously. Re-apply the
            # validated grasp gain every physics step, matching the reference
            # runner, instead of only setting it during initialization.
            for commands in [env.unitree_bridge.left_hand_cmd, env.unitree_bridge.right_hand_cmd]:
                for cmd in commands.motor_cmd:
                    cmd.kp = args.hand_kp
            if phase >= .3:
                env.sim_step()
            else:
                env.unitree_bridge.PublishLowState(env.prepare_obs())
            if step % 4 == 0:
                env.update_viewer(); mujoco.mj_kinematics(model, data)
                contacts = set()
                for contact in data.contact:
                    if block_geom in (contact.geom1, contact.geom2):
                        other = contact.geom2 if contact.geom1 == block_geom else contact.geom1
                        name = model.body(int(model.geom_bodyid[other])).name
                        if name.startswith("right_hand_thumb"): contacts.add("thumb")
                        if name.startswith(("right_hand_middle", "right_hand_index")): contacts.add("fingers")
                rotation = data.xmat[wrist].reshape(3, 3)
                center = data.xpos[wrist] + rotation @ np.array([.125, .045, 0])
                block_pos = data.xpos[block].copy()
                if initial_block_z is None and sent_play: initial_block_z = float(block_pos[2])
                row = [t, *block_pos, *center, *data.qpos[:3], len(contacts)]
                states.append(row); qpos.append(data.qpos.copy()); qvel.append(data.qvel.copy())
                sample = {
                    "sim_time_s": round(t, 4), "phase_time_s": round(phase, 4),
                    "block_world_m": block_pos.round(6).tolist(),
                    "grasp_center_world_m": center.round(6).tolist(),
                    "block_minus_grasp_center_m": (block_pos-center).round(6).tolist(),
                    "distance_m": round(float(np.linalg.norm(block_pos-center)), 6),
                    "root_world_m": data.qpos[:3].round(6).tolist(),
                    "contact_groups": sorted(contacts),
                    "block_lift_m": round(float(block_pos[2]-(initial_block_z or block_pos[2])), 6),
                    "controller_ready": ready_at is not None, "playing": sent_play,
                }
                telemetry.write(json.dumps(sample) + "\n")
            if phase >= 12.8: break
            next_tick += env.sim_dt
            time.sleep(max(0, next_tick-time.monotonic()))
    finally:
        child.terminate()
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
        os.close(master); controller_log.close(); telemetry.close(); runner.close()
    np.savez(args.output_dir / "simulation.npz", states=states, qpos=qpos, qvel=qvel, fps=50.)
    array = np.asarray(states)
    episode = array[array[:, 0] >= ready_at+2] if ready_at is not None else array
    initial_z = float(episode[0, 3])
    result = {
        "arms_at_sides": args.arms_at_sides, "root_position_assistance": False,
        "elastic_band_enabled": False, "hand_kp": args.hand_kp,
        "robot_initial_x": args.robot_x, "robot_initial_y": args.robot_y,
        "controller_ready_at": ready_at, "scene": config["ROBOT_SCENE"],
        "block_lift_max_m": float(episode[:, 3].max()-initial_z),
        "block_lift_final_m": float(array[-1, 3]-initial_z),
        "minimum_hand_block_distance_m": float(np.min(np.linalg.norm(episode[:,1:4]-episode[:,4:7], axis=1))),
        "final_contacts": int(array[-1, -1]), "minimum_root_height_m": float(array[:, 9].min()),
        "final_block_position": array[-1,1:4].tolist(), "final_root_position": array[-1,7:10].tolist(),
    }
    result["success"] = result["block_lift_final_m"] > .08 and result["final_contacts"] == 2
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
