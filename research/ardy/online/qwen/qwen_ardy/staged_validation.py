#!/usr/bin/env python3
"""Five-stage, single-MuJoCo-process validation of SONIC hot references."""
from __future__ import annotations

import argparse
import json
import os
import pty
import select
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

from .hand_controller import ContactDrivenHand, OPEN
from .ik_reference import publish_cartesian_correction
from .alignment import check_pregrasp
from .reference_runtime import publish_segment, submit_switch, switch_status
from .stages import STAGE_PROMPTS, Stage


PLAN = [
    (Stage.START, 0, 80),
    (Stage.RAISE_HAND, 80, 230),
    # Frame 280 is the calibrated pre-grasp pose.  Continuing the body-only
    # approach to frame 320 overshoots it and makes a later Cartesian repair
    # unnecessarily large.  The hand remains open throughout this segment.
    (Stage.LOWER_TO_BLOCK, 230, 280),
    (Stage.GRASP, 280, 380),
    (Stage.LIFT_WITH_BLOCK, 380, 600),
]


def contacts(model, data, block_geom):
    found = set()
    for contact in data.contact:
        if block_geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == block_geom else contact.geom1
        name = model.body(int(model.geom_bodyid[other])).name
        if name.startswith("right_hand_thumb"): found.add("thumb")
        if name.startswith(("right_hand_middle", "right_hand_index")): found.add("fingers")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--local-qwen-model", type=Path,
                        default=Path(str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "workspace/ljy/models/Qwen3-VL-2B-Instruct")))
    parser.add_argument("--qwen-device", default="musa:7")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.reference_root.mkdir(parents=True, exist_ok=True)
    decisions = args.output_dir / "stage_decisions.jsonl"
    for index, (stage, begin, end) in enumerate(PLAN):
        publish_segment(args.source, args.reference_root, f"{index:02d}_{stage.value}", begin, end,
                        STAGE_PROMPTS[stage], {"source": "live_mujoco", "numeric_in_prompt": False})

    sonic_root = Path(os.environ.get("SONIC_ROOT", str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "GR00T-WholeBodyControl")))
    deploy = Path(os.environ.get("SONIC_DEPLOY", str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "workspace/ljy/sonic_hot_reload/gear_sonic_deploy")))
    os.chdir(sonic_root)
    config = SimLoopConfig().load_wbc_yaml(); config["ROBOT_SCENE"] = args.scene
    config["ENABLE_ELASTIC_BAND"] = False
    runner = BaseSimulator(config, onscreen=args.viewer); env = runner.sim_env; env.elastic_band = None
    env.mj_data.qpos[:7] = env.mj_model.qpos0[:7]
    env.mj_data.qpos[:2] = [.24, -.03]
    config["DEFAULT_DOF_ANGLES"] = np.array(config["DEFAULT_DOF_ANGLES"]).copy()
    side_pose = {"left_shoulder_roll_joint": .15, "right_shoulder_roll_joint": -.15,
                 "left_elbow_joint": 1.4, "right_elbow_joint": 1.4}
    for name, value in side_pose.items():
        index = int(np.flatnonzero(env.body_joint_index == env.mj_model.joint(name).id)[0])
        config["DEFAULT_DOF_ANGLES"][index] = value
    env.mj_data.qpos[env.body_joint_index + env.qpos_offset - 1] = config["DEFAULT_DOF_ANGLES"]
    for index, motor in enumerate(env.unitree_bridge.low_cmd.motor_cmd[:env.num_body_dof]):
        motor.q = float(config["DEFAULT_DOF_ANGLES"][index])
        motor.kp, motor.kd = float(config["MOTOR_KP"][index]), float(config["MOTOR_KD"][index])
    mujoco.mj_forward(env.mj_model, env.mj_data)
    model, data = env.mj_model, env.mj_data
    block = model.body("wood_block").id; block_geom = model.geom("wood_block_geom").id
    wrist = model.body("right_wrist_yaw_link").id
    renderer = mujoco.Renderer(model, 480, 640)

    qwen_log = (args.output_dir / "local_qwen.log").open("w")
    qwen = subprocess.Popen([
        os.sys.executable, "-m", "qwen_ardy.local_qwen_server",
        "--model", str(args.local_qwen_model), "--device", args.qwen_device,
        "--scene", str((sonic_root / args.scene).resolve()),
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=qwen_log,
       text=True, bufsize=1, cwd=args.output_dir)
    ready = json.loads(qwen.stdout.readline())
    if ready.get("event") != "ready":
        raise RuntimeError(f"local Qwen failed to start: {ready}")

    master, slave = pty.openpty()
    command = [str(deploy / "target/release/g1_deploy_onnx_ref"), "lo",
               str(deploy / "policy/release/model_decoder.onnx"), str(args.reference_root),
               "--obs-config", str(deploy / "policy/release/observation_config.yaml"),
               "--encoder-file", str(deploy / "policy/release/model_encoder.onnx"),
               "--input-type", "keyboard", "--policy-precision", "32", "--disable-crc-check"]
    log_path = args.output_dir / "controller.log"; log = log_path.open("w")
    child = subprocess.Popen(command, stdin=slave, stdout=log, stderr=subprocess.STDOUT, cwd=deploy)
    os.close(slave)
    hand = ContactDrivenHand(rate_per_second=.8)
    stage_index = 0; started = playing = False; ready_at = None; ready_wall = None; stage_play_at = None
    stable_contact = 0.; initial_block_z = None; samples = []; pending_switch = None
    pending_plan = None; request_id = 0; correction_count = 0
    correction_joint_target = None; correction_motor_indexes = None
    pending_correction_target = None; correction_start = None; correction_started_at = None
    active_reference_name = "00_start"
    next_tick = time.monotonic()
    try:
        for step in range(int(90 / env.sim_dt)):
            text = log_path.read_text(errors="replace")
            if pending_plan is not None and select.select([qwen.stdout], [], [], 0)[0]:
                response = json.loads(qwen.stdout.readline())
                plan = pending_plan; pending_plan = None
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "local Qwen failed"))
                hard_gate = plan["hard_gate"]
                record = {"from": plan["from_stage"].value,
                          "to": plan["next_stage"].value,
                          "qwen": response["decision"],
                          "hard_alignment_gate": hard_gate.__dict__,
                          "image": str(plan["image"]),
                          "state_source": "live_mujoco"}
                with decisions.open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if plan["from_stage"] == Stage.LOWER_TO_BLOCK and not hard_gate.aligned:
                    correction_count += 1
                    if correction_count > 4:
                        raise RuntimeError(
                            f"alignment failed after 4 corrections: {hard_gate.__dict__}")
                    correction_name = f"02_adjust_{correction_count:02d}_{hard_gate.direction}"
                    correction, pending_correction_target, correction_motor_indexes = publish_cartesian_correction(
                        args.reference_root / active_reference_name,
                        args.reference_root, correction_name,
                        model, data, env.body_joint_index, data.xpos[block].copy(),
                        response["decision"].get(
                            "next_text", "carefully align the open right hand with the block"))
                    submit_switch(args.reference_root, correction)
                    pending_switch = (stage_index, correction)
                else:
                    submit_switch(args.reference_root, plan["segment"])
                    pending_switch = (plan["next_index"], plan["segment"])
            if pending_switch is not None:
                pending_index, pending_segment = pending_switch
                if switch_status(args.reference_root, pending_segment):
                    stage_index = pending_index
                    active_reference_name = pending_segment.name
                    pending_switch = None
                    if pending_segment.name.startswith("02_adjust_"):
                        correction_joint_target = pending_correction_target
                        correction_start = np.array([
                            data.qpos[model.jnt_qposadr[env.body_joint_index[i]]]
                            for i in range(len(env.body_joint_index))])
                        correction_started_at = data.time
                        pending_correction_target = None
                    if PLAN[stage_index][0] == Stage.LIFT_WITH_BLOCK:
                        correction_joint_target = None
                        correction_motor_indexes = None
                    os.write(master, b"t"); stage_play_at = data.time
            if ready_at is None and "Init Done" in text:
                ready_at = data.time; ready_wall = time.monotonic()
            if ready_at is not None and not started:
                os.write(master, b"]"); started = True
            if started and not playing and time.monotonic() - ready_wall > 1.5:
                os.write(master, b"t"); playing = True; stage_play_at = data.time
                initial_block_z = float(data.xpos[block][2])

            found = contacts(model, data, block_geom)
            current_stage = PLAN[stage_index][0]
            if current_stage == Stage.GRASP and {"thumb", "fingers"} <= found:
                stable_contact += env.sim_dt
            else:
                stable_contact = 0.
            target = hand.update(env.sim_dt, current_stage in (Stage.GRASP, Stage.LIFT_WITH_BLOCK),
                                 "thumb" in found, "fingers" in found)
            for commands in (env.unitree_bridge.left_hand_cmd, env.unitree_bridge.right_hand_cmd):
                for motor in commands.motor_cmd: motor.kp, motor.kd = 6., .1
            for index, motor in enumerate(env.unitree_bridge.right_hand_cmd.motor_cmd):
                motor.q = float(target[index])
            # A live Cartesian constraint is enforced only during alignment and
            # grasp.  SONIC continues to control balance and every other joint;
            # the hand stays open until the hard gate explicitly admits GRASP.
            if correction_joint_target is not None and current_stage in (
                    Stage.LOWER_TO_BLOCK, Stage.GRASP):
                for index in correction_motor_indexes:
                    elapsed = max(0., data.time - correction_started_at)
                    u = min(1., elapsed / 2.5)
                    u = u * u * (3. - 2. * u)
                    value = ((1. - u) * correction_start[index]
                             + u * correction_joint_target[index])
                    motor = env.unitree_bridge.low_cmd.motor_cmd[index]
                    motor.q = float(value)
                    motor.kp = max(float(motor.kp), 45.)
                    motor.kd = max(float(motor.kd), 1.5)

            if pending_plan is not None:
                # True stage pause: keep SONIC's safety heartbeat alive without
                # advancing MuJoCo time, qpos, qvel, contacts, or the block.
                env.unitree_bridge.PublishLowState(env.prepare_obs())
            elif ready_wall is not None and time.monotonic() - ready_wall > .3:
                env.sim_step()
            else: env.unitree_bridge.PublishLowState(env.prepare_obs())
            mujoco.mj_kinematics(model, data)
            if step % 4 == 0:
                rotation = data.xmat[wrist].reshape(3, 3)
                center = data.xpos[wrist] + rotation @ np.array([.125, .045, 0])
                samples.append([data.time, stage_index, *data.xpos[block], *center, len(found)])
                if args.viewer: env.update_viewer()

            complete = f"{active_reference_name} completed" in text
            grasp_ready = current_stage == Stage.GRASP and stable_contact >= .25
            if (playing and pending_switch is None and pending_plan is None and
                    stage_index < len(PLAN) - 1 and (complete or grasp_ready)):
                image_path = args.output_dir / f"stage_{stage_index:02d}_{current_stage.value}.png"
                rotation = data.xmat[wrist].reshape(3, 3)
                center = data.xpos[wrist] + rotation @ np.array([.125, .045, 0])
                delta = (data.xpos[block] - center).round(6)
                next_stage = PLAN[stage_index + 1][0]
                next_index = stage_index + 1
                segment = args.reference_root / f"{next_index:02d}_{next_stage.value}"
                hard_gate = check_pregrasp(delta, bool(found))
                request = {"id": request_id, "image": str(image_path),
                           "stage": current_stage.value, "qpos": data.qpos.tolist(),
                           "qvel": data.qvel.tolist(), "state": {
                               "block_world_m": data.xpos[block].round(6).tolist(),
                               "grasp_center_world_m": center.round(6).tolist(),
                               "block_minus_grasp_center_m": delta.tolist(),
                               "contact_groups": sorted(found),
                               "hard_alignment_gate": hard_gate.__dict__,
                           }}
                qwen.stdin.write(json.dumps(request) + "\n"); qwen.stdin.flush()
                pending_plan = {"from_stage": current_stage, "next_stage": next_stage,
                                "next_index": next_index, "segment": segment,
                                "image": image_path, "hard_gate": hard_gate}
                request_id += 1
            if stage_index == len(PLAN) - 1 and pending_switch is None and complete:
                break
            next_tick += env.sim_dt; time.sleep(max(0, next_tick - time.monotonic()))
    finally:
        child.terminate()
        try: child.wait(10)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
        qwen.terminate()
        try: qwen.wait(10)
        except subprocess.TimeoutExpired: qwen.kill(); qwen.wait()
        os.close(master); log.close(); qwen_log.close()
    renderer.close(); runner.close()
    array = np.asarray(samples)
    baseline_z = initial_block_z if initial_block_z is not None else float(array[0, 4])
    result = {"stages_reached": int(array[:, 1].max()) + 1,
              "controller_ready": ready_at is not None,
              "block_lift_final_m": float(array[-1, 4] - baseline_z),
              "block_lift_max_m": float(array[:, 4].max() - baseline_z),
              "final_contacts": int(array[-1, -1]),
              "single_mujoco_process": True,
              "timed_hand_trajectory": False}
    result["success"] = result["stages_reached"] == 5 and result["block_lift_final_m"] > .08 and result["final_contacts"] == 2
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez(args.output_dir / "simulation.npz", samples=array)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
