#!/usr/bin/env python3
"""Method B: Qwen-qualified prompt + real ARDY generation for every stage."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pty
import select
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

from .ardy_visual_constraint import build_constraints
from .hand_controller import ContactDrivenHand
from .reference_runtime import publish_segment, submit_switch, switch_status


GRASP_OFFSET = np.array([.125, .045, 0.0])
# Calibrated from the successful fuyuhan physical demo at its aligned frame.
DEMO_GRASP_OFFSET = np.array([-.00532856, -.00433942, .01452346])
STAGES = [
    {"name": "approach", "duration": 5.0,
     "prompt": "A person takes a small step forward and carefully reaches with the open right hand.",
     "target": "approach", "close": False, "demo_frames": [0, 125]},
    {"name": "align", "duration": 4.0,
     "prompt": "A person carefully lowers and aligns the open right hand for grasping while staying balanced.",
     "target": "grasp", "close": False, "demo_frames": [100, 200]},
    {"name": "close_and_hold", "duration": 2.0,
     "prompt": "A person holds the right wrist steadily at the grasp position and firmly closes the right hand.",
     "target": "grasp", "close": True, "demo_frames": [150, 200]},
    {"name": "lift", "duration": 4.0,
     "prompt": "A person securely holds the object and slowly lifts the right hand while staying balanced.",
     "target": "lift", "close": True, "demo_frames": [200, 300]},
]


def json_line(stream):
    while True:
        line = stream.readline()
        if not line:
            raise RuntimeError("worker exited")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


def block_contacts(model, data, block_geom):
    found = set()
    for contact in data.contact:
        if block_geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == block_geom else contact.geom1
        name = model.body(int(model.geom_bodyid[other])).name
        if name.startswith("right_hand_thumb"):
            found.add("thumb")
        if name.startswith(("right_hand_middle", "right_hand_index")):
            found.add("fingers")
    return found


def target_for(stage, block_position, current_center):
    grasp = block_position + DEMO_GRASP_OFFSET
    if stage["target"] == "current":
        return current_center.copy()
    if stage["target"] == "approach":
        return grasp + np.array([0., 0., .16])
    if stage["target"] == "lift":
        return grasp + np.array([0., 0., .16])
    return grasp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--ardy-repo", type=Path, required=True)
    p.add_argument("--qwen-model", type=Path, required=True)
    p.add_argument("--qwen-device", default="auto")
    p.add_argument("--ardy-gpu", default="6",
                   help="Physical MUSA GPU exposed exclusively to the ARDY worker.")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--seed-base", type=int, default=20)
    p.add_argument("--constraint-weight", type=float, default=2.0)
    a = p.parse_args()
    a.source = a.source.resolve(); a.scene = a.scene.resolve()
    a.output_dir = a.output_dir.resolve(); a.reference_root = a.reference_root.resolve()
    a.output_dir.mkdir(parents=True, exist_ok=False); a.reference_root.mkdir(parents=True, exist_ok=True)
    events_path = a.output_dir / "stage_events.jsonl"

    sonic_root = Path(os.environ.get("SONIC_ROOT", str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "GR00T-WholeBodyControl")))
    deploy = Path(os.environ.get("SONIC_DEPLOY", str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "workspace/ljy/sonic_hot_reload/gear_sonic_deploy")))
    initial = publish_segment(a.source, a.reference_root, "000_initial", 0, 300,
                              "A person stands balanced with both arms relaxed at their sides.",
                              {"type": "stable_startup_window"})

    os.chdir(sonic_root)
    config = SimLoopConfig().load_wbc_yaml(); config["ROBOT_SCENE"] = str(a.scene)
    config["ENABLE_ELASTIC_BAND"] = False
    runner = BaseSimulator(config, onscreen=a.viewer); env = runner.sim_env; env.elastic_band = None
    env.mj_data.qpos[:7] = env.mj_model.qpos0[:7]; env.mj_data.qpos[:2] = [.24, -.03]
    config["DEFAULT_DOF_ANGLES"] = np.array(config["DEFAULT_DOF_ANGLES"]).copy()
    for name, value in {"left_shoulder_roll_joint": .15, "right_shoulder_roll_joint": -.15,
                        "left_elbow_joint": 1.4, "right_elbow_joint": 1.4}.items():
        index = int(np.flatnonzero(env.body_joint_index == env.mj_model.joint(name).id)[0])
        config["DEFAULT_DOF_ANGLES"][index] = value
    env.mj_data.qpos[env.body_joint_index + env.qpos_offset - 1] = config["DEFAULT_DOF_ANGLES"]
    for index, motor in enumerate(env.unitree_bridge.low_cmd.motor_cmd[:env.num_body_dof]):
        motor.q = float(config["DEFAULT_DOF_ANGLES"][index])
        motor.kp = float(config["MOTOR_KP"][index]); motor.kd = float(config["MOTOR_KD"][index])
    mujoco.mj_forward(env.mj_model, env.mj_data)
    model, data = env.mj_model, env.mj_data
    wrist = model.body("right_wrist_yaw_link").id
    block = model.body("wood_block").id; block_geom = model.geom("wood_block_geom").id

    source_names = [node.get("name") for node in
                    ET.parse(a.ardy_repo / "ardy/assets/skeletons/g1skel34/xml/g1.xml").findall(
                        ".//worldbody//joint") if node.get("type") != "free"]
    source_qadr = np.array([model.joint(name).qposadr[0] for name in source_names])
    demo_qpos = np.loadtxt(a.source / "ardy_motion.csv", delimiter=",", ndmin=2)
    # Immutable task target.  It is never written after this snapshot.
    block_target = data.xpos[block].copy(); initial_block_z = float(block_target[2])

    qlog = (a.output_dir / "qwen.log").open("w")
    qwen = subprocess.Popen([
        sys.executable, "-m", "qwen_ardy.local_qwen_server", "--model", str(a.qwen_model),
        "--device", a.qwen_device, "--scene", str(a.scene)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=qlog,
        text=True, bufsize=1, cwd=a.output_dir)
    qready = json_line(qwen.stdout)
    if qready.get("event") != "ready": raise RuntimeError(qready)

    alog = (a.output_dir / "ardy.log").open("w")
    ardy_env = os.environ.copy()
    ardy_env["CUDA_VISIBLE_DEVICES"] = a.ardy_gpu
    ardy_env["MUSA_VISIBLE_DEVICES"] = a.ardy_gpu
    # Keep ARDY fully local and reproducible.  Its LLM2Vec wrapper resolves
    # released model IDs underneath TEXT_ENCODERS_DIR when this is set;
    # without it Transformers attempts a Hugging Face network request.
    ardy_env["TEXT_ENCODERS_DIR"] = str(a.ardy_repo / "text_encoders")
    ardy_env["TEXT_ENCODER_MODE"] = "local"
    ardy_env["HF_HUB_OFFLINE"] = "1"
    ardy = subprocess.Popen([
        sys.executable, "-m", "qwen_ardy.local_ardy_server", "--ardy-repo", str(a.ardy_repo),
        "--checkpoints", str(a.ardy_repo / "checkpoints"), "--converter",
        str(sonic_root / "gear_sonic_deploy/reference/convert_ardy.py"), "--device", "cuda:0"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=alog,
        text=True, bufsize=1, cwd=a.output_dir, env=ardy_env)
    aready = json_line(ardy.stdout)
    if aready.get("event") != "ready": raise RuntimeError(aready)

    master, slave = pty.openpty(); clog_path = a.output_dir / "controller.log"; clog = clog_path.open("w")
    controller = subprocess.Popen([
        str(deploy / "target/release/g1_deploy_onnx_ref"), "lo",
        str(deploy / "policy/release/model_decoder.onnx"), str(a.reference_root),
        "--obs-config", str(deploy / "policy/release/observation_config.yaml"),
        "--encoder-file", str(deploy / "policy/release/model_encoder.onnx"),
        "--input-type", "keyboard", "--policy-precision", "32", "--disable-crc-check"],
        stdin=slave, stdout=clog, stderr=subprocess.STDOUT, cwd=deploy)
    os.close(slave)

    hand = ContactDrivenHand(rate_per_second=.8)
    started = playing = False; ready_step = None; active_name = initial.name
    loop_step = 0; objects_reset = False
    stage_index = -1; phase = "initial"; pending_reference = None
    request_id = 0; samples = []; next_tick = time.monotonic(); final_complete = False
    completion_baseline = 0; attempts = [0] * len(STAGES)
    executor = ThreadPoolExecutor(max_workers=1); constraint_future = None; pending_plan = None

    def write_event(value):
        with events_path.open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    def request_qwen(index):
        nonlocal request_id, phase
        stage = STAGES[index]; attempts[index] += 1
        rotation = data.xmat[wrist].reshape(3, 3); center = data.xpos[wrist] + rotation @ GRASP_OFFSET
        image = a.output_dir / f"stage_{index:02d}_{stage['name']}.png"
        request = {"id": request_id, "mode": "rewrite_fixed_stage", "image": str(image),
                   "stage": stage["name"], "fixed_prompt": stage["prompt"],
                   "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(), "state": {
                       "block_world_m": block_target.tolist(),
                       "grasp_center_world_m": center.tolist(),
                       "contact_groups": sorted(block_contacts(model, data, block_geom))}}
        qwen.stdin.write(json.dumps(request) + "\n"); qwen.stdin.flush()
        request_id += 1; phase = "qwen"

    try:
        while not final_complete:
            text = clog_path.read_text(errors="replace")
            if ready_step is None and "Init Done" in text: ready_step = loop_step
            control_phase = ((loop_step - ready_step) * env.sim_dt
                             if ready_step is not None else -100.)
            if control_phase >= .1 and not started: os.write(master, b"]"); started = True
            if control_phase >= 1.5 and not objects_reset:
                for name in ("wood_block_free", "red_block_free"):
                    joint = model.joint(name).id
                    qadr = model.jnt_qposadr[joint]; dadr = model.jnt_dofadr[joint]
                    data.qpos[qadr:qadr+7] = model.qpos0[qadr:qadr+7]
                    data.qvel[dadr:dadr+6] = 0.
                objects_reset = True
            if control_phase >= 2.0 and not playing:
                # Do not replay the demo prefix here.  It used to move the
                # robot before the first live constraint was built, making
                # the measured boundary pose (and occasionally balance)
                # nondeterministic.  Keep SONIC paused at reference frame 0
                # while Qwen and ARDY prepare the first real stage.  The
                # switch branch below starts playback atomically only after
                # that reference is ready.
                playing = True
                stage_index = 0
                request_qwen(stage_index)

            completion_count = text.count(f"{active_name} completed")
            if phase == "executing" and completion_count > completion_baseline:
                completion_baseline = completion_count
                if stage_index == len(STAGES) - 1:
                    final_complete = True
                else:
                    if stage_index >= 0 and STAGES[stage_index]["name"] == "align":
                        rotation = data.xmat[wrist].reshape(3, 3)
                        center = data.xpos[wrist] + rotation @ GRASP_OFFSET
                        error = float(np.linalg.norm(center - (data.xpos[block] + DEMO_GRASP_OFFSET)))
                        write_event({"stage": "align_gate", "attempt": attempts[stage_index],
                                     "error_m": error, "grasp_center_world_m": center.tolist(),
                                     "block_world_m": data.xpos[block].tolist(),
                                     "root_world_m": data.qpos[:3].tolist()})
                        if error > .035:
                            if attempts[stage_index] >= 4:
                                raise RuntimeError(f"alignment gate failed after 4 attempts: {error:.4f} m")
                            request_qwen(stage_index)
                            continue
                    stage_index += 1; request_qwen(stage_index)

            if phase == "qwen" and select.select([qwen.stdout], [], [], 0)[0]:
                response = json_line(qwen.stdout)
                if not response.get("ok"): raise RuntimeError(response)
                stage = STAGES[stage_index]; decision = response["decision"]
                rotation = data.xmat[wrist].reshape(3, 3)
                center = data.xpos[wrist] + rotation @ GRASP_OFFSET
                live_block = data.xpos[block].copy()
                endpoint = target_for(stage, live_block, center)
                demo_begin, demo_end = stage["demo_frames"]
                root_delta = (demo_qpos[min(demo_end - 1, len(demo_qpos) - 1), :2]
                              - demo_qpos[demo_begin, :2])
                # SONIC treats the reference root as a policy condition rather
                # than tracking its translation one-for-one.  Solve the arm in
                # the measured live world frame; subtracting the nominal demo
                # root delta produced an anatomically extreme arm posture.
                ik_endpoint = endpoint.copy()
                attempt = attempts[stage_index]
                job = a.output_dir / f"stage_{stage_index:02d}_{stage['name']}_try{attempt:02d}"; job.mkdir()
                constraints = job / "constraints.json"
                anchor = np.r_[data.qpos[:7], data.qpos[source_qadr]].tolist()
                request = {"id": stage_index, "seed": a.seed_base + stage_index,
                           "prompt": decision["ardy_prompt"], "duration_s": stage["duration"],
                           "constraints": str(constraints), "output_stem": str(job / "motion"),
                           "reference_dir": str(job / f"reference_{stage_index:02d}_{stage['name']}_try{attempt:02d}"),
                           "anchor_qpos": anchor,
                           "text_weight": 2., "constraint_weight": a.constraint_weight,
                           "scaffold_qpos_csv": str(a.source / "ardy_motion.csv"),
                           "scaffold_frames": stage["demo_frames"], "residual_scale": .01,
                           "endpoint_projection_scale": .02}
                frozen_qpos = data.qpos.copy()
                seed_qpos = frozen_qpos.copy()
                demo_endpoint = min(stage["demo_frames"][1] - 1, len(demo_qpos) - 1)
                seed_qpos[source_qadr] = demo_qpos[demo_endpoint, 7:7+len(source_qadr)]
                pending_plan = {"stage": stage, "decision": decision, "endpoint": endpoint,
                                "ik_endpoint": ik_endpoint, "root_delta": root_delta,
                                "request": request, "pause_qpos": frozen_qpos}
                constraint_future = executor.submit(
                    build_constraints, a.ardy_repo, a.scene, frozen_qpos,
                    ik_endpoint, constraints, stage["duration"], 25, seed_qpos)
                phase = "constraint"

            if phase == "constraint" and constraint_future.done():
                constraint_meta = constraint_future.result()
                solved = np.asarray(constraint_meta["solved_mujoco_qpos"], dtype=float)
                pending_plan["request"]["target_qpos"] = np.r_[
                    solved[:7], solved[source_qadr]].tolist()
                ardy.stdin.write(json.dumps(pending_plan["request"]) + "\n"); ardy.stdin.flush()
                write_event({"stage": pending_plan["stage"]["name"],
                             "fixed_prompt": pending_plan["stage"]["prompt"],
                             "qwen": pending_plan["decision"],
                             "endpoint_world_m": pending_plan["endpoint"].tolist(),
                             "ik_endpoint_before_root_motion_m": pending_plan["ik_endpoint"].tolist(),
                             "scaffold_root_delta_xy_m": pending_plan["root_delta"].tolist(),
                             "attempt": attempts[stage_index],
                             "constraint": constraint_meta,
                             "pause_qpos": pending_plan["pause_qpos"].tolist()})
                constraint_future = None; pending_plan = None; phase = "ardy"

            if phase == "ardy" and select.select([ardy.stdout], [], [], 0)[0]:
                response = json_line(ardy.stdout)
                if not response.get("ok"): raise RuntimeError(response)
                pending_reference = Path(response["reference_dir"])
                submit_switch(a.reference_root, pending_reference); phase = "switch"

            if phase == "switch" and switch_status(a.reference_root, pending_reference):
                active_name = pending_reference.name; os.write(master, b"t")
                completion_baseline = text.count(f"{active_name} completed")
                phase = "executing"; pending_reference = None

            found = block_contacts(model, data, block_geom)
            close = stage_index >= 0 and STAGES[stage_index]["close"]
            hand_target = hand.update(env.sim_dt, close, "thumb" in found, "fingers" in found)
            for commands in (env.unitree_bridge.left_hand_cmd, env.unitree_bridge.right_hand_cmd):
                for motor in commands.motor_cmd: motor.kp, motor.kd = 6., .1
            for index, motor in enumerate(env.unitree_bridge.right_hand_cmd.motor_cmd):
                motor.q = float(hand_target[index])

            # During Qwen/ARDY/switch latency, preserve qpos/qvel/block state and
            # only keep SONIC's LowState heartbeat alive.
            if phase in ("qwen", "constraint", "ardy", "switch"):
                env.unitree_bridge.PublishLowState(env.prepare_obs())
            elif control_phase >= .3:
                env.sim_step()
            else:
                env.unitree_bridge.PublishLowState(env.prepare_obs())
            mujoco.mj_kinematics(model, data)
            if len(samples) == 0 or data.time > samples[-1][0] + .04:
                rotation = data.xmat[wrist].reshape(3, 3); center = data.xpos[wrist] + rotation @ GRASP_OFFSET
                samples.append([data.time, stage_index, *data.xpos[block], *center, len(found)])
            if a.viewer: env.update_viewer()
            loop_step += 1
            next_tick += env.sim_dt; time.sleep(max(0, next_tick - time.monotonic()))
    finally:
        for process in (controller, qwen, ardy):
            process.terminate()
            try: process.wait(10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        os.close(master); clog.close(); qlog.close(); alog.close(); runner.close()
        executor.shutdown(wait=False, cancel_futures=True)

    array = np.asarray(samples)
    result = {"method": "B_real_ardy_per_stage", "stages_completed": len(STAGES),
              "single_mujoco_process": True, "block_target_read_once": block_target.tolist(),
              "block_lift_max_m": float(array[:, 4].max() - initial_block_z),
              "block_lift_final_m": float(array[-1, 4] - initial_block_z),
              "minimum_hand_block_distance_m": float(np.linalg.norm(array[:, 2:5]-array[:, 5:8], axis=1).min()),
              "final_contacts": int(array[-1, -1])}
    result["success"] = result["block_lift_final_m"] > .08 and result["final_contacts"] == 2
    np.savez_compressed(a.output_dir / "simulation.npz", samples=array)
    (a.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
