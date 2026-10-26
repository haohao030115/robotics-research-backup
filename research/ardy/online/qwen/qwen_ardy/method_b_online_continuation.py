#!/usr/bin/env python3
"""True online Method B.

Only the first stage exists before simulation starts.  At every stage boundary
SONIC actively holds its final reference frame while MuJoCo keeps stepping.
The live image/state is then sent to Qwen, one constrained ARDY segment is
generated, and that segment is attached to the existing SONIC root/heading
context before playback resumes.
"""
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

from .ardy_visual_constraint import build_continuity_constraints
from .build_method_b_reference import BASE_PROMPT, stage_constraints, stitch_start
from .hand_controller import ContactDrivenHand
from .reference_runtime import hold_status, publish_segment, submit_switch, switch_status


GRASP_OFFSET = np.array([.125, .045, 0.0])
DEMO_GRASP_OFFSET = np.array([-.00532856, -.00433942, .01452346])
STAGES = [
    ("approach", 0, 75, 3.0,
     BASE_PROMPT + " In this segment, begin balanced and approach carefully with the open right hand.", False),
    ("align", 75, 150, 3.0,
     BASE_PROMPT + " In this segment, carefully align the open right hand while staying balanced.", False),
    ("close_and_hold", 150, 200, 2.0,
     BASE_PROMPT + " In this segment, hold the right wrist steadily at the grasp position.", True),
    ("lift", 200, 300, 4.0,
     BASE_PROMPT + " In this segment, slowly lift the right hand while staying balanced.", True),
]


def read_json_line(stream):
    while True:
        line = stream.readline()
        if not line:
            raise RuntimeError("model worker exited")
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


def write_chained_constraints(ardy_repo: Path, scene: Path, items: list[dict],
                              constraint_qpos: np.ndarray | None, duration: float,
                              constraints_path: Path) -> None:
    """Build the expensive ARDY FK constraint off the real-time sim thread."""
    if constraint_qpos is not None:
        continuity_path = constraints_path.with_name("constraint_chain_start.json")
        build_continuity_constraints(ardy_repo, scene, constraint_qpos,
                                     continuity_path, duration)
        items[0] = json.loads(continuity_path.read_text())[0]
    constraints_path.write_text(json.dumps(items, indent=2) + "\n")


def convert_cumulative_stage(output_dir: Path, stage_index: int, converter: Path,
                             ardy_repo: Path, frame_repeat: int = 1) -> Path:
    """Convert the generated prefix in one root frame, then publish only its last stage."""
    parts = []
    prompts = []
    for index, (name, *_rest) in enumerate(STAGES[:stage_index + 1]):
        part = np.load(output_dir / f"stage_{index:02d}_{name}" / "motion.npz")
        parts.append({key: np.asarray(part[key]) for key in part.files})
        prompts.append(str(part["text"]))
        part.close()
    combined = {}
    for key in parts[0]:
        if key == "fps":
            combined[key] = parts[0][key]
        elif key == "text":
            combined[key] = np.asarray(" | ".join(prompts))
        else:
            combined[key] = np.concatenate([part[key] for part in parts], axis=0)
    job = output_dir / f"stage_{stage_index:02d}_{STAGES[stage_index][0]}"
    cumulative_npz = job / "cumulative_prefix.npz"
    np.savez(cumulative_npz, **combined)
    # convert_ardy intentionally prefers a matching qpos CSV.  This projected
    # demo-manifold qpos (not the raw diffusion NPZ alone) is what made the
    # verified Method B reference grasp accurately.
    projected_qpos = [np.loadtxt(
        output_dir / f"stage_{index:02d}_{STAGES[index][0]}" / "motion.csv",
        delimiter=",", ndmin=2) for index in range(stage_index + 1)]
    np.savetxt(cumulative_npz.with_suffix(".csv"), np.concatenate(projected_qpos), delimiter=",")
    full_reference = job / "cumulative_reference"
    subprocess.run([sys.executable, str(converter), str(cumulative_npz),
                    "--ardy-repo", str(ardy_repo), "--output-dir", str(full_reference)],
                   check=True, stdout=subprocess.DEVNULL)
    lengths = [int(part["posed_joints"].shape[0]) for part in parts]
    source_begin = sum(lengths[:-1])
    source_end = sum(lengths)
    # ARDY samples at 25 Hz while convert_ardy publishes SONIC CSV at 50 Hz.
    # Derive the ratio from the actual converter output rather than hard-code it.
    with (full_reference / "joint_pos.csv").open() as stream:
        output_frames = sum(1 for _ in stream) - 1  # subtract CSV header
    scale = output_frames / source_end
    begin = int(round(source_begin * scale))
    end = int(round(source_end * scale))
    published_root = job / "published"
    return publish_segment(full_reference, published_root,
                           f"reference_{stage_index:02d}_{STAGES[stage_index][0]}",
                           begin, end, STAGES[stage_index][4],
                           {"type": "cumulative_root_context", "prefix_stages": stage_index + 1},
                           frame_repeat=frame_repeat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--ardy-repo", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--qwen-device", default="auto")
    parser.add_argument("--ardy-gpu", default="6")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    for key in ("source", "reference_root", "output_dir", "scene", "ardy_repo", "qwen_model"):
        setattr(args, key, getattr(args, key).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.reference_root.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "stage_events.jsonl"

    sonic_root = Path(os.environ.get("SONIC_ROOT", str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "GR00T-WholeBodyControl")))
    deploy = Path(os.environ.get("SONIC_DEPLOY", str(Path(os.environ.get("RESEARCH_ROOT", "/workspace/group2")) / "workspace/ljy/sonic_hot_reload/gear_sonic_deploy")))
    initial = publish_segment(args.source, args.reference_root, "000_online_hold", 0, 1,
                              "Initial active hold", {"type": "hold"})
    original_constraints = json.loads((args.source / "constraints.json").read_text())
    source_qpos = np.loadtxt(args.source / "ardy_motion.csv", delimiter=",", ndmin=2)

    os.chdir(sonic_root)
    config = SimLoopConfig().load_wbc_yaml()
    config["ROBOT_SCENE"] = str(args.scene)
    config["ENABLE_ELASTIC_BAND"] = False
    # Keep MuJoCo's control context headless/EGL even when a live window is
    # requested.  The native GLFW viewer has proved able to segfault the
    # entire control process on the remote X/VNC stack.  A separate OpenCV
    # monitor below presents frames without owning the physics context.
    runner = BaseSimulator(config, onscreen=False)
    env = runner.sim_env
    env.elastic_band = None
    env.mj_data.qpos[:7] = env.mj_model.qpos0[:7]
    env.mj_data.qpos[:2] = [.24, -.03]
    config["DEFAULT_DOF_ANGLES"] = np.array(config["DEFAULT_DOF_ANGLES"]).copy()
    for name, value in {"left_shoulder_roll_joint": .15, "right_shoulder_roll_joint": -.15,
                        "left_elbow_joint": 1.4, "right_elbow_joint": 1.4}.items():
        index = int(np.flatnonzero(env.body_joint_index == env.mj_model.joint(name).id)[0])
        config["DEFAULT_DOF_ANGLES"][index] = value
    env.mj_data.qpos[env.body_joint_index + env.qpos_offset - 1] = config["DEFAULT_DOF_ANGLES"]
    for index, motor in enumerate(env.unitree_bridge.low_cmd.motor_cmd[:env.num_body_dof]):
        motor.q = float(config["DEFAULT_DOF_ANGLES"][index])
        motor.kp = float(config["MOTOR_KP"][index])
        motor.kd = float(config["MOTOR_KD"][index])
    mujoco.mj_forward(env.mj_model, env.mj_data)
    model, data = env.mj_model, env.mj_data
    viewer = None
    viewer_log = None
    viewer_state = args.output_dir / "live_view_state.npz"
    wrist = model.body("right_wrist_yaw_link").id
    block = model.body("wood_block").id
    block_geom = model.geom("wood_block_geom").id
    source_names = [node.get("name") for node in
                    ET.parse(args.ardy_repo / "ardy/assets/skeletons/g1skel34/xml/g1.xml").findall(
                        ".//worldbody//joint") if node.get("type") != "free"]
    source_qadr = np.array([model.joint(name).qposadr[0] for name in source_names])
    block_target = data.xpos[block].copy()
    initial_block_z = float(block_target[2])

    qlog = (args.output_dir / "qwen.log").open("w")
    qwen = subprocess.Popen([
        sys.executable, "-m", "qwen_ardy.local_qwen_server", "--model", str(args.qwen_model),
        "--device", args.qwen_device, "--scene", str(args.scene)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=qlog,
        text=True, bufsize=1, cwd=args.output_dir)
    if read_json_line(qwen.stdout).get("event") != "ready":
        raise RuntimeError("Qwen failed to start")

    alog = (args.output_dir / "ardy.log").open("w")
    ardy_env = os.environ.copy()
    ardy_env.update(CUDA_VISIBLE_DEVICES=args.ardy_gpu, MUSA_VISIBLE_DEVICES=args.ardy_gpu,
                    TEXT_ENCODERS_DIR=str(args.ardy_repo / "text_encoders"),
                    TEXT_ENCODER_MODE="local", HF_HUB_OFFLINE="1")
    converter = sonic_root / "gear_sonic_deploy/reference/convert_ardy.py"
    ardy = subprocess.Popen([
        sys.executable, "-m", "qwen_ardy.local_ardy_server", "--ardy-repo", str(args.ardy_repo),
        "--checkpoints", str(args.ardy_repo / "checkpoints"), "--converter", str(converter),
        "--device", "cuda:0"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=alog,
        text=True, bufsize=1, cwd=args.output_dir, env=ardy_env)
    if read_json_line(ardy.stdout).get("event") != "ready":
        raise RuntimeError("ARDY failed to start")

    def publish_view_state(stage: int, phase_name: str) -> None:
        temporary = viewer_state.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, qpos=data.qpos.copy(), stage=stage,
                     phase=np.asarray(phase_name), sim_time=float(data.time))
        temporary.replace(viewer_state)

    if args.viewer:
        publish_view_state(-1, "startup")
        viewer_log = (args.output_dir / "viewer.log").open("w")
        viewer = subprocess.Popen([
            sys.executable, "-m", "qwen_ardy.live_viewer",
            "--scene", str(args.scene), "--state-file", str(viewer_state)],
            stdout=viewer_log, stderr=subprocess.STDOUT, cwd=args.output_dir)

    controller_log_path = args.output_dir / "controller.log"
    controller_log = controller_log_path.open("w")

    def launch_controller():
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen([
            str(deploy / "target/release/g1_deploy_onnx_ref"), "lo",
            str(deploy / "policy/release/model_decoder.onnx"), str(args.reference_root),
            "--obs-config", str(deploy / "policy/release/observation_config.yaml"),
            "--encoder-file", str(deploy / "policy/release/model_encoder.onnx"),
            "--input-type", "keyboard", "--policy-precision", "32", "--disable-crc-check"],
            stdin=slave_fd, stdout=controller_log, stderr=subprocess.STDOUT, cwd=deploy)
        os.close(slave_fd)
        return process, master_fd

    controller, master = launch_controller()
    controller_restarts = 0

    hand = ContactDrivenHand(rate_per_second=.8)
    held_hand_target = hand.update(0., False, False, False)
    loop_step = 0
    ready_step = None
    started = False
    objects_reset = False
    phase = "startup"
    stage_index = -1
    request_id = 0
    active_name = initial.name
    pending_reference = None
    previous_constraint_qpos = None
    executor = ThreadPoolExecutor(max_workers=1)
    constraint_future = None
    pending_ardy_request = None
    pending_ardy_event = None
    conversion_future = None
    samples = []
    next_tick = time.monotonic()
    final_complete = False

    def event(value):
        with events_path.open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    def request_next_stage(index):
        nonlocal request_id, phase
        name, begin, end, duration, fixed_prompt, close = STAGES[index]
        rotation = data.xmat[wrist].reshape(3, 3)
        center = data.xpos[wrist] + rotation @ GRASP_OFFSET
        image = args.output_dir / f"stage_{index:02d}_{name}_live.png"
        request = {"id": request_id, "mode": "rewrite_fixed_stage", "image": str(image),
                   "stage": name, "fixed_prompt": fixed_prompt,
                   "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                   "state": {"block_world_m": data.xpos[block].tolist(),
                             "grasp_center_world_m": center.tolist(),
                             "contact_groups": sorted(block_contacts(model, data, block_geom)),
                             "completed": [item[0] for item in STAGES[:index]],
                             "remaining": [item[0] for item in STAGES[index:]]}}
        qwen.stdin.write(json.dumps(request) + "\n")
        qwen.stdin.flush()
        event({"event": "qwen_requested_after_live_hold", "stage": name,
               "sim_time": float(data.time), "image": str(image),
               "live_qpos": data.qpos.tolist(), "hand_target": held_hand_target.tolist()})
        request_id += 1
        phase = "qwen"

    try:
        while not final_complete:
            controller_text = controller_log_path.read_text(errors="replace")
            if controller.poll() is not None:
                if ("Lost LowState data connection" in controller_text
                        and phase == "startup" and controller_restarts < 3):
                    controller_restarts += 1
                    event({"event": "sonic_lowstate_retry",
                           "attempt": controller_restarts,
                           "sim_time": float(data.time)})
                    os.close(master)
                    # Keep the DDS publisher warm before creating a fresh
                    # SONIC subscriber.  Models stay resident in memory.
                    for _ in range(200):
                        env.unitree_bridge.PublishLowState(env.prepare_obs())
                        time.sleep(env.sim_dt)
                    controller, master = launch_controller()
                    ready_step = None
                    started = False
                    continue
                tail = "\n".join(controller_text.splitlines()[-20:])
                raise RuntimeError(
                    f"SONIC controller exited before completion (code={controller.returncode}).\n{tail}")
            if ready_step is None and "Init Done" in controller_text:
                ready_step = loop_step
            control_phase = ((loop_step - ready_step) * env.sim_dt if ready_step is not None else -100.)
            # Give DDS LowState discovery a full second after SONIC reports
            # Init Done.  Starting after only 100 ms intermittently tripped
            # SONIC's safety check on the shared remote host.
            if control_phase >= 1.0 and not started:
                os.write(master, b"]")
                started = True
            if control_phase >= 1.5 and not objects_reset:
                for joint_name in ("wood_block_free", "red_block_free"):
                    joint = model.joint(joint_name).id
                    qadr = model.jnt_qposadr[joint]
                    dadr = model.jnt_dofadr[joint]
                    data.qpos[qadr:qadr+7] = model.qpos0[qadr:qadr+7]
                    data.qvel[dadr:dadr+6] = 0.
                objects_reset = True
                mujoco.mj_forward(model, data)
            if control_phase >= 2.0 and phase == "startup":
                stage_index = 0
                request_next_stage(stage_index)

            if phase == "executing" and hold_status(args.reference_root, active_name):
                event({"event": "stage_held", "stage": STAGES[stage_index][0],
                       "sim_time": float(data.time), "status": hold_status(args.reference_root, active_name)})
                if stage_index == len(STAGES) - 1:
                    final_complete = True
                else:
                    stage_index += 1
                    request_next_stage(stage_index)

            if phase == "qwen" and select.select([qwen.stdout], [], [], 0)[0]:
                response = read_json_line(qwen.stdout)
                if not response.get("ok"):
                    raise RuntimeError(response)
                name, begin, end, duration, fixed_prompt, close = STAGES[stage_index]
                job = args.output_dir / f"stage_{stage_index:02d}_{name}"
                job.mkdir()
                constraints_path = job / "constraints.json"
                items = stage_constraints(original_constraints, begin, end)
                constraint_qpos = None
                if previous_constraint_qpos is not None:
                    # Reproduce the successful fixed-Method-B constraint chain:
                    # the next ARDY sample starts from the prior generated
                    # endpoint projected onto the verified demo manifold.  This
                    # is planning state only; SONIC still aligns playback to
                    # the live held reference/robot context.
                    constraint_qpos = model.qpos0.copy()
                    constraint_qpos[:7] = previous_constraint_qpos[:7]
                    constraint_qpos[source_qadr] = previous_constraint_qpos[7:7+len(source_qadr)]
                prompt = response["decision"]["ardy_prompt"]
                reference = job / f"reference_{stage_index:02d}_{name}"
                pending_ardy_request = {
                    "id": stage_index, "seed": 0, "prompt": prompt,
                    "duration_s": duration, "constraints": str(constraints_path),
                    "output_stem": str(job / "motion"), "reference_dir": str(reference),
                    "text_weight": 2.0, "constraint_weight": 2.0,
                    "preserve_generated_qpos": True}
                pending_ardy_event = {
                    "event": "qwen_completed_constraint_build_started", "stage": name,
                    "sim_time": float(data.time), "fixed_prompt": fixed_prompt,
                    "qwen_prompt": prompt, "qwen": response["decision"],
                    "constraints": str(constraints_path)}
                constraint_future = executor.submit(
                    write_chained_constraints, args.ardy_repo, args.scene, items,
                    constraint_qpos, duration, constraints_path)
                phase = "constraint"

            if phase == "constraint" and constraint_future.done():
                constraint_future.result()
                ardy.stdin.write(json.dumps(pending_ardy_request) + "\n")
                ardy.stdin.flush()
                pending_ardy_event["event"] = "constraint_completed_ardy_requested"
                pending_ardy_event["sim_time_ardy_request"] = float(data.time)
                event(pending_ardy_event)
                constraint_future = None
                pending_ardy_request = None
                pending_ardy_event = None
                phase = "ardy"

            if phase == "ardy" and select.select([ardy.stdout], [], [], 0)[0]:
                response = read_json_line(ardy.stdout)
                if not response.get("ok"):
                    raise RuntimeError(response)
                name, begin, end, duration, fixed_prompt, close = STAGES[stage_index]
                motion_csv = args.output_dir / f"stage_{stage_index:02d}_{name}" / "motion.csv"
                generated_qpos = np.loadtxt(motion_csv, delimiter=",", ndmin=2)
                fixed_stage = source_qpos[begin:end].copy()
                generated_qpos = fixed_stage + .05 * (generated_qpos - fixed_stage)
                generated_qpos[:, 3:7] /= np.maximum(
                    np.linalg.norm(generated_qpos[:, 3:7], axis=1, keepdims=True), 1e-8)
                if previous_constraint_qpos is not None:
                    generated_qpos = stitch_start(generated_qpos, previous_constraint_qpos)
                previous_constraint_qpos = generated_qpos[-1].copy()
                np.savetxt(motion_csv, generated_qpos, delimiter=",")
                conversion_future = executor.submit(
                    convert_cumulative_stage, args.output_dir, stage_index, converter,
                    args.ardy_repo, 1)
                event({"event": "ardy_completed_cumulative_conversion_started",
                       "stage": STAGES[stage_index][0], "sim_time": float(data.time),
                       "ardy_elapsed_s": response["elapsed_s"]})
                phase = "convert"

            if phase == "convert" and conversion_future.done():
                pending_reference = conversion_future.result()
                conversion_future = None
                submit_switch(args.reference_root, pending_reference)
                event({"event": "cumulative_conversion_completed_switch_requested",
                       "stage": STAGES[stage_index][0], "sim_time": float(data.time),
                       "reference": str(pending_reference)})
                phase = "switch"

            if phase == "switch" and switch_status(args.reference_root, pending_reference):
                active_name = pending_reference.name
                os.write(master, b"t")
                event({"event": "continuous_segment_started", "stage": STAGES[stage_index][0],
                       "sim_time": float(data.time), "reference": str(pending_reference),
                       "status": switch_status(args.reference_root, pending_reference)})
                pending_reference = None
                phase = "executing"

            contacts = block_contacts(model, data, block_geom)
            # During Qwen/ARDY latency the last hand command is copied unchanged.
            # It only evolves while the corresponding body reference is playing.
            if phase == "executing":
                close = STAGES[stage_index][5]
                held_hand_target = hand.update(env.sim_dt, close,
                                               "thumb" in contacts, "fingers" in contacts)
            for commands in (env.unitree_bridge.left_hand_cmd, env.unitree_bridge.right_hand_cmd):
                for motor in commands.motor_cmd:
                    motor.kp, motor.kd = 6., .1
            for index, motor in enumerate(env.unitree_bridge.right_hand_cmd.motor_cmd):
                motor.q = float(held_hand_target[index])

            # Keep MuJoCo and the SONIC control loop alive during model inference.
            # The paused controller sees a last-frame-filled reference window and
            # actively holds the body instead of freezing qpos/qvel artificially.
            if control_phase >= .3:
                env.sim_step()
            else:
                env.unitree_bridge.PublishLowState(env.prepare_obs())
            mujoco.mj_kinematics(model, data)
            if not samples or data.time > samples[-1][0] + .04:
                rotation = data.xmat[wrist].reshape(3, 3)
                center = data.xpos[wrist] + rotation @ GRASP_OFFSET
                samples.append([data.time, stage_index, *data.xpos[block], *center, len(contacts),
                                1 if phase == "executing" else 0, float(data.qpos[2])])
            if viewer is not None and loop_step % 10 == 0:
                publish_view_state(stage_index, phase)
            loop_step += 1
            next_tick += env.sim_dt
            time.sleep(max(0., next_tick - time.monotonic()))
    finally:
        for process in (controller, qwen, ardy, viewer):
            if process is None:
                continue
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        os.close(master)
        if viewer_log is not None:
            viewer_log.close()
        controller_log.close(); qlog.close(); alog.close(); runner.close()
        executor.shutdown(wait=False, cancel_futures=True)

    array = np.asarray(samples)
    waits = array[array[:, -2] == 0]
    result = {"method": "B_true_online_final_frame_hold", "stages_completed": len(STAGES),
              "single_mujoco_process": True, "physics_kept_running_during_inference": True,
              "block_target_read_once": block_target.tolist(),
              "block_lift_max_m": float(array[:, 4].max() - initial_block_z),
              "block_lift_final_m": float(array[-1, 4] - initial_block_z),
              "minimum_hand_block_distance_m": float(np.linalg.norm(array[:, 2:5]-array[:, 5:8], axis=1).min()),
              "final_contacts": int(array[-1, -3]),
              "wait_samples": int(len(waits)),
              "minimum_root_height_m": float(array[:, -1].min())}
    result["success"] = result["block_lift_final_m"] > .08 and result["final_contacts"] == 2
    np.savez_compressed(args.output_dir / "simulation.npz", samples=array)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
