#!/usr/bin/env python3
"""Continuous Visual Trace → ARDY → SONIC grasp validation (no fixed stages)."""
from __future__ import annotations

import argparse, json, os, pty, select, subprocess, time, xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

from .alignment import check_pregrasp
from .ardy_visual_constraint import build_constraints
from .hand_controller import ContactDrivenHand
from .reference_runtime import publish_segment, submit_switch, switch_status
from .rolling_planner import RollingVisualPlanner
from .trace_bridge import build_ardy_condition, ardy_direction_prompt
from .visual_trace import CropTransform, task_crop, world_point_to_uv

GRASP_OFFSET = np.array([.125, .045, 0.])


def json_line(stream):
    while True:
        line = stream.readline()
        if not line: raise RuntimeError("worker exited before JSON response")
        try: return json.loads(line)
        except json.JSONDecodeError: continue


def contacts(model, data, block_geom):
    found = set()
    for contact in data.contact:
        if block_geom not in (contact.geom1, contact.geom2): continue
        other = contact.geom2 if contact.geom1 == block_geom else contact.geom1
        name = model.body(int(model.geom_bodyid[other])).name
        if name.startswith("right_hand_thumb"): found.add("thumb")
        if name.startswith(("right_hand_middle", "right_hand_index")): found.add("fingers")
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True); p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--scene", required=True)
    p.add_argument("--local-qwen-model", type=Path, required=True); p.add_argument("--qwen-device", default="musa:7")
    p.add_argument("--ardy-repo", type=Path, required=True); p.add_argument("--timeout", type=float, default=90.)
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False); args.reference_root.mkdir(parents=True, exist_ok=True)
    initial = publish_segment(args.source, args.reference_root, "000_initial", 0, 80,
        "A person stands balanced with both arms relaxed.", {"type":"initial_hold"})
    sonic_root = Path(os.environ["SONIC_ROOT"]); deploy = Path(os.environ["SONIC_DEPLOY"])
    scene_path = (sonic_root / args.scene).resolve(); os.chdir(sonic_root)
    config = SimLoopConfig().load_wbc_yaml(); config["ROBOT_SCENE"] = args.scene; config["ENABLE_ELASTIC_BAND"] = False
    runner = BaseSimulator(config, onscreen=args.viewer); env=runner.sim_env; env.elastic_band=None
    env.mj_data.qpos[:7]=env.mj_model.qpos0[:7]; env.mj_data.qpos[:2]=[.24,-.03]
    config["DEFAULT_DOF_ANGLES"] = np.array(config["DEFAULT_DOF_ANGLES"]).copy()
    for name,value in {"left_shoulder_roll_joint":.15,"right_shoulder_roll_joint":-.15,
                       "left_elbow_joint":1.4,"right_elbow_joint":1.4}.items():
        i=int(np.flatnonzero(env.body_joint_index==env.mj_model.joint(name).id)[0]); config["DEFAULT_DOF_ANGLES"][i]=value
    env.mj_data.qpos[env.body_joint_index+env.qpos_offset-1]=config["DEFAULT_DOF_ANGLES"]
    for i,motor in enumerate(env.unitree_bridge.low_cmd.motor_cmd[:env.num_body_dof]):
        motor.q=float(config["DEFAULT_DOF_ANGLES"][i]); motor.kp=float(config["MOTOR_KP"][i]); motor.kd=float(config["MOTOR_KD"][i])
    mujoco.mj_forward(env.mj_model,env.mj_data); model,data=env.mj_model,env.mj_data
    block=model.body("wood_block").id; block_geom=model.geom("wood_block_geom").id; wrist=model.body("right_wrist_yaw_link").id
    source_names=[joint.get("name") for joint in ET.parse(
        args.ardy_repo/"ardy/assets/skeletons/g1skel34/xml/g1.xml").findall(".//worldbody//joint")
        if joint.get("type")!="free"]
    source_qadr=np.array([model.joint(name).qposadr[0] for name in source_names])
    right_motor_indices=np.array([i for i,joint_id in enumerate(env.body_joint_index)
        if model.joint(int(joint_id)).name.startswith("right_") and any(part in model.joint(int(joint_id)).name
            for part in ("shoulder","elbow","wrist"))],dtype=int)

    logs=[]; qposes=[]; hand=ContactDrivenHand(rate_per_second=.8); initial_z=float(data.xpos[block][2])
    worker_logs=[]
    qlog=(args.output_dir/"qwen.log").open("w"); alog=(args.output_dir/"ardy.log").open("w")
    qwen=subprocess.Popen([os.sys.executable,"-m","qwen_ardy.local_qwen_server","--model",str(args.local_qwen_model),
        "--device",args.qwen_device,"--scene",str(scene_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=qlog,text=True,bufsize=1)
    qready=json_line(qwen.stdout)
    converter=sonic_root/"gear_sonic_deploy/reference/convert_ardy.py"
    ardy=subprocess.Popen([os.sys.executable,"-m","qwen_ardy.local_ardy_server","--ardy-repo",str(args.ardy_repo),
        "--checkpoints",str(args.ardy_repo/"checkpoints"),"--converter",str(converter)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=alog,text=True,bufsize=1)
    aready=json_line(ardy.stdout)
    # Warm ARDY's text encoder and first generation before SONIC/physics starts;
    # otherwise the short initial reference can finish during the cold request.
    warm_dir=args.output_dir/"ardy_warmup"; warm_dir.mkdir()
    warm_constraints=warm_dir/"constraints.json"
    build_constraints(args.ardy_repo,scene_path,data.qpos.copy(),
                      data.xpos[wrist]+data.xmat[wrist].reshape(3,3)@GRASP_OFFSET,
                      warm_constraints,3.)
    warm_request={"id":-1,"prompt":"A person stands balanced and keeps their right hand steady.",
      "duration_s":3.,"constraints":str(warm_constraints),"constraint_weight":2.,
      "output_stem":str(warm_dir/"motion"),"reference_dir":str(warm_dir/"reference"),
      "anchor_qpos":np.r_[data.qpos[:7],data.qpos[source_qadr]].tolist()}
    ardy.stdin.write(json.dumps(warm_request)+"\n"); ardy.stdin.flush()
    warm_response=json_line(ardy.stdout)
    if not warm_response.get("ok"): raise RuntimeError(warm_response)
    planner=RollingVisualPlanner(qwen,interval_s=2.2,stale_after_s=8.)

    master,slave=pty.openpty(); clog=(args.output_dir/"controller.log").open("w")
    command=[str(deploy/"target/release/g1_deploy_onnx_ref"),"lo",str(deploy/"policy/release/model_decoder.onnx"),
      str(args.reference_root),"--obs-config",str(deploy/"policy/release/observation_config.yaml"),
      "--encoder-file",str(deploy/"policy/release/model_encoder.onnx"),"--input-type","keyboard","--policy-precision","32","--disable-crc-check"]
    sonic=subprocess.Popen(command,stdin=slave,stdout=clog,stderr=subprocess.STDOUT,cwd=deploy); os.close(slave)
    started=False; playing=False; ready_wall=None; pending_switch=None; ready_reference=None
    ardy_busy=False; ardy_job=None; plan_id=0; last_switch_sim=0.
    targets_by_id={}; ready_target=None; pending_target=None
    servo_start=None; servo_target=None; servo_started_at=None
    close_requested=False; stable_contact=0.; next_tick=time.monotonic(); wall_start=time.monotonic()
    try:
      failure_reason=None
      while data.time < args.timeout:
        text=(args.output_dir/"controller.log").read_text(errors="replace")
        if ready_wall is None and "Init Done" in text: ready_wall=time.monotonic()
        if ready_wall is not None and not started: os.write(master,b"]"); started=True
        if started and not playing and time.monotonic()-ready_wall>1.5:
            os.write(master,b"t"); playing=True; last_switch_sim=data.time
        found=contacts(model,data,block_geom); rotation=data.xmat[wrist].reshape(3,3); centre=data.xpos[wrist]+rotation@GRASP_OFFSET
        delta=data.xpos[block]-centre; gate=check_pregrasp(delta,bool(found))
        if {"thumb","fingers"}<=found: stable_contact+=env.sim_dt
        else: stable_contact=0.
        holding=stable_contact>.2; lifted=float(data.xpos[block][2]-initial_z)>.08
        close_requested = close_requested or gate.aligned
        target=hand.update(env.sim_dt,close_requested, "thumb" in found,"fingers" in found)
        for commands in (env.unitree_bridge.left_hand_cmd,env.unitree_bridge.right_hand_cmd):
            for motor in commands.motor_cmd: motor.kp,motor.kd=6.,.1
        for i,motor in enumerate(env.unitree_bridge.right_hand_cmd.motor_cmd): motor.q=float(target[i])

        now=time.monotonic(); image=args.output_dir/f"crop_{plan_id:04d}.png"; full=args.output_dir/f"frame_{plan_id:04d}.png"
        if float(data.qpos[2]) < .45:
            failure_reason=f"robot_fell_root_z={float(data.qpos[2]):.3f}"; break
        try:
            hand_uv=world_point_to_uv(model,data,"blocks_overview",centre,(640,480)); block_uv=world_point_to_uv(model,data,"blocks_overview",data.xpos[block],(640,480))
        except ValueError as error:
            failure_reason=str(error); break
        crop=task_crop(hand_uv,block_uv,(640,480))
        if playing and planner.maybe_submit(now,image,data.qpos.tolist(),data.qvel.tolist(),
            {"contacts":sorted(found),"holding":holding,"lifted":lifted,
             "target_crop_uv":crop.full_uv_to_crop_uv(block_uv)},
             full_image=full,crop=crop.to_dict()): plan_id+=1
        plan=planner.poll(now)
        if plan and not ardy_busy:
            decision=plan.decision
            crop = CropTransform(**plan.request["crop"])
            target_crop_uv=np.asarray(plan.request["state"]["target_crop_uv"])
            raw_trace_uv=np.asarray(decision["trace"]["end_uv"],dtype=float)
            trace_error=float(np.linalg.norm(raw_trace_uv-target_crop_uv))
            decision["trace"]["raw_end_uv"]=raw_trace_uv.tolist()
            decision["trace"]["target_uv_error"]=trace_error
            if trace_error>.08:
                decision["trace"]["end_uv"]=target_crop_uv.tolist()
                decision["trace"]["grounding_status"]="vlm_rejected_mujoco_corrected"
            else:
                decision["trace"]["grounding_status"]="vlm_accepted"
            if holding:
                decision["next_subtask"]="lift"; decision["trace"]["end_uv"]=crop.full_uv_to_crop_uv(
                    world_point_to_uv(model,data,"blocks_overview",centre+np.array([0,0,.12]),(640,480)))
            projection_z=(float(centre[2]+.12) if holding else float(data.xpos[block][2]))
            max_step=(.13 if gate.distance_m<.13 else .05)
            condition=build_ardy_condition(model,data,decision,centre,projection_z,
                                           crop=crop,max_step_m=max_step)
            # Geometry is authoritative at the safety-critical transition.
            if gate.aligned: condition["next_subtask"]="close_gripper"; close_requested=True
            job=args.output_dir/f"plan_{plan.request_id:04d}"; job.mkdir()
            constraints=job/"constraints.json"
            constraint_meta=build_constraints(args.ardy_repo,scene_path,data.qpos.copy(),np.asarray(condition["constraint"]["endpoint_world_m"]),constraints,3.)
            solved_qpos=np.asarray(constraint_meta["solved_mujoco_qpos"])
            request={"id":plan.request_id,"prompt":condition["prompt"],"duration_s":3.,"constraints":str(constraints),
                     "constraint_weight":6.,"output_stem":str(job/"ardy_motion"),
                     "reference_dir":str(job/f"reference_{plan.request_id:04d}"),
                     "anchor_qpos":np.r_[data.qpos[:7],data.qpos[source_qadr]].tolist(),
                     "target_qpos":np.r_[solved_qpos[:7],solved_qpos[source_qadr]].tolist()}
            targets_by_id[plan.request_id]=np.array([
                solved_qpos[model.jnt_qposadr[env.body_joint_index[i]]]
                for i in range(len(env.body_joint_index))])
            ardy.stdin.write(json.dumps(request)+"\n"); ardy.stdin.flush(); ardy_busy=True; ardy_job=(request,condition,decision,crop)
            logs.append({"event":"plan","time":data.time,"decision":decision,"condition":condition,"gate":gate.__dict__})
        if ardy_busy and select.select([ardy.stdout],[],[],0)[0]:
            response=json_line(ardy.stdout)
            if not response.get("ok"): raise RuntimeError(response)
            ready_reference=Path(response["reference_dir"])
            ready_target=targets_by_id.pop(response["id"]); ardy_busy=False
            logs.append({"event":"ardy_ready","time":data.time,**response})
        if (ready_reference is not None and pending_switch is None and
                data.time-last_switch_sim >= 3.2):
            submit_switch(args.reference_root,ready_reference)
            pending_switch=ready_reference; pending_target=ready_target
            ready_reference=None; ready_target=None
        if pending_switch is not None and switch_status(args.reference_root,pending_switch):
            os.write(master,b"t"); last_switch_sim=data.time
            servo_start=np.array([data.qpos[model.jnt_qposadr[env.body_joint_index[i]]]
                                  for i in range(len(env.body_joint_index))])
            servo_target=pending_target; servo_started_at=data.time; pending_target=None
            logs.append({"event":"sonic_switch","time":data.time,"reference":str(pending_switch)}); pending_switch=None
        if servo_target is not None:
            u=min(1.,max(0.,(data.time-servo_started_at)/3.0)); u=u*u*(3.-2.*u)
            for i in right_motor_indices:
                motor=env.unitree_bridge.low_cmd.motor_cmd[int(i)]
                motor.q=float((1.-u)*servo_start[i]+u*servo_target[i])
                motor.kp=max(float(motor.kp),90.); motor.kd=max(float(motor.kd),2.5)
        if ready_wall is not None and time.monotonic()-ready_wall>.3: env.sim_step()
        else: env.unitree_bridge.PublishLowState(env.prepare_obs())
        mujoco.mj_kinematics(model,data)
        if int(data.time/env.sim_dt)%2==0: qposes.append(data.qpos.copy())
        if args.viewer: env.update_viewer()
        if lifted and holding: break
        next_tick+=env.sim_dt; time.sleep(max(0,next_tick-time.monotonic()))
    finally:
      for process in (sonic,qwen,ardy): process.terminate()
      for process in (sonic,qwen,ardy):
        try: process.wait(10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
      os.close(master); clog.close(); qlog.close(); alog.close(); runner.close()
    result={"success":bool(lifted and holding),"failure_reason":failure_reason,
            "block_lift_m":float(data.xpos[block][2]-initial_z),
            "contacts":sorted(found),"plans":sum(x["event"]=="plan" for x in logs),
            "qwen_ready":qready,"ardy_ready":aready,"fixed_stages":False,"timed_hand_trajectory":False}
    (args.output_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    (args.output_dir/"events.json").write_text(json.dumps(logs,indent=2)+"\n")
    np.savez_compressed(args.output_dir/"trajectory.npz",qpos=np.asarray(qposes)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
