#!/usr/bin/env python3
"""Human-in-the-loop text → persistent ARDY → hot SONIC interactive simulator."""
from __future__ import annotations

import argparse, json, os, pty, queue, select, subprocess, sys, threading, time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

from .ardy_visual_constraint import build_continuity_constraints
from .hand_controller import ContactDrivenHand
from .reference_runtime import publish_segment, submit_switch, switch_status


def json_line(stream):
    while True:
        line=stream.readline()
        if not line: raise RuntimeError("ARDY worker exited")
        try: return json.loads(line)
        except json.JSONDecodeError: continue


def input_thread(messages):
    for line in sys.stdin:
        messages.put(line.strip())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True); p.add_argument("--reference-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--scene",required=True)
    p.add_argument("--ardy-repo",type=Path,required=True); p.add_argument("--viewer",action="store_true")
    p.add_argument("--segment-seconds",type=float,default=8.)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=False); a.reference_root.mkdir(parents=True,exist_ok=True)
    publish_segment(a.source,a.reference_root,"000_initial",0,80,
                    "A person stands balanced with both arms relaxed.",{"type":"initial"})
    sonic_root=Path(os.environ["SONIC_ROOT"]); deploy=Path(os.environ["SONIC_DEPLOY"]); scene_path=(sonic_root/a.scene).resolve()
    os.chdir(sonic_root); config=SimLoopConfig().load_wbc_yaml(); config["ROBOT_SCENE"]=a.scene; config["ENABLE_ELASTIC_BAND"]=False
    runner=BaseSimulator(config,onscreen=a.viewer); env=runner.sim_env; env.elastic_band=None
    env.mj_data.qpos[:7]=env.mj_model.qpos0[:7]; env.mj_data.qpos[:2]=[.24,-.03]
    config["DEFAULT_DOF_ANGLES"]=np.array(config["DEFAULT_DOF_ANGLES"]).copy()
    for name,value in {"left_shoulder_roll_joint":.15,"right_shoulder_roll_joint":-.15,
                       "left_elbow_joint":1.4,"right_elbow_joint":1.4}.items():
        i=int(np.flatnonzero(env.body_joint_index==env.mj_model.joint(name).id)[0]); config["DEFAULT_DOF_ANGLES"][i]=value
    env.mj_data.qpos[env.body_joint_index+env.qpos_offset-1]=config["DEFAULT_DOF_ANGLES"]
    for i,motor in enumerate(env.unitree_bridge.low_cmd.motor_cmd[:env.num_body_dof]):
        motor.q=float(config["DEFAULT_DOF_ANGLES"][i]); motor.kp=float(config["MOTOR_KP"][i]); motor.kd=float(config["MOTOR_KD"][i])
    mujoco.mj_forward(env.mj_model,env.mj_data); model,data=env.mj_model,env.mj_data
    source_names=[j.get("name") for j in ET.parse(a.ardy_repo/"ardy/assets/skeletons/g1skel34/xml/g1.xml").findall(".//worldbody//joint") if j.get("type")!="free"]
    source_qadr=np.array([model.joint(name).qposadr[0] for name in source_names])
    converter=sonic_root/"gear_sonic_deploy/reference/convert_ardy.py"; alog=(a.output_dir/"ardy.log").open("w")
    ardy=subprocess.Popen([sys.executable,"-m","qwen_ardy.local_ardy_server","--ardy-repo",str(a.ardy_repo),
        "--checkpoints",str(a.ardy_repo/"checkpoints"),"--converter",str(converter)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=alog,text=True,bufsize=1)
    print("正在加载 ARDY，请等待……",flush=True); print("ARDY:",json_line(ardy.stdout),flush=True)
    warm=a.output_dir/"warmup"; warm.mkdir(); constraints=warm/"constraints.json"
    build_continuity_constraints(a.ardy_repo,scene_path,data.qpos.copy(),constraints,a.segment_seconds)
    anchor=np.r_[data.qpos[:7],data.qpos[source_qadr]].tolist()
    ardy.stdin.write(json.dumps({"id":-1,"prompt":"A person stands balanced and keeps both arms steady.",
      "duration_s":a.segment_seconds,"constraints":str(constraints),"output_stem":str(warm/"motion"),
      "reference_dir":str(warm/"reference"),"anchor_qpos":anchor})+"\n"); ardy.stdin.flush()
    if not json_line(ardy.stdout).get("ok"): raise RuntimeError("ARDY warm-up failed")
    master,slave=pty.openpty(); clog=(a.output_dir/"controller.log").open("w")
    sonic=subprocess.Popen([str(deploy/"target/release/g1_deploy_onnx_ref"),"lo",str(deploy/"policy/release/model_decoder.onnx"),
      str(a.reference_root),"--obs-config",str(deploy/"policy/release/observation_config.yaml"),"--encoder-file",str(deploy/"policy/release/model_encoder.onnx"),
      "--input-type","keyboard","--policy-precision","32","--disable-crc-check"],stdin=slave,stdout=clog,stderr=subprocess.STDOUT,cwd=deploy); os.close(slave)
    messages=queue.Queue(); threading.Thread(target=input_thread,args=(messages,),daemon=True).start()
    hand=ContactDrivenHand(rate_per_second=.8); hand_close=False; job_id=0; busy=False; pending=None; started=False; playing=False; ready_wall=None
    active_prompt=None; queued_prompt=None
    events=[]; qposes=[]; next_tick=time.monotonic(); print("输入 /help 查看命令。",flush=True)

    def start_generation(prompt, reason):
        nonlocal job_id,busy
        job=a.output_dir/f"command_{job_id:04d}"; job.mkdir(); constraints=job/"constraints.json"
        build_continuity_constraints(a.ardy_repo,scene_path,data.qpos.copy(),constraints,a.segment_seconds)
        anchor=np.r_[data.qpos[:7],data.qpos[source_qadr]].tolist(); reference=job/f"reference_{job_id:04d}"
        request={"id":job_id,"prompt":prompt,"duration_s":a.segment_seconds,"constraints":str(constraints),
                 "output_stem":str(job/"motion"),"reference_dir":str(reference),"anchor_qpos":anchor}
        ardy.stdin.write(json.dumps(request)+"\n"); ardy.stdin.flush(); busy=True
        events.append({"event":"human_text" if reason=="user" else "auto_continue",
                       "time":data.time,"text":prompt})
        print(f"ARDY 正在生成 {a.segment_seconds:.1f}s 动作: {prompt}"+
              ("" if reason=="user" else "  [自动续写]"),flush=True)
        job_id+=1
    try:
      while True:
        text=(a.output_dir/"controller.log").read_text(errors="replace")
        if ready_wall is None and "Init Done" in text: ready_wall=time.monotonic()
        if ready_wall is not None and not started: os.write(master,b"]"); started=True
        if started and not playing and time.monotonic()-ready_wall>1.5: os.write(master,b"t"); playing=True; print("SONIC 已接管，可以输入动作。",flush=True)
        while not messages.empty():
            command=messages.get()
            if not command: continue
            if command=="/quit": return
            if command=="/help":
                print("输入英文身体动作描述；动作播完后停在末帧；新 text 可在中途替换；/open；/close；/status；/quit",flush=True); continue
            if command=="/open": hand_close=False; print("右手张开",flush=True); continue
            if command=="/close": hand_close=True; print("右手开始接触驱动闭合",flush=True); continue
            if command=="/status":
                print(f"sim_time={data.time:.2f}s, ARDY_busy={busy}, active_prompt={active_prompt!r}, queued_prompt={queued_prompt!r}",flush=True); continue
            active_prompt=command
            if busy or pending is not None:
                queued_prompt=command
                print("新 text 已排队，当前生成/切换完成后立即生效。",flush=True)
            else:
                start_generation(active_prompt,"user")
        if busy and select.select([ardy.stdout],[],[],0)[0]:
            response=json_line(ardy.stdout)
            if not response.get("ok"): print("ARDY 失败:",response,flush=True); busy=False
            else: pending=Path(response["reference_dir"]); submit_switch(a.reference_root,pending); busy=False; print(f"reference 已生成 ({response['elapsed_s']:.2f}s)，等待 SONIC 切换",flush=True)
        if pending is not None and switch_status(a.reference_root,pending):
            os.write(master,b"t"); events.append({"event":"switch","time":data.time,"reference":str(pending)}); print("SONIC 已开始执行新动作；播完后会停在末帧，不会自动生成下一段。",flush=True); pending=None
            if queued_prompt is not None:
                prompt=queued_prompt; queued_prompt=None; start_generation(prompt,"user")
        target=hand.update(env.sim_dt,hand_close,False,False)
        for commands in (env.unitree_bridge.left_hand_cmd,env.unitree_bridge.right_hand_cmd):
            for motor in commands.motor_cmd: motor.kp,motor.kd=6.,.1
        for i,motor in enumerate(env.unitree_bridge.right_hand_cmd.motor_cmd): motor.q=float(target[i])
        if ready_wall is not None and time.monotonic()-ready_wall>.3: env.sim_step()
        else: env.unitree_bridge.PublishLowState(env.prepare_obs())
        if int(data.time/env.sim_dt)%2==0: qposes.append(data.qpos.copy())
        if a.viewer: env.update_viewer()
        next_tick+=env.sim_dt; time.sleep(max(0,next_tick-time.monotonic()))
    finally:
      for process in (sonic,ardy): process.terminate()
      os.close(master); clog.close(); alog.close(); runner.close()
      np.savez_compressed(a.output_dir/"trajectory.npz",qpos=np.asarray(qposes))
      (a.output_dir/"events.json").write_text(json.dumps(events,indent=2,ensure_ascii=False)+"\n")
      print("已保存运行记录：",a.output_dir,flush=True)


if __name__=="__main__": main()
