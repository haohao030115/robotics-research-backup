#!/usr/bin/env python3
"""Run the existing SONIC controller and original block scene; log physical grasp."""
import argparse, json, os, pty, subprocess, time
from pathlib import Path
import numpy as np
import mujoco
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from .hand_controller import ContactDrivenHand

ROOT=Path(os.environ.get('SONIC_ROOT',str(Path(os.environ.get('RESEARCH_ROOT','/workspace/group2'))/'GR00T-WholeBodyControl')))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-dir',type=Path,default=ROOT/'gear_sonic_deploy/reference/ardy_grasp')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/ardy_grasp/playback')
    parser.add_argument('--robot-x',type=float,default=0.,help='Robot starting x; objects and scene unchanged')
    parser.add_argument('--robot-y',type=float,default=0.)
    parser.add_argument('--scene',default='gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks.xml')
    parser.add_argument('--hand-kp',type=float,default=1.5)
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--arms-at-sides',action='store_true')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT)
    config=SimLoopConfig().load_wbc_yaml()
    config['ROBOT_SCENE']=args.scene
    config['ENABLE_ELASTIC_BAND']=False
    runner=BaseSimulator(config,onscreen=args.viewer)
    env=runner.sim_env
    env.elastic_band=None  # No elastic band, root forces, or fixed base.
    env.mj_data.qpos[0]=args.robot_x
    env.mj_data.qpos[1]=args.robot_y
    # Initial stance. Physics remains paused while the controller loads its models.
    if args.arms_at_sides:
        config['DEFAULT_DOF_ANGLES']=np.array(config['DEFAULT_DOF_ANGLES']).copy()
        for name,value in {'left_shoulder_roll_joint':.15,'right_shoulder_roll_joint':-.15,'left_elbow_joint':1.4,'right_elbow_joint':1.4}.items():
            i=int(np.flatnonzero(env.body_joint_index==env.mj_model.joint(name).id)[0])
            config['DEFAULT_DOF_ANGLES'][i]=value
    env.mj_data.qpos[env.body_joint_index+env.qpos_offset-1]=config['DEFAULT_DOF_ANGLES']
    for i in range(env.num_body_dof):
        cmd=env.unitree_bridge.low_cmd.motor_cmd[i]
        cmd.q=config['DEFAULT_DOF_ANGLES'][i]
        cmd.kp=config['MOTOR_KP'][i]; cmd.kd=config['MOTOR_KD'][i]
    for commands in [env.unitree_bridge.left_hand_cmd,env.unitree_bridge.right_hand_cmd]:
        for cmd in commands.motor_cmd:
            cmd.kp=1.5; cmd.kd=.1
    mujoco.mj_forward(env.mj_model,env.mj_data)
    m,d=env.mj_model,env.mj_data
    if env.viewer:
        env.viewer.cam.lookat[:]=[.35,0,.85]
        env.viewer.cam.distance=2.; env.viewer.cam.azimuth=135; env.viewer.cam.elevation=-20
    wrist=m.body('right_wrist_yaw_link').id
    block=m.body('wood_block').id
    geom=m.geom('wood_block_geom').id
    master,slave=pty.openpty()
    command=['./target/release/g1_deploy_onnx_ref','lo','policy/release/model_decoder.onnx',str(args.motion_dir.resolve()),'--obs-config','policy/release/observation_config.yaml','--encoder-file','policy/release/model_encoder.onnx','--input-type','keyboard','--policy-precision','32','--disable-crc-check','--enable-csv-logs','--logs-dir',str((args.output_dir/'controller').resolve())]
    log=(args.output_dir/'controller.log').open('w')
    child=subprocess.Popen(command,stdin=slave,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT/'gear_sonic_deploy')
    os.close(slave)
    states=[]; qpos=[]; qvel=[]; success_hold=0.; task_success=False
    sent_start=False; sent_play=False; objects_reset=False
    hand=ContactDrivenHand(rate_per_second=.8)
    ready_at=None
    start=time.monotonic(); next_tick=start
    print('Controller pid:',child.pid,flush=True)
    try:
        for step in range(int(150/env.sim_dt)):
            t=step*env.sim_dt
            if ready_at is None and step%20==0 and 'Init Done' in (args.output_dir/'controller.log').read_text():
                ready_at=t; print('Controller ready',t,flush=True)
            phase=t-ready_at if ready_at is not None else -100
            if phase>=.1 and not sent_start:
                os.write(master,b']'); sent_start=True
                print('Start control',t,flush=True)
            if phase>=1.5 and not objects_reset:
                # Reset the episode objects BEFORE playback, then let them settle.
                for name in ['wood_block_free','red_block_free']:
                    j=m.joint(name).id; q=m.jnt_qposadr[j]; v=m.jnt_dofadr[j]
                    d.qpos[q:q+7]=m.qpos0[q:q+7]; d.qvel[v:v+6]=0
                objects_reset=True
            if phase>=2 and not sent_play:
                os.write(master,b't'); sent_play=True
                print('Play ARDY reference',t,flush=True)
            found=set()
            for c in d.contact:
                if geom in [c.geom1,c.geom2]:
                    other=c.geom2 if c.geom1==geom else c.geom1
                    name=m.body(int(m.geom_bodyid[other])).name
                    if name.startswith('right_hand_thumb'): found.add('thumb')
                    if name.startswith(('right_hand_middle','right_hand_index')): found.add('fingers')
            playback_elapsed=max(0.,phase-2.)
            # Stage boundary from the validated ARDY demo: the palm is aligned
            # at about 5 s; close while the reference holds the wrist, before
            # the lift portion starts at 8 s.
            hand_target=hand.update(env.sim_dt,playback_elapsed>=5.,
                                    'thumb' in found,'fingers' in found)
            for commands in [env.unitree_bridge.left_hand_cmd,env.unitree_bridge.right_hand_cmd]:
                for cmd in commands.motor_cmd: cmd.kp=args.hand_kp; cmd.kd=.1
            for i,cmd in enumerate(env.unitree_bridge.right_hand_cmd.motor_cmd):
                cmd.q=float(hand_target[i])
            if phase>=.3:
                env.sim_step()
            else:
                # Loading stage: publish initial observations, physics starts with policy.
                env.unitree_bridge.PublishLowState(env.prepare_obs())
            if step%4==0:
                env.update_viewer()
                mujoco.mj_kinematics(m,d)
                center=d.xpos[wrist]+d.xmat[wrist].reshape(3,3)@np.array([.125,.045,0])
                states.append([t,*d.xpos[block],*center,*d.qpos[:3],len(found)])
                qpos.append(d.qpos.copy()); qvel.append(d.qvel.copy())
            block_lift=float(d.xpos[block][2]-m.qpos0[m.jnt_qposadr[m.joint('wood_block_free').id]+2])
            if block_lift>.08 and {'thumb','fingers'}<=found:
                success_hold+=env.sim_dt
            else:
                success_hold=0.
            if success_hold>=.20:
                task_success=True
                break
            if phase>=14.5: break
            next_tick+=env.sim_dt
            time.sleep(max(0,next_tick-time.monotonic()))
    finally:
        child.terminate()
        child.wait(timeout=10)
        os.close(master); log.close()
        runner.close()
    np.savez(args.output_dir/'simulation.npz',states=states,qpos=qpos,qvel=qvel,fps=50.)
    a=np.array(states)
    episode=a[a[:,0]>=ready_at+2] if ready_at is not None else a
    initial_z=float(episode[0,3])
    result={'arms_at_sides':args.arms_at_sides,'root_position_assistance':False,'elastic_band_enabled':False,'hand_kp':args.hand_kp,'robot_initial_x':args.robot_x,'robot_initial_y':args.robot_y,'controller_ready_at':ready_at,'playback_start_at':ready_at+2 if ready_at is not None else None,'scene':config['ROBOT_SCENE'],'block_lift_max_m':float(episode[:,3].max()-initial_z),'block_lift_final_m':float(a[-1,3]-initial_z),'minimum_hand_block_distance_m':float(np.min(np.linalg.norm(episode[:,1:4]-episode[:,4:7],axis=1))),'final_contacts':int(a[-1,-1]),'minimum_root_height_m':float(a[:,9].min()),'final_block_position':a[-1,1:4].tolist(),'final_root_position':a[-1,7:10].tolist()}
    result['success']=bool(task_success and result['block_lift_final_m']>.08 and result['final_contacts']==2)
    result['stable_lift_contact_s']=float(success_hold)
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
