#!/usr/bin/env python3
"""Compile canonical offline IK conditions and synchronized hand plan for Kimodo.

Does not generate model motion or execute physical actuation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path(os.environ.get('RESEARCH_ROOT', Path(__file__).resolve().parents[3])).resolve()
PROJECT=Path(os.environ.get('KIMODO_RUNTIME_SUPPORT', Path(__file__).resolve().parents[1]/'runtime_support')).resolve()
SONIC=Path(os.environ.get('SONIC_REPO', ROOT/'GR00T-WholeBodyControl')).resolve()
ROBOT=SONIC/'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml'
os.environ['SONIC_REPO']=str(SONIC)
sys.path[:0]=[str(Path(__file__).resolve().parent),str(PROJECT/'scripts'),str(SONIC)]
import project_paths
project_paths.default_robot_xml=lambda:ROBOT
from scene import load_scene
from cases import reset_case,read_cases
from episode_schema import neutral_arms,NEUTRAL_POSE
from expert_trajectory import orders,JointMap,CRITERION
from grasp_reference import GraspReference
from kimodo.skeleton import G1Skeleton34
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo_pose_constraint_codec import make_pose_constraint_record,save_pose_constraint_record,load_pose_constraints

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--case',default='center');ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--duration',type=float,default=16.2);a=ap.parse_args()
    if a.duration!=16.2:ap.error('Pilot duration is 16.2 s; change the explicit integration contract before other durations')
    a.out.mkdir(parents=True,exist_ok=False)
    m,d=load_scene(supported=False,robot_path=ROBOT)
    case=next(c for c in read_cases()['cases'] if c['id']==a.case);reset_case(m,d,case)
    for side in ('left','right'):
        for part in ('hip_pitch','knee','ankle_pitch'):
            d.qpos[m.joint(f'{side}_{part}_joint').qposadr[0]]=NEUTRAL_POSE[part]
    neutral_arms(m,d);mujoco.mj_forward(m,d)
    feet=[i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name in ('left_ankle_roll_link','right_ankle_roll_link') and m.geom_contype[i]]
    root=int(m.joint('floating_base_joint').qposadr[0])
    d.qpos[root+2]-=min(d.geom_xpos[i,2]-m.geom_size[i,0] for i in feet)+.0001;mujoco.mj_forward(m,d)
    assert (m.nq,m.nv,m.nu,m.neq)==(57,55,43,0)
    assert np.allclose(m.opt.gravity,[0,0,-9.81])
    bn,hn,_=orders();hand=JointMap(m,hn,14)
    converter=MujocoQposConverter(G1Skeleton34())
    names=[j.get('name') for j in ET.parse(converter.xml_path).findall('.//worldbody//joint') if j.get('type')!='free']
    assert len(names)==29 and set(names)==set(bn)
    qa=np.array([m.joint(n).qposadr[0] for n in names])
    ns,n=487,810;st=np.arange(ns)/30;t=np.arange(n)/50
    plan=GraspReference(m,d,lift_height=.18);plan.natural_start=True
    sq=np.empty((ns,36));hq=np.empty((n,14));wp=np.empty((ns,2,3));wq=np.empty((ns,2,4));errors=[]
    source_ticks={5*k:k for k in range(ns)};reference_ticks={3*k:k for k in range(n)}
    wrists=[m.body(s+'_wrist_yaw_link').id for s in ('left','right')]
    limit_ok=True
    for tick in sorted(source_ticks.keys()|reference_ticks.keys()):
        time=tick/150
        if time>=1:plan.control(time-1)
        mujoco.mj_forward(m,plan.plan)
        for name in bn+hn:
            j=m.joint(name);value=plan.plan.qpos[j.qposadr[0]]
            if j.limited[0] and not j.range[0]-1e-8<=value<=j.range[1]+1e-8:limit_ok=False
        if tick in source_ticks:
            k=source_ticks[tick];sq[k]=np.r_[plan.plan.qpos[root:root+7],plan.plan.qpos[qa]]
            wp[k]=plan.plan.xpos[wrists];wq[k]=plan.plan.xquat[wrists]
        if tick in reference_ticks:hq[reference_ticks[tick]]=plan.plan.qpos[hand.qa]
        if time>=8:
            u=np.clip((time-10)/2,0,1);goal=plan.target+np.array([0,0,.18*u*u*(3-2*u)])
            errors.append(float(np.linalg.norm(plan.plan.xpos[plan.wrist]-goal)))
    zero=np.r_[d.qpos[root:root+7],d.qpos[qa]]
    frame_zero=bool(np.max(abs(sq[0]-zero))<1e-10 and np.max(abs(hq[0]))<1e-10)
    if not limit_ok or not frame_zero or max(errors)>.02:raise ValueError('Offline phase/joint-limit/initial-state validation failed')
    # Convert the dense uniform motion before selecting key poses; SDK velocities are never computed from sparse timings.
    record=make_pose_constraint_record(converter,sq,np.arange(ns),30,fullbody_rows=(0,30))
    selected=sorted(set(range(0,ns,6))|{30,ns-1})
    for key in ('canonical_qpos','local_rot_mats','root_positions','frame_indices'):record[key]=[record[key][k] for k in selected]
    record['fullbody_rows']=[selected.index(k) for k in (0,30)]
    save_pose_constraint_record(a.out/'pose_constraints.json',record)
    _,checks=load_pose_constraints(a.out/'pose_constraints.json',converter)
    np.savez_compressed(a.out/'task_plan.npz',source_timestamps=st,source_qpos=sq,wrist_world_pos=wp,wrist_world_quat=wq,
                        timestamps=t,hand_ref_q=hq,hand_ref_dq=np.gradient(hq,t,axis=0))
    scene=PROJECT/'scenes/tabletop.xml'
    spec=dict(contract_version='kimodo_grasp_integration_v0.1',task_id=f'kimodo-{a.case}-seed{a.seed}-16p2-v1',run_id=a.out.name,
              case=case,seed=a.seed,duration_s=a.duration,source_fps=30,reference_hz=50,source_min_frames=ns,reference_frames=n,
              prompt='A standing person reaches the right hand over the table, grasps the red cube with the right thumb and fingers, lifts it, and holds it steadily above the table.',
              phases={'initial':[0,1],'raise_outside_table':[1,4],'cross_table_edge':[4,6],'lower':[6,8],'close':[8,10],'lift':[10,12],'hold':[12,16.2]},
              initial_qpos=d.qpos.tolist(),initial_qvel=d.qvel.tolist(),body_joint_order=bn,hand_joint_order=hn,source_qpos_joint_order=names,
              robot=str(ROBOT),robot_sha256=digest(ROBOT),scene=str(scene),scene_sha256=digest(scene),
              cube_initial_position=d.xpos[plan.cube].tolist(),cube_side_m=(m.geom_size[m.geom('task_cube_geom').id]*2).tolist(),
              cube_mass_kg=float(m.body_mass[plan.cube]),cube_friction=m.geom_friction[m.geom('task_cube_geom').id].tolist(),
              grasp_wrist_target_world=plan.target.tolist(),lift_height_plan_m=.18,criterion=CRITERION,
              constraints=str((a.out/'pose_constraints.json').resolve()))
    write(a.out/'task.json',spec)
    metrics=dict(grasp_and_lift_position_error_max_m=max(errors),joint_limits_pass=limit_ok,frame_zero_pass=frame_zero,
                 model_generation_run=False,physics_replay_run=False,source_frames=ns,reference_frames=n,**checks)
    write(a.out/'planning_metrics.json',metrics);print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
