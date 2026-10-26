#!/usr/bin/env python3
"""Iterative Cartesian reference correction from a measured free-base SONIC trial.

Only reference joint angles change. Simulator states, body actuation, collision
and forces are untouched; each candidate requires a new physical SONIC rollout.
"""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from scipy.ndimage import gaussian_filter1d
from scene import load_scene
from grasp_reference import GraspReference
from expert_trajectory import JointMap,load_reference,write_json,sha256
from reference_io import export_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--run',required=True)
    p.add_argument('--name',required=True);p.add_argument('--gain',type=float,default=.8)
    args=p.parse_args();ep=args.episode;out=ep/'candidates'/args.name
    if out.exists():p.error('Candidate exists')
    trial=ep/'validation'/args.run;rm=json.loads((trial/'metadata.json').read_text())
    base=ep/'candidates'/rm['candidate_type'];arrays,meta=load_reference(base)
    frames=np.load(trial/'reference_frame.npy')
    if frames.tolist()!=list(range(len(arrays['timestamps']))):p.error('Complete rollout required')
    m,d=load_scene();b=JointMap(m,meta['body_joint_order'],29)
    source_q=np.load(ep/'source/qpos.npy');actual=np.load(trial/'qpos.npy')
    target_positions=np.load(ep/'source/eef_pos.npy');target_rot=[]
    wrist=m.body('right_wrist_yaw_link').id
    for q in source_q:
        d.qpos[:]=q;mujoco.mj_forward(m,d);target_rot.append(d.xmat[wrist].reshape(3,3).copy())
    g=GraspReference(m,d)
    columns=[b.names.index(m.joint(j).name) for j in g.arm]
    correction=np.zeros_like(arrays['body_ref_q'])
    for f,q in enumerate(actual):
        g.plan.qpos[:]=q;g.rotation=target_rot[f]
        g.ik(target_positions[f])
        correction[f,columns]=g.plan.qpos[g.aq]-q[g.aq]
    correction=gaussian_filter1d(correction,3,axis=0,mode='nearest')
    blend=np.clip((arrays['timestamps']-.5)/1,0,1);blend=blend**2*(3-2*blend)
    increment=np.clip(args.gain*correction,-.5,.5)*blend[:,None]
    arrays['body_ref_q']=np.clip(arrays['body_ref_q']+increment,m.jnt_range[b.jids,0],m.jnt_range[b.jids,1])
    arrays['body_ref_dq']=np.gradient(arrays['body_ref_q'],arrays['timestamps'],axis=0)
    meta.update(candidate_type=args.name,expert_valid=False,failure_reason='requires_new_physical_sonic_replay',
                sonic_config=dict(meta['sonic_config'],encoder_mode=0),
                refinement={'method':'Cartesian iterative learning through arm IK at measured freebase pose',
                            'from_run':args.run,'gain':args.gain,'max_step_rad':.5,
                            'measured_qpos_sha256':sha256(trial/'qpos.npy')},
                root_velocities=(np.loadtxt(base/'body_lin_vel.csv',delimiter=',',skiprows=1,ndmin=2)[:,:3],
                                 np.loadtxt(base/'body_ang_vel.csv',delimiter=',',skiprows=1,ndmin=2)[:,:3]))
    meta.pop('cartesian_body_part_indexes',None);meta.pop('cartesian_body_names',None)
    export_reference(out,arrays,b.names,meta['hand_joint_order'],meta)
    print(out, 'maximum correction',float(np.max(abs(increment))),flush=True)
if __name__=='__main__':main()
