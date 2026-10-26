#!/usr/bin/env python3
"""Audit model-space neutral geometry and measured rollout contacts, without stepping."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scene import load_scene
from expert_trajectory import write_json


def audit(episode, rollout):
    meta=json.loads((episode/'metadata.json').read_text())
    model,data=load_scene(root_height=meta['source_root_height_m'])
    initial=np.load(episode/'source/qpos.npy')[0]
    data.qpos[:]=initial;mujoco.mj_forward(model,data)
    body_names=[model.body(i).name or '' for i in range(model.nbody)]
    table=[i for i in range(model.ngeom) if body_names[model.geom_bodyid[i]]=='task_table']
    cube=[model.geom('task_cube_geom').id]
    sides={}
    for side in ('left','right'):
        geom_ids=[i for i in range(model.ngeom) if model.geom_contype[i] and
            body_names[model.geom_bodyid[i]].startswith(side+'_') and
            any(part in body_names[model.geom_bodyid[i]] for part in ('shoulder','elbow','wrist','hand'))]
        distances={name:min(float(mujoco.mj_geomDistance(model,data,a,b,2.,None))
                            for a in geom_ids for b in others) for name,others in [('table',table),('cube',cube)]}
        wrist=model.body(side+'_wrist_yaw_link').id
        sides[side]=dict(wrist_position=data.xpos[wrist].tolist(),minimum_arm_table_clearance_m=distances['table'],
                         minimum_arm_cube_clearance_m=distances['cube'])
    initial_contacts=[]
    for contact in data.contact:
        a,b=[body_names[model.geom_bodyid[g]] for g in (contact.geom1,contact.geom2)]
        if a=='world' or b=='world' or {a,b}=={'task_table','task_red_cube'}:continue
        initial_contacts.append([a,b,float(contact.dist)])
    folder=episode/rollout
    q=np.load(folder/'qpos.npy');ts=np.load(folder/'timestamps.npy')
    counts={'left_arm_environment':0,'robot_self_contact':0,'early_robot_table':0}
    pairs=set();min_height=float('inf');max_tilt=0.;left_drift=0.
    left_names=[f'left_{n}_joint' for n in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']]
    left_q=np.array([model.joint(n).qposadr[0] for n in left_names])
    pelvis=model.body('pelvis').id
    for state,t in zip(q,ts):
        data.qpos[:]=state;mujoco.mj_forward(model,data)
        min_height=min(min_height,float(data.xpos[pelvis,2]))
        max_tilt=max(max_tilt,float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3,3)[2,2],-1,1)))))
        left_drift=max(left_drift,float(np.max(abs(state[left_q]-initial[left_q]))))
        seen=set()
        for contact in data.contact:
            a,b=[body_names[model.geom_bodyid[g]] for g in (contact.geom1,contact.geom2)]
            if a=='world' or b=='world' or a.startswith('task_') and b.startswith('task_'):continue
            if a.startswith('task_'):a,b=b,a
            if b.startswith('task_'):
                if a.startswith('left_') and any(n in a for n in ['shoulder','elbow','wrist','hand']):seen.add('left_arm_environment')
                if b=='task_table' and t<6:seen.add('early_robot_table')
            else:seen.add('robot_self_contact')
            pairs.add((a,b))
        for key in seen:counts[key]+=1
    result=dict(neutral_pose=meta.get('initial_pose',{}).get('definition',meta.get('initial_pose',{})),initial_geometry=sides,initial_unexpected_contacts=initial_contacts,
                sampled_rollout=str(folder),sample_rate_hz=50,sampled_frames=len(q),sampled_contact_frames=counts,
                contact_body_pairs=sorted(pairs),minimum_pelvis_height_m=min_height,
                maximum_pelvis_tilt_degrees=max_tilt,left_arm_max_joint_deviation_rad=left_drift,
                initial_pose_geometry_valid=not initial_contacts and all(v['minimum_arm_table_clearance_m']>.05 and v['minimum_arm_cube_clearance_m']>.05 for v in sides.values()))
    write_json(folder/'neutral_audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--rollout',default='source');a=p.parse_args()
    print(json.dumps(audit(a.episode,a.rollout),indent=2))
