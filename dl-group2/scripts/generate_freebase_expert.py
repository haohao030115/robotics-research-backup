#!/usr/bin/env python3
"""Contact-constrained whole-body inverse dynamics; actuators are the only inputs.

Feet, COM and pelvis orientation tasks stabilize the free base. Grasp IK is a
wrist geometry reference only. No equality, applied force, or state edits occur
after initialization. Failed attempts remain explicitly uncertified.
"""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from scipy.linalg import solve
from scene import load_scene
from cases import reset_case,read_cases,case_index
from grasp_reference import GraspReference
from expert_trajectory import JointMap,orders,robot_state,PhysicalGraspEvaluator,write_json,save_arrays,sha256
from reference_io import export_reference
from reference_io import write_csv
from episode_schema import dataset_fields, neutral_arms, NEUTRAL_POSE


class WholeBody:
    def __init__(self,m,d,body,hand,neutral=False):
        self.m,self.d,self.body,self.hand=m,d,body,hand
        root=m.joint('floating_base_joint'); self.rva=int(root.dofadr[0])
        self.v=np.r_[np.arange(self.rva,self.rva+6),body.va]
        self.n=len(self.v)
        self.feet=[m.body(s+'_ankle_roll_link').id for s in ('left','right')]
        self.footpos=d.xpos[self.feet].copy();self.footrot=d.xmat[self.feet].reshape(2,3,3).copy()
        self.pelvis=m.body('pelvis').id
        self.height=d.xpos[self.pelvis,2]
        self.com_target=d.subtree_com[self.pelvis,:2].copy() if neutral else np.array([.025,0.])
        self.prev_j=None
        self.mass=np.zeros((m.nv,m.nv))
        self.g=GraspReference(m,d)
        self.g.lift_height=.18
        self.last_wrench=np.zeros(12)

    def control(self,t):
        rootqa=int(self.m.joint('floating_base_joint').qposadr[0])
        self.g.plan.qpos[rootqa:rootqa+7]=self.d.qpos[rootqa:rootqa+7]
        if t >= 1:self.g.control(t-1)

    def apply(self):
        m,d,b=self.m,self.d,self.body
        qvel=d.qvel[self.v]
        js=[];errs=[]
        for k,foot in enumerate(self.feet):
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
            mujoco.mj_jacBody(m,d,jp,jr,foot)
            r=d.xmat[foot].reshape(3,3);rd=self.footrot[k]
            er=.5*sum(np.cross(r[:,i],rd[:,i]) for i in range(3))
            js.append(np.vstack([jp[:,self.v],jr[:,self.v]]))
            errs.append(np.r_[self.footpos[k]-d.xpos[foot],er])
        jc=np.vstack(js)
        jcom=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,jcom,self.pelvis)
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,jp,jr,self.pelvis)
        j=np.vstack([jc,jcom[:2,self.v],jp[2:3,self.v],jr[:,self.v]])
        jdq=np.zeros(len(j)) if self.prev_j is None else (j-self.prev_j)@qvel/m.opt.timestep
        self.prev_j=j.copy()
        r=d.xmat[self.pelvis].reshape(3,3)
        er=.5*sum(np.cross(r[:,i],np.eye(3)[:,i]) for i in range(3))
        error=np.r_[np.concatenate(errs),self.com_target-d.subtree_com[self.pelvis,:2],self.height-d.xpos[self.pelvis,2],er]
        desired=100*error-20*j@qvel-jdq
        desired[:12]=400*error[:12]-40*jc@qvel-jdq[:12]
        posture=100*(self.g.plan.qpos[b.qa]-d.qpos[b.qa])-20*d.qvel[b.va]
        # Soft posture; exact foot acceleration constraint; strongly weighted balance.
        a=np.vstack([np.c_[np.zeros((29,6)),np.eye(29)],10*j[12:]])
        y=np.r_[posture,10*desired[12:]]
        h=a.T@a+np.eye(self.n)*1e-5
        kkt=np.block([[h,jc.T],[jc,np.zeros((12,12))]])
        accel=solve(kkt,np.r_[a.T@y,desired[:12]],assume_a='sym')[:self.n]
        mujoco.mj_fullM(m,d,self.mass)
        dyn=self.mass[np.ix_(self.v,self.v)]@accel+d.qfrc_bias[self.v]-d.qfrc_passive[self.v]
        # Resolve ground contact wrenches from the six unactuated equations.
        # Penalize moments so the solution favours forces within the foot support.
        invweight=np.diag(np.tile([1,1,1,.002,.002,.002],2))
        rootj=jc[:,:6].T
        wrench=invweight@rootj.T@solve(rootj@invweight@rootj.T,dyn[:6],assume_a='sym')
        self.last_wrench=wrench
        tau=dyn[6:]-jc[:,6:].T@wrench
        d.ctrl[b.aids]=np.clip(tau,m.jnt_actfrcrange[b.jids,0],m.jnt_actfrcrange[b.jids,1])
        h=self.hand
        htau=12*(self.g.plan.qpos[h.qa]-d.qpos[h.qa])-.3*d.qvel[h.va]+d.qfrc_bias[h.va]
        d.ctrl[h.aids]=np.clip(htau,m.jnt_actfrcrange[h.jids,0],m.jnt_actfrcrange[h.jids,1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--duration',type=float,default=16)
    p.add_argument('--knee',type=float,default=NEUTRAL_POSE['knee'])
    p.add_argument('--com-x',type=float)
    p.add_argument('--cube-x',type=float)
    p.add_argument('--cube-y',type=float)
    p.add_argument('--case-id',default='center')
    p.add_argument('--random-seed',type=int,default=0)
    p.add_argument('--lift',type=float,default=.18)
    args=p.parse_args()
    if args.knee != NEUTRAL_POSE['knee']:p.error('New episodes require the shared neutral knee configuration')
    if args.episode.exists():p.error('Episode already exists')
    m,d=load_scene(supported=False)
    config=read_cases();case=dict(config['cases'][case_index(config,'center')])
    if args.cube_x is not None:case['block_xy']=[args.cube_x,case['block_xy'][1]]
    if args.cube_y is not None:case['block_xy']=[case['block_xy'][0],args.cube_y]
    case['id']=args.case_id
    reset_case(m,d,case)
    fields=dataset_fields(m,d,case,args.episode.name,args.random_seed)
    bn,hn,mn=orders();b,h=JointMap(m,bn,29),JointMap(m,hn,14)
    # Legal one-time starting pose. All subsequent state changes use mj_step.
    for side in ('left','right'):
        for part,val in [('hip_pitch',-args.knee/2),('knee',args.knee),('ankle_pitch',-args.knee/2)]:
            d.qpos[m.joint(f'{side}_{part}_joint').qposadr[0]]=val
    arm_pose=neutral_arms(m,d)
    mujoco.mj_forward(m,d)
    feet=[i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name in ('left_ankle_roll_link','right_ankle_roll_link') and m.geom_contype[i]]
    rootqa=int(m.joint('floating_base_joint').qposadr[0])
    d.qpos[rootqa+2]-=min(d.geom_xpos[i,2]-m.geom_size[i,0] for i in feet)+.0001
    mujoco.mj_forward(m,d)
    assert m.neq==0 and m.jnt_type[m.joint('floating_base_joint').id]==mujoco.mjtJoint.mjJNT_FREE
    assert np.allclose(m.opt.gravity,[0,0,-9.81])
    initial_wrists={side:d.xpos[m.body(side+'_wrist_yaw_link').id].tolist() for side in ('left','right')}
    w=WholeBody(m,d,b,h,neutral=True);w.g.natural_start=True;w.g.lift_height=args.lift
    if args.com_x is not None:w.com_target[0]=args.com_x
    ev=PhysicalGraspEvaluator(m,d)
    args.episode.mkdir(parents=True)
    states={};refs=[];href=[];times=[];contacts=[];wrenches=[];failure=None
    steps=round(args.duration/m.opt.timestep);stride=round(.02/m.opt.timestep)
    try:
        for step in range(steps):
            if step%5==0:w.control(d.time)
            w.apply()
            if step%stride==0:
                times.append(d.time);refs.append(w.g.plan.qpos[b.qa].copy());href.append(w.g.plan.qpos[h.qa].copy())
                for key,value in robot_state(m,d,b,h).items():states.setdefault(key,[]).append(value)
                wrenches.append(w.last_wrench.copy())
            assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            physical=ev.step(d)
            if step%stride==0:contacts.append({'timestamp':float(d.time),**physical,'contacts':ev.last_contacts})
            if step%500==0:print(f'{d.time:.2f}s root={d.xpos[w.pelvis]} lift={physical["lift_m"]:.3f} hold={ev.hold:.3f}',flush=True)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or any(x.number for x in d.warning):raise RuntimeError('nonfinite_or_mujoco_warning')
            if d.xpos[w.pelvis,2]<.3:raise RuntimeError('robot_fell')
    except Exception as e:failure=f'{type(e).__name__}: {e}'
    t=np.array(times);states={k:np.array(v) for k,v in states.items()}
    report=ev.report(d);complete=failure is None and len(t)==round(args.duration*50)
    success=complete and report['physical_grasp_success'] and not report['fall_steps'] and not report['robot_self_contact_steps'] and not report['left_arm_environment_contact_steps']
    meta={**fields,'initial_pose':{'name':NEUTRAL_POSE['name'],'definition':NEUTRAL_POSE,'neutral_standard_compliant':True,'arm_joint_positions':arm_pose,'wrist_positions':initial_wrists},'body_joint_order':bn,'hand_joint_order':hn,'motor_command_joint_order':mn,
          'sampling_rate_hz':50,'quaternion_order':'wxyz','velocity_frame':'world','case':case,
          'source_controller':'contact-constrained whole-body inverse dynamics + wrist IK + independent hand PD',
          'source_success':bool(success),'pelvis_supported_during_source_generation':False,
          'source_root_height_m':float(states['root_pos'][0,2]),'source_trajectory':'freebase physical rollout',
          'expert_valid':False,'failure_reason':failure or ('sonic_replay_required' if success else 'source_physical_grasp_failed'),
          'elastic_band':False,'no_external_forces':True,'model_equality_count':m.neq,
          'gravity':m.opt.gravity.tolist(),'mapping':{'body':b.describe(),'hand':h.describe()},
          'sonic_config':{'encoder_mode':0},'source_arguments':vars(args)|{'episode':str(args.episode)},
          'environment_fingerprint':{'cube_friction':m.geom_friction[m.geom('task_cube_geom').id].tolist(),'table_size':m.geom_size[m.geom('task_table_top').id].tolist(),'light_pos':m.light_pos.tolist(),'light_dir':m.light_dir.tolist(),'light_diffuse':m.light_diffuse.tolist(),'camera_pos':m.cam_pos.tolist(),'camera_quat':m.cam_quat.tolist(),'neutral_definition':NEUTRAL_POSE},
          'generator_sha256':sha256(__file__),'shape':{k:list(v.shape) for k,v in states.items()}}
    # Actual source is the measured body reference, with synchronous hand commands.
    arrays={'timestamps':t,'body_ref_q':states['body_q'],'body_ref_dq':states['body_dq'],
            'hand_ref_q':np.array(href),'hand_ref_dq':np.gradient(np.array(href),t,axis=0),
            'root_ref_pos':states['root_pos'],'root_ref_quat':states['root_quat']}
    rootvel=states['root_lin_vel']
    ang=states['root_ang_vel']
    save_arrays(args.episode/'source',{'timestamps':t,**states,'body_target_q':np.array(refs),'contact_wrench_estimate':np.array(wrenches)})
    report.update(episode_completed=complete,source_success=bool(success),expert_valid=False,failure_reason=meta['failure_reason'])
    write_json(args.episode/'source/report.json',report);write_json(args.episode/'source/contacts.json',contacts)
    candidate=dict(meta,candidate_type='actual',root_velocities=(rootvel,ang))
    export_reference(args.episode/'candidates/actual',arrays,bn,hn,candidate)
    export_reference(args.episode/'sonic_reference',arrays,bn,hn,dict(meta,candidate_type='actual',root_velocities=(rootvel,ang)))
    save_arrays(args.episode,{**arrays,**states})
    for key,val in {**arrays,**states}.items():
        if val.ndim>2:continue
        names=bn if key.startswith('body_') else hn if key.startswith('hand_') else [key] if val.ndim==1 else [f'{key}_{i}' for i in range(val.shape[1])]
        write_csv(args.episode/f'{key}.csv',val[:,None] if val.ndim==1 else val,names)
    for name in ('info.txt','metadata.txt'):
        (args.episode/name).write_text((args.episode/'sonic_reference'/name).read_text())
    write_json(args.episode/'metadata.json',meta)
    write_json(args.episode/'joint_names.json',{'body':bn,'hand':hn})
    print(json.dumps(report,indent=2),flush=True)
    if not success:raise SystemExit(2)

if __name__=='__main__':main()
