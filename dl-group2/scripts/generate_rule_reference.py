"""Independent rule source in canonical geometry; never imports an existing trajectory."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scene import load_scene
from project_paths import SONIC_REPO,ROBOT_RELATIVE,ROOT
from cases import reset_case
from expert_trajectory import orders,JointMap,PhysicalGraspEvaluator,robot_state,write_json,save_arrays,sha256
from generate_freebase_expert import WholeBody
from readiness_grasp_reference import ReadinessGraspReference
from episode_schema import neutral_arms
from reference_io import export_reference

def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--case-id',required=True);p.add_argument('--cube-x',type=float,required=True);p.add_argument('--cube-y',type=float,required=True);p.add_argument('--task-arm',choices=['left','right'],required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--approach-scale',type=float,default=1.);p.add_argument('--wrist-target-offset-y',type=float,default=0.);p.add_argument('--waist-yaw-target',type=float,default=0.);p.add_argument('--duration',type=float,default=18.);args=p.parse_args()
 if args.episode.exists():raise ValueError('Output already exists')
 case=dict(id=args.case_id,block_xy=[args.cube_x,args.cube_y],block_yaw_deg=0.)
 calibration=json.loads(args.calibration.read_text());m,d=load_scene(robot_path=SONIC_REPO/ROBOT_RELATIVE,supported=False);reset_case(m,d,case)
 bn,hn,_=orders();b,h=JointMap(m,bn,29),JointMap(m,hn,14)
 for side in ('left','right'):
  for part,val in [('hip_pitch',-.15),('knee',.3),('ankle_pitch',-.15)]:d.qpos[m.joint(side+'_'+part+'_joint').qposadr[0]]=val
 neutral_arms(m,d);mujoco.mj_forward(m,d)
 feet=[i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name in ('left_ankle_roll_link','right_ankle_roll_link') and m.geom_contype[i]]
 rqa=int(m.joint('floating_base_joint').qposadr[0]);d.qpos[rqa+2]-=min(d.geom_xpos[i,2]-m.geom_size[i,0] for i in feet)+.0001;mujoco.mj_forward(m,d)
 w=WholeBody(m,d,b,h,neutral=True);w.g=ReadinessGraspReference(m,d,args.task_arm,calibration['thresholds'],args.approach_scale,wrist_target_offset_y=args.wrist_target_offset_y);w.g.plan.qpos[m.joint('waist_yaw_joint').qposadr[0]]=args.waist_yaw_target
 ev=PhysicalGraspEvaluator(m,d,task_arm=args.task_arm);args.episode.mkdir(parents=True);states={};href=[];times=[];failure=None
 stride=round(.02/m.opt.timestep)
 try:
  for step in range(round(args.duration/m.opt.timestep)):
   if step%5==0:w.control(d.time)
   w.apply()
   if step%stride==0:
    times.append(step*m.opt.timestep);href.append(w.g.plan.qpos[h.qa].copy())
    state=robot_state(m,d,b,h);state['eef_pos']=d.xpos[w.g.wrist].copy();state['eef_quat']=d.xquat[w.g.wrist].copy()
    for k,v in state.items():states.setdefault(k,[]).append(v)
   if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or m.neq:raise RuntimeError('External support')
   mujoco.mj_step(m,d);mujoco.mj_forward(m,d);metrics=ev.step(d)
   if step%1000==0:print(round(d.time,2),w.g.stage,metrics['lift_m'],metrics['hold_seconds'],w.g.last_metrics,flush=True)
   if not np.isfinite(d.qpos).all() or any(x.number for x in d.warning) or d.xpos[m.body('pelvis').id,2]<.3:raise RuntimeError('Invalid dynamics/fall')
 except Exception as e:failure=repr(e)
 t=np.asarray(times);states={k:np.asarray(v) for k,v in states.items()};report=ev.report(d)
 complete=failure is None and len(t)==round(args.duration*50)
 valid=bool(complete and report['physical_grasp_success'] and not report['fall_steps'] and not report['robot_self_contact_steps'] and not report['inactive_arm_environment_contact_steps'] and w.g.closed_at is not None)
 arrays=dict(timestamps=t,body_ref_q=states['body_q'],body_ref_dq=states['body_dq'],root_ref_pos=states['root_pos'],root_ref_quat=states['root_quat'],hand_ref_q=np.asarray(href),hand_ref_dq=np.gradient(np.asarray(href),t,axis=0))
 meta=dict(body_joint_order=bn,hand_joint_order=hn,case=case,task_arm=args.task_arm,initial_pose={'name':'neutral_standing_v1'},source_success=valid,expert_valid=False,pelvis_supported_during_source_generation=False,source_root_height_m=float(states['root_pos'][0,2]),source_trajectory='independent state-conditioned rule WBC source',sonic_config={'encoder_mode':0},candidate_type='actual',failure_reason=failure or 'canonical_replay_required',source_arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},readiness_calibration=calibration,closure_events=w.g.events,source_hashes={name:sha256(ROOT/'scripts'/name) for name in ['generate_rule_reference.py','readiness_grasp_reference.py','grasp_reference.py','generate_freebase_expert.py']})
 report.update(source_success=valid,episode_completed=complete,failure_reason=failure,closure_events=w.g.events)
 save_arrays(args.episode/'source',dict(timestamps=t,**states));write_json(args.episode/'source/report.json',report)
 export_reference(args.episode/'candidates/actual',arrays,bn,hn,dict(meta,root_velocities=(states['root_lin_vel'],states['root_ang_vel'])))
 (args.episode/'info.txt').write_text((args.episode/'candidates/actual/info.txt').read_text());write_json(args.episode/'metadata.json',meta);print(json.dumps(report),flush=True)
 if not valid:sys.exit(2)
if __name__=='__main__':main()
