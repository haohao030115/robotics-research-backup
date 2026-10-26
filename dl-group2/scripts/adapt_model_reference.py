"""Case-specific Cartesian deformation + bounded joint IK of a qualified model reference.
No simulator stepping, cube manipulation, model relabeling, or whole-reference translation.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scene import load_scene
from cases import reset_case
from project_paths import SONIC_REPO,ROBOT_RELATIVE,ROOT
from expert_trajectory import orders,JointMap,save_arrays,write_json,sha256
from readiness_grasp_reference import ReadinessGraspReference
from reference_io import export_reference

def main():
 p=argparse.ArgumentParser();p.add_argument('--entry-blend',type=float,default=6.);p.add_argument('--source',type=Path,required=True);p.add_argument('--episode',type=Path,required=True);p.add_argument('--case-json',required=True);p.add_argument('--producer',choices=['kimodo','ardy'],required=True);p.add_argument('--allow-historical-source',action='store_true');p.add_argument('--source-replay-report',type=Path);args=p.parse_args()
 if args.episode.exists():raise ValueError('Output exists')
 sm=json.loads((args.source/'metadata.json').read_text());sv=json.loads((args.source/'reference_validation.json').read_text());sp=json.loads((args.source/'provenance.json').read_text())
 producer_matches=sm.get('producer_type', 'kimodo' if sm.get('schema_name')=='kimodo_expert_reference' else None)==args.producer
 historically_qualified=(sp.get('historical_physical_success') is True and
                         sp.get('historical_max_lift',0.)>=.10 and
                         sp.get('historical_hold_seconds',0.)>=2.)
 source_qualified=sv.get('physical_success') is True or (args.allow_historical_source and historically_qualified)
 if not producer_matches or not source_qualified:raise ValueError('Source must be correctly labeled and qualified')
 latest_source_replay=None
 if args.source_replay_report:
  latest_source_replay=json.loads(args.source_replay_report.read_text())
 with np.load(args.source/'reference.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 case=json.loads(args.case_json);arm=case['task_arm'];case={k:v for k,v in case.items() if k!='task_arm'}
 bn,hn,_=orders();m,d=load_scene(robot_path=SONIC_REPO/ROBOT_RELATIVE);reset_case(m,d,case);b,h=JointMap(m,bn,29),JointMap(m,hn,14);rqa=int(m.joint('floating_base_joint').qposadr[0])
 sc=sm.get('case',sp.get('task_config',{}).get('case',{}));source_xy=sc.get('block_xy',sm.get('cube_initial_pos',[])[:2])
 if len(source_xy)!=2:raise ValueError('Source XY missing')
 source_arm=sm.get('task_arm',sp.get('task_arm','right'));mirror=source_arm!=arm
 def other(name):return name.replace('left_','TEMP_').replace('right_','left_').replace('TEMP_','right_')
 if mirror:
  for key,names,jm in [('body_ref_q',bn,b),('hand_ref_q',hn,h)]:
   old=a[key].copy()
   for j,n in enumerate(names):
    origin=other(n);origin=origin if origin in names else n
    axis=m.jnt_axis[jm.jids[j]];sign=1. if abs(axis[1])>.5 else -1.
    a[key][:,j]=sign*old[:,names.index(origin)]
  a['root_ref_pos'][:,1]*=-1
  a['root_ref_quat'][:,[1,3]]*=-1
  source_xy=[source_xy[0],-source_xy[1]]
 delta=np.r_[np.array(case['block_xy'])-source_xy,0.]
 d.qpos[b.qa]=a['body_ref_q'][0];d.qpos[h.qa]=a['hand_ref_q'][0];d.qpos[rqa:rqa+3]=a['root_ref_pos'][0];d.qpos[rqa+3:rqa+7]=a['root_ref_quat'][0];mujoco.mj_forward(m,d)
 planner=ReadinessGraspReference(m,d,arm,dict(position_m=0,orientation_rad=0,speed_mps=0));qpos=[];targets=[];errors=[]
 t=a['time_s'];u=np.clip((t-1)/args.entry_blend,0,1);u=u*u*(3-2*u)
 # Blend from unchanged reset to the case-specific endpoint, solving actual model IK every frame.
 for i in range(len(t)):
  d.qpos[b.qa]=a['body_ref_q'][i];d.qpos[h.qa]=a['hand_ref_q'][i];d.qpos[rqa:rqa+3]=a['root_ref_pos'][i];d.qpos[rqa+3:rqa+7]=a['root_ref_quat'][i];mujoco.mj_forward(m,d)
  goal=d.xpos[planner.wrist].copy()+u[i]*delta;planner.rotation=d.xmat[planner.wrist].reshape(3,3).copy();planner.plan.qpos[:]=d.qpos;planner.ik(goal);mujoco.mj_forward(m,planner.plan)
  error=float(np.linalg.norm(planner.plan.xpos[planner.wrist]-goal));errors.append(error)
  if error>.02:raise ValueError(f'Case IK unreachable at {i}: {error}')
  a['body_ref_q'][i]=planner.plan.qpos[b.qa];qpos.append(planner.plan.qpos.copy());targets.append(planner.plan.xpos[planner.wrist].copy())
 a['body_ref_dq']=np.gradient(a['body_ref_q'],t,axis=0);arrays=dict(a,timestamps=t,hand_ref_dq=np.gradient(a['hand_ref_q'],t,axis=0))
 args.episode.mkdir(parents=True);save_arrays(args.episode/'source',dict(timestamps=t,qpos=np.asarray(qpos),eef_pos=np.asarray(targets)))
 meta=dict(body_joint_order=bn,hand_joint_order=hn,case=case,task_arm=arm,initial_pose={'name':'neutral_standing_v1'},source_success=True,source_success_scope='historically qualified source only; every new case requires independent current canonical replay',expert_valid=False,pelvis_supported_during_source_generation=False,source_root_height_m=float(a['root_ref_pos'][0,2]),source_trajectory='model reference with case-specific Cartesian deformation and per-frame IK',sonic_config={'encoder_mode':0},candidate_type='actual',producer_type=args.producer,source_reference=str(args.source),source_reference_sha256=sha256(args.source/'reference.npz'),source_provenance=sp,adaptation=dict(method='smooth case-endpoint deformation and actual-model IK',entry_blend_s=args.entry_blend,mirrored_kinematic_retarget=mirror,new_model_sampling_calls=0,cached_model_output=True,max_ik_error_m=max(errors),body_non_arm_motion_preserved=True,reset_preserved=True,adapter_sha256=sha256(__file__),source_qualification_basis='historical_physical_success_only' if not sv.get('physical_success') else 'current_reference_validation',source_latest_canonical_replay=latest_source_replay),failure_reason='new_case_requires_canonical_replay')
 export_reference(args.episode/'candidates/actual',arrays,bn,hn,dict(meta,root_velocities=(np.gradient(a['root_ref_pos'],t,axis=0),np.zeros((len(t),3)))))
 write_json(args.episode/'metadata.json',meta);shutil.copy2(args.episode/'candidates/actual/info.txt',args.episode/'info.txt');print(json.dumps(meta['adaptation']))
if __name__=='__main__':main()
