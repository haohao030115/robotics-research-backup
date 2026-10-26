"""Export only a newly replay-certified final reference; failures never enter episodes/."""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
from scene import load_scene
from project_paths import SONIC_REPO,ROBOT_RELATIVE,ROOT
from expert_trajectory import orders,JointMap,sha256,write_json

def export(ep,out,episode_id,producer):
 meta=json.loads((ep/'metadata.json').read_text());run=ep/'validation'/meta['selected_validation_run'];report=json.loads((run/'report.json').read_text())
 if not report.get('expert_valid') or not report.get('episode_completed') or report['failure_reason']:raise ValueError('Actual replay not certified')
 candidate=ep/'candidates'/meta['selected_candidate'];a={k:np.load(candidate/(v+'.npy'),allow_pickle=False) for k,v in dict(time_s='timestamps',body_ref_q='body_ref_q',body_ref_dq='body_ref_dq',root_ref_pos='root_ref_pos',root_ref_quat='root_ref_quat',hand_ref_q='hand_ref_q').items()}
 bn,hn,_=orders();m,d=load_scene(robot_path=SONIC_REPO/ROBOT_RELATIVE)
 for key,names in [('body_ref_q',bn),('hand_ref_q',hn)]:
  j=JointMap(m,names,len(names));q=a[key]
  if not np.all((q>=m.jnt_range[j.jids,0]-1e-8)&(q<=m.jnt_range[j.jids,1]+1e-8)):raise ValueError('Reference joint limit violation '+key)
 t=a['time_s'];n=len(t)
 if not all(np.isfinite(x).all() and len(x)==n for x in a.values()) or not np.allclose(np.diff(t),.02,atol=1e-8):raise ValueError('Array/timeline invalid')
 if not np.allclose(np.linalg.norm(a['root_ref_quat'],axis=1),1,atol=1e-5):raise ValueError('Quaternion invalid')
 out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'reference.npz',**a)
 metadata=dict(schema_name='canonical_expert_reference',schema_version=1,episode_id=episode_id,producer_type=producer,task_arm=meta['task_arm'],case=meta['case'],reference_hz=50,num_frames=n,duration_s=float(t[-1]-t[0]),body_dof=29,hand_dof=14,body_joint_names=bn,hand_joint_names=hn,root_quat_order='wxyz',quaternion_convention='wxyz',root_pose_frame='world',units=dict(time='s',position='m',joint_position='rad',joint_velocity='rad/s'),robot='G1',scene='canonical_tabletop',initial_pose='canonical_reset',cube_initial_pos=[*meta['case']['block_xy'],.785],reference_complete=True)
 provenance=dict(producer_type=producer,generator_name='readiness_rule_wbc' if producer=='rule_based' else 'qualified_model_reference_case_IK_adaptation',source_metadata=meta,source_reference_candidate=str(candidate),actual_replay=str(run),actual_replay_report_sha256=sha256(run/'report.json'),final_reference_sha256=sha256(out/'reference.npz'),generation_code_sha256={f.name:sha256(f) for f in ROOT.glob('scripts/*rule*.py')},body_velocity_source='selected candidate; measured source velocity or finite difference after IK correction',new_model_sampling_calls=0 if producer!='rule_based' else None,rule_specific_runtime_adapter_required=False,downstream_compiler_run=False)
 validation=dict(reference_schema_valid=True,finite=True,timestamps_monotonic=True,joint_limits_valid=True,quaternion_valid=True,reference_complete=True,tested_with_canonical_replay=True,physical_success=True,replay_complete=report['episode_completed'],max_lift=report['max_lift_m'],hold_duration=report['max_continuous_hold_seconds'],final_hold_duration=report['final_continuous_hold_seconds'],reference_sha256=sha256(out/'reference.npz'),replay_report_path=str(run/'report.json'),criterion=report['criterion'],task_arm=meta['task_arm'],downstream_compiler_accepted=False)
 for name,value in [('metadata',metadata),('provenance',provenance),('reference_validation',validation)]:write_json(out/(name+'.json'),value)
 return dict(episode_id=episode_id,producer_type=producer,block_xy=meta['case']['block_xy'],task_arm=meta['task_arm'],physical_success=True,max_lift=report['max_lift_m'],hold_duration=report['max_continuous_hold_seconds'],reference_path=str(out),source_reference_sha256=meta.get('source_reference_sha256'),replay_report_path=str(run/'report.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--producer',required=True);x=p.parse_args();print(export(x.episode,x.out,x.out.name,x.producer))
