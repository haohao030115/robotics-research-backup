#!/usr/bin/env python3
"""Publish a validated SONIC+hand episode to the dataset's CSV schema.

Certification requires a complete unassisted source and replay. Failed complete
replays may be exported for inspection with --include-failed, retaining invalid status. Preserves all source, failed
candidate and trial artifacts. SONIC reference bytes come from the validated
candidate, not regenerated joint/root trajectories.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from expert_trajectory import load_reference,write_json,sha256
from reference_io import write_csv
from verify_expert_episode import verify
from episode_schema import validate_dataset_metadata, validate_images


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--include-failed',action='store_true',help='Export a complete failed replay for inspection; never certify it')
    args=p.parse_args()
    ep=args.episode.resolve();summary=verify(ep, check_csv=False)
    if not summary['expert_valid'] and not args.include_failed:p.error('No validated expert')
    meta=json.loads((ep/'metadata.json').read_text());run=ep/'validation'/meta['selected_validation_run']
    validate_dataset_metadata(meta)
    image_audit=validate_images(run,np.load(ep/'timestamps.npy'))
    trial=json.loads((run/'metadata.json').read_text())
    report=json.loads((run/'report.json').read_text())
    if not report['frame_coverage_complete']:p.error('Complete replay required for rectangular dataset export')
    candidate=ep/'candidates'/meta['selected_candidate']
    ref,_=load_reference(candidate);arrays=dict(ref)
    actual_keys=['body_q','body_dq','hand_q','hand_dq','root_pos','root_quat','root_lin_vel','root_ang_vel','cube_pos','cube_quat','eef_pos','eef_quat','qpos','qvel','ctrl','sim_timestamps']
    for key in actual_keys:
        a=np.load(run/f'{key}.npy');assert len(a)==len(ref['timestamps']) and np.isfinite(a).all();arrays[key]=a
        np.save(ep/f'{key}.npy',a)
    for key,value in arrays.items():
        names=meta['body_joint_order'] if key in ['body_ref_q','body_ref_dq','body_q','body_dq'] else meta['hand_joint_order'] if key in ['hand_ref_q','hand_ref_dq','hand_q','hand_dq'] else [key] if value.ndim==1 else [f'{key}_{i}' for i in range(value.shape[1])]
        write_csv(ep/f'{key}.csv',value[:,None] if value.ndim==1 else value,names)
    sonic=ep/'sonic_reference';sonic.mkdir(exist_ok=True)
    for name in ['joint_pos.csv','joint_vel.csv','body_pos.csv','body_quat.csv','body_lin_vel.csv','body_ang_vel.csv','metadata.txt','info.txt']:
        shutil.copy2(candidate/name,sonic/name)
    text=(sonic/'info.txt').read_text()
    if summary['expert_valid']:
        text=text.replace('expert_valid: false (requires physical SONIC+hand validation)','expert_valid: true (complete unassisted physical source and SONIC+hand replay validated)')
    else:
        text+='\nexpert_valid: false; inspection only; excluded from training\n'
    (sonic/'info.txt').write_text(text+f'validated_run: {run.name}\n')
    meta.update(shape={k:list(v.shape) for k,v in arrays.items()},
                sampling_rate_hz=50,quaternion_order='wxyz',velocity_frame='world',
                actual_rollout=str(run),source_controller=trial['source_controller'],
                reference_refinement=trial.get('refinement'),
                initialization_support=trial['initialization_support'],
                body_startup_controller=trial['body_startup_controller'],
                hand_controller={'type':'independent torque PD','kp':12.,'kd':.3,'target_dq_used':False},
                success_metrics=json.loads((run/'report.json').read_text()),
                sonic_reference_root_velocities='source position/quaternion kinematic derivatives; actual velocities use world Jacobians',
                sync_clock=trial['sync_clock'],
                numerical_state_finite=True,episode_completed=report['episode_completed'],
                exporter_sha256=sha256(__file__),
                sonic_reference_path=str(sonic),
                runtime_config={'ENABLE_ELASTIC_BAND':trial['elastic_band']!='off',
                                'pelvis_weld':trial['pelvis_supported_during_final_rollout']!='none',
                                'external_forces':trial['elastic_band']!='off' or trial['initialization_support']!='none'},
                vision={'path':str(run/'vision'),'cameras':['head_rgb','wrist_rgb','task_overview'],
                        'sampling_rate_hz':10,'rgb_shape':[image_audit['frames'],240,320,3],
                        'frame_alignment':'images.npz reference_frame indexes reference/actual arrays'})
    write_json(ep/'joint_names.json',{'body':meta['body_joint_order'],'hand':meta['hand_joint_order']})
    write_json(ep/'metadata.json',meta);write_json(sonic/'metadata.json',meta)
    (ep/'info.txt').write_text(text+f'validated_run: {run.name}\n')
    write_json(ep/'csv_manifest.json',{str(f.relative_to(ep)):sha256(f) for f in sorted(ep.glob('*.csv'))+sorted(sonic.glob('*.csv'))})
    print(f'Exported episode: {ep}; T={len(ref["timestamps"])}; expert_valid={summary["expert_valid"]}')
if __name__=='__main__':main()
