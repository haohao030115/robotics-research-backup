#!/usr/bin/env python3
"""Resumable, bounded XY-only expert production; failures are first-class records."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from project_paths import ROOT

from episode_schema import validate_dataset_metadata, validate_images
from verify_expert_episode import verify
from audit_neutral_pose import audit


def write_json(path,data):
    """Atomic progress/checkpoint publication on the shared filesystem."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def command(script,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as out:
        return subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],
            cwd=ROOT,stdout=out,stderr=subprocess.STDOUT).returncode


def manifest_entry(ep,base):
    meta=read(ep/'metadata.json');validate_dataset_metadata(meta)
    checked=verify(ep)
    if not checked['expert_valid']:raise ValueError('Not a physically certified expert')
    run=ep/'validation'/meta['selected_validation_run']
    ts=np.load(ep/'timestamps.npy');alignment=validate_images(run,ts)
    with np.load(run/'vision/images.npz') as z:frames=z['reference_frame'].tolist()
    if frames!=list(range(0,len(ts),5)) or not np.allclose(np.diff(ts),.02,atol=1e-9,rtol=0):raise ValueError('Incomplete 10 Hz / 50 Hz alignment')
    return dict(episode_id=ep.name,episode_path=os.path.relpath(ep,base),expert_valid=True,
        schema_version=3,cube_initial_xy=meta['task_config']['cube_initial_position'][:2],
        selected_validation_rollout=meta['selected_validation_run'],state_reference_length=len(ts),
        rgb_path=os.path.relpath(run/'vision/images.npz',base),rgb_reference_frame_indices=frames,
        instruction=meta['instruction'],generation_method=meta.get('generation_method','unknown'),
        replay_attempts=meta.get('replay_attempts'),correction_iterations=meta.get('correction_iterations'),
        source_candidate=meta.get('source_candidate','actual'),selected_candidate=meta['selected_candidate'],
        state_keys=['body_q','hand_q'],target_keys=['body_ref_q','hand_ref_q'],
        policy_camera='head_rgb',optional_policy_camera='wrist_rgb',
        excluded_default_inputs=['cube_pos','cube_quat','task_overview'],alignment_checked=True)


def produce(ep,sample,args):
    logs=ep.parent.parent/'logs'/ep.name
    started=time.time()
    if not (ep/'metadata.json').exists():
        if ep.exists():return dict(**sample,episode_id=ep.name,expert_valid=False,failure_reason=['incomplete_source_directory'],replay_attempts=0,correction_iterations=0,generation_method='failed_sample')
        rc=command('generate_freebase_expert.py',['--episode',ep,'--case-id',sample['case_id'],
            '--cube-x',sample['xy'][0],'--cube-y',sample['xy'][1],'--random-seed',args.seed],logs/'source.log')
        if not (ep/'metadata.json').exists():return dict(**sample,episode_id=ep.name,expert_valid=False,failure_reason=[f'source_process_exit_{rc}'],replay_attempts=0,correction_iterations=0,generation_method='failed_sample')
    meta=read(ep/'metadata.json')
    attempts=[];corrections=0;candidate='actual';failure=[]
    if meta['source_success']:
        for number in range(1,args.max_replays+1):
            run=f'replay_{number:03d}';folder=ep/'validation'/run
            if not (folder/'report.json').exists():
                if folder.exists():
                    failure=['incomplete_replay_directory'];break
                rc=command('run_sonic_grasp.py',['--episode',ep,'--candidate',candidate,'--run-name',run,'--no-images'],logs/f'{run}.log')
                if not (folder/'report.json').exists():
                    failure=[f'replay_process_exit_{rc}'];attempts.append(dict(run=run,candidate=candidate,expert_valid=False,failure_reason=failure));break
            r=read(folder/'report.json');rm=read(folder/'metadata.json')
            candidate=rm['candidate_type']
            attempts.append(dict(run=run,candidate=candidate,expert_valid=r['expert_valid'],failure_reason=r['failure_reason']))
            if r['expert_valid']:break
            failure=r['failure_reason']
            if not r['frame_coverage_complete']:break
            if corrections>=args.max_corrections or number>=args.max_replays:break
            # Safety/synchronization failures are not grasp calibration problems.
            if any(reason!='cube_lift_opposing_contact_and_2s_hold_not_met' for reason in failure):break
            corrections+=1;candidate=f'corrected_{corrections:03d}'
            if not (ep/'candidates'/candidate/'metadata.json').exists():
                rc=command('refine_freebase_reference.py',['--episode',ep,'--run',run,'--name',candidate,'--gain',.6],logs/f'{candidate}.log')
                if rc:failure=[f'correction_process_exit_{rc}'];break
    else:
        failure=[str(meta['failure_reason'])]
    meta=read(ep/'metadata.json');valid=bool(meta['expert_valid'])
    method='corrected_expert' if valid and corrections else 'direct_expert' if valid else 'failed_sample'
    meta.update(generation_method=method,correction_iterations=corrections,replay_attempts=len(attempts),
                source_candidate='actual',production_sample=sample)
    meta.setdefault('selected_candidate','actual')
    write_json(ep/'metadata.json',meta)
    # Retain RGB for failed complete/partial replay too; only selected success is eligible.
    packaging_failures=[]
    for attempt in attempts:
        folder=ep/'validation'/attempt['run']
        if not (folder/'report.json').exists() or not read(folder/'report.json')['recorded_frames']:continue
        if not (folder/'vision/images.npz').exists():
            rc=command('render_expert_rollout.py',['--episode',ep,'--rollout',f"validation/{attempt['run']}"],logs/f"{attempt['run']}_rgb.log")
            if rc:packaging_failures.append(f"rgb_export_failed:{attempt['run']}")
    if valid:
        try:
            geometry=audit(ep,f"validation/{meta['selected_validation_run']}")
            if not geometry['initial_pose_geometry_valid']:raise ValueError('neutral geometry check failed')
            rc=command('export_expert_episode.py',['--episode',ep],logs/'export.log')
            if rc:raise ValueError(f'export exit {rc}')
            checked=verify(ep)
            if not checked['expert_valid']:raise ValueError('dataset verification failed')
        except Exception as error:packaging_failures.append(str(error))
    selected=meta.get('selected_validation_run')
    latest=attempts[-1]['run'] if attempts else None
    report=read(ep/'validation'/(selected if valid else latest)/'report.json') if (selected if valid else latest) and (ep/'validation'/(selected if valid else latest)/'report.json').exists() else read(ep/'source/report.json')
    return dict(**sample,episode_id=ep.name,episode_path=str(ep),expert_valid=valid,
        training_eligible=valid and not packaging_failures,generation_method=method,
        correction_iterations=corrections,replay_attempts=len(attempts),attempt_history=attempts,
        source_candidate='actual',selected_candidate=meta.get('selected_candidate'),selected_validation_run=selected,
        failure_reason=[] if valid else failure,packaging_failures=packaging_failures,
        final_lift_m=report['final_lift_m'],hold_seconds=report['final_continuous_hold_seconds'],
        robot_table_contact_steps=report['robot_table_contact_steps'],robot_self_contact_steps=report.get('robot_self_contact_steps'),
        wall_seconds=time.time()-started)


def publish(base,plan,results):
    rows=list(results.values());eligible=[]
    for row in rows:
        if row.get('training_eligible'):
            eligible.append(row['manifest_entry'])
    summary=dict(schema_version=3,configuration=plan['configuration'],sampled_cases=len(plan['samples']),
        processed_cases=len(rows),valid_experts=sum(r.get('training_eligible',False) for r in rows),
        failed_cases=sum(not r.get('training_eligible',False) for r in rows),
        physical_failed_cases=sum(not r.get('expert_valid',False) for r in rows),
        packaging_failed_cases=sum(r.get('expert_valid',False) and not r.get('training_eligible',False) for r in rows),
        direct_success=sum(r.get('training_eligible',False) and r['generation_method']=='direct_expert' for r in rows),
        corrected_success=sum(r.get('training_eligible',False) and r['generation_method']=='corrected_expert' for r in rows),
        average_replay_attempts=float(np.mean([r['replay_attempts'] for r in rows])) if rows else 0,
        average_correction_iterations=float(np.mean([r['correction_iterations'] for r in rows])) if rows else 0,
        cases=rows)
    # Fixed spatial bins expose failures as well as successes, including empty bins.
    cfg=plan['configuration'];bins=[]
    for i in range(3):
        for j in range(3):
            xs=np.linspace(cfg['x_min'],cfg['x_max'],4);ys=np.linspace(cfg['y_min'],cfg['y_max'],4)
            members=[r for r in rows if xs[i]<=r['xy'][0] and (r['xy'][0]<xs[i+1] or i==2 and r['xy'][0]<=xs[i+1]) and ys[j]<=r['xy'][1] and (r['xy'][1]<ys[j+1] or j==2 and r['xy'][1]<=ys[j+1])]
            bins.append(dict(x_range=xs[i:i+2].tolist(),y_range=ys[j:j+2].tolist(),sampled=len(members),valid=sum(r.get('training_eligible',False) for r in members)))
    summary['xy_coverage']=bins
    for cell in bins:cell['success_rate']=cell['valid']/cell['sampled'] if cell['sampled'] else None
    sampled=np.asarray([sample['xy'] for sample in plan['samples']],float)
    valid_xy=np.asarray([row['xy'] for row in rows if row.get('training_eligible')],float)
    summary['sampled_xy_bounds']={'min':sampled.min(axis=0).tolist(),'max':sampled.max(axis=0).tolist()}
    summary['valid_xy_bounds']={'min':valid_xy.min(axis=0).tolist(),'max':valid_xy.max(axis=0).tolist()} if len(valid_xy) else None
    summary['total_replay_attempts']=sum(row['replay_attempts'] for row in rows)
    summary['total_correction_iterations']=sum(row['correction_iterations'] for row in rows)
    if (base/'region_selection.json').exists():summary['region_selection_evidence']='region_selection.json'
    write_json(base/'summary.json',summary)
    write_json(base/'training_manifest.json',dict(schema_version=3,episodes=eligible,default_observation=['head_rgb','body_q','hand_q','instruction'],targets=['body_ref_q','hand_ref_q']))
    write_json(base/'failure_manifest.json',dict(schema_version=3,cases=[r for r in rows if not r.get('training_eligible',False)]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-episodes',type=int,default=30)
    p.add_argument('--x-min',type=float,required=True);p.add_argument('--x-max',type=float,required=True)
    p.add_argument('--y-min',type=float,required=True);p.add_argument('--y-max',type=float,required=True)
    p.add_argument('--seed',type=int,default=20261002)
    p.add_argument('--max-replays',type=int,default=3);p.add_argument('--max-corrections',type=int,default=2)
    p.add_argument('--mode',choices=['random','corners'],default='random')
    a=p.parse_args()
    if a.num_episodes<1 or not 1<=a.max_replays<=6 or not 0<=a.max_corrections<a.max_replays:p.error('Invalid finite sampling/retry budget')
    if not np.isfinite([a.x_min,a.x_max,a.y_min,a.y_max]).all() or not a.x_min<a.x_max or not a.y_min<a.y_max:p.error('Invalid XY bounds')
    base=a.output.resolve();base.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(a).items() if k!='output'}
    plan_path=base/'sampling_plan.json'
    if plan_path.exists():
        plan=read(plan_path)
        if plan['configuration']!=config:p.error('Resume configuration differs from saved sampling plan')
    else:
        rng=np.random.default_rng(a.seed)
        points=np.array([[a.x_max,a.y_max],[a.x_max,a.y_min],[a.x_min,a.y_max],[a.x_min,a.y_min]]) if a.mode=='corners' else rng.uniform([a.x_min,a.y_min],[a.x_max,a.y_max],size=(a.num_episodes,2))
        names=['xp02_yp02','xp02_ym02','xm02_yp02','xm02_ym02'] if a.mode=='corners' else [f'{base.name}_random_{i:04d}' for i in range(a.num_episodes)]
        plan=dict(configuration=config,samples=[dict(case_id=n,xy=xy.tolist(),sample_index=i) for i,(n,xy) in enumerate(zip(names,points))])
        write_json(plan_path,plan)
    results_path=base/'results.json';results=read(results_path) if results_path.exists() else {}
    # DDS uses fixed loopback ports; never overlap two producer processes on a node.
    lock=open('/tmp/dl_group2_expert_production.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sample in plan['samples']:
        name=sample['case_id']
        if name not in results:
            print(f"START {name} {sample['xy']}",flush=True)
            ep=base/'episodes'/name
            try:row=produce(ep,sample,a)
            except Exception as error:
                row=dict(**sample,episode_id=name,episode_path=str(ep),expert_valid=False,training_eligible=False,
                    generation_method='failed_sample',failure_reason=[f'{type(error).__name__}: {error}'],
                    replay_attempts=len(list((ep/'validation').glob('*/report.json'))),correction_iterations=len(list((ep/'candidates').glob('corrected_*'))))
            if row.get('training_eligible'):row['manifest_entry']=manifest_entry(ep,base)
            results[name]=row;write_json(results_path,results)
            print(json.dumps({k:v for k,v in row.items() if k!='manifest_entry'}),flush=True)
        publish(base,plan,results)
    if read(base/'training_manifest.json')['episodes']:
        from audit_expert_dataset import audit as audit_dataset
        audit_dataset(base/'training_manifest.json')
    else:
        write_json(base/'dataset_audit.json',{'passed':False,'failure_reason':'no_eligible_experts'})
    from report_expert_dataset import report as report_dataset
    report_dataset(base)
    print('COMPLETE '+str(base/'summary.json'),flush=True)


if __name__=='__main__':main()
