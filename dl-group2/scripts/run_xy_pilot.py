#!/usr/bin/env python3
"""Bounded, resumable XY pilot. One DDS replay at a time; retain every attempt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from episode_schema import validate_dataset_metadata, validate_images
from audit_neutral_pose import audit
from expert_trajectory import write_json
from project_paths import ROOT


def read(path):
    return json.loads(path.read_text())


def execute(script, args, log, timeout=600):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        try:
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],
                                  cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            return result.returncode
        except subprocess.TimeoutExpired:
            return 124


def summarize(base, cases):
    rows=[]
    for case_id,xy in cases:
        ep=base/f'pilot_xy_{case_id}'
        if not (ep/'metadata.json').exists():continue
        meta=read(ep/'metadata.json')
        attempts=[]
        for path in sorted((ep/'validation').glob('*/report.json')):
            r=read(path);rm=read(path.parent/'metadata.json')
            attempts.append(dict(run=path.parent.name,candidate=rm['candidate_type'],
                expert_valid=r['expert_valid'],failure_reason=r['failure_reason'],
                reference_corrected=bool(rm.get('refinement'))))
        validate_dataset_metadata(meta)
        selected=meta.get('selected_validation_run')
        report=read(ep/'validation'/selected/'report.json') if selected else read(ep/'source/report.json')
        if not meta['expert_valid'] and attempts:
            report=read(ep/'validation'/attempts[-1]['run']/'report.json')
        row=dict(case_id=case_id,episode_id=ep.name,cube_initial_position=meta['task_config']['cube_initial_position'],
                 attempts=len(attempts),attempt_count_definition='number of completed SONIC replay attempts; source attempts recorded separately',
                 source_attempts=1,expert_valid=meta['expert_valid'],
                 final_lift=report['final_lift_m'],hold_seconds=report['final_continuous_hold_seconds'],
                 robot_table_collision=report['robot_table_contact_steps']>0,
                 robot_table_contact_steps=report['robot_table_contact_steps'],
                 robot_table_contact_bodies=report['robot_table_contact_bodies'],
                 robot_self_contact_steps=report.get('robot_self_contact_steps'),
                 left_arm_environment_contact_steps=report.get('left_arm_environment_contact_steps'),
                 correction_iterations=len({a['candidate'] for a in attempts if a['reference_corrected']}),
                 selected_candidate=meta.get('selected_candidate'),selected_validation_run=selected,
                 metrics_validation_run=(selected if meta['expert_valid'] else attempts[-1]['run'] if attempts else None),
                 failure_reason=[] if meta['expert_valid'] else report.get('failure_reason'),
                 attempt_history=attempts)
        meta['pilot']=row
        write_json(ep/'metadata.json',meta)
        rows.append(row)
    n=sum(r['attempts'] for r in rows);wins=sum(a['expert_valid'] for r in rows for a in r['attempt_history'])
    summary=dict(schema_version=3,scope='neutral standing + cube XY only; other scene parameters fixed',
        cases=rows,positions_completed=len(rows),valid_positions=sum(r['expert_valid'] for r in rows),
        sonic_replay_attempts=n,sonic_replay_successes=wins,sonic_replay_success_rate=wins/n if n else None,
        direct_success_positions=[r['case_id'] for r in rows if r['attempt_history'] and r['attempt_history'][0]['expert_valid']],
        development_archive='pilot_development (excluded from training; pre-standard pose attempts)',
        training_episodes=[r['episode_id'] for r in rows if r['expert_valid']],
        failed_episodes=[r['episode_id'] for r in rows if not r['expert_valid']])
    development=base/'pilot_development'
    archived_replays=list(development.glob('**/validation/*/report.json'))
    archived_sources=list(development.glob('**/source/report.json'))
    summary['development_attempts']={'source_rollouts':len(archived_sources),
        'sonic_replays':len(archived_replays),'excluded_from_training':True,
        'note':'Includes unsuccessful neutral designs and their position tests; not hidden from cost accounting.'}
    summary['total_sonic_replays_including_development']=n+len(archived_replays)
    write_json(base/'pilot_xy_summary.json',summary)
    lines=['# Neutral standing / XY pilot', '',
        'All positions use the same mirrored neutral pose; only initial cube XY varies.', '',
        '| Position | XY (m) | SONIC attempts | Corrections | Expert valid | Final lift (cm) | Final hold (s) |',
        '|---|---|---:|---:|---|---:|---:|']
    for row in rows:
        x,y,_=row['cube_initial_position']
        lines.append(f"| {row['case_id']} | ({x:.2f}, {y:.2f}) | {row['attempts']} | {row['correction_iterations']} | {row['expert_valid']} | {row['final_lift']*100:.2f} | {row['hold_seconds']:.3f} |")
    lines += ['',f"Final-standard SONIC success: {wins}/{n}. Eligible positions: {summary['valid_positions']}/{len(rows)}.",
        '', 'Attempt counts include independent confirmations. Pose development attempts are recorded separately in the JSON summary and remain excluded from training.',
        '', 'Robot/table contacts are reported without filtering in the JSON. Refer to the selected full-rate physical report for contact bodies, self-contact and left-arm safety.',
        '', 'RGB is under each selected validation run: vision/images.npz plus three MP4 files. Reference/state: 50 Hz; RGB: 10 Hz, indexed by reference_frame and timestamps.']
    (base/'PILOT_XY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-replays',type=int,default=4)
    parser.add_argument('--corners-if-stable',action='store_true')
    args=parser.parse_args()
    if not 1<=args.max_replays<=6:parser.error('Pilot replay budget must be 1..6')
    base=ROOT/'outputs/expert_episodes'
    origin=read(base/'freebase_wbc_002/metadata.json')['task_config']['cube_initial_position']
    offsets=[('center',0,0),('xp02',.02,0),('xm02',-.02,0),('yp02',0,.02),('ym02',0,-.02)]
    cases=[(name,[origin[0]+x,origin[1]+y]) for name,x,y in offsets]
    index=0
    while index<len(cases):
        case_id,xy=cases[index];ep=base/f'pilot_xy_{case_id}';logs=base/'pilot_logs'/case_id
        print(f'Start {case_id} {xy}',flush=True)
        if not ep.exists():
            execute('generate_freebase_expert.py',['--episode',ep,'--case-id',case_id,'--cube-x',xy[0],'--cube-y',xy[1],'--random-seed',0],logs/'source.log')
        if not (ep/'metadata.json').exists():raise RuntimeError(f'Source produced no metadata: {ep}; see {logs}')
        meta=read(ep/'metadata.json')
        if meta['source_success']:
            candidate='actual'
            for attempt in range(1,args.max_replays+1):
                if read(ep/'metadata.json')['expert_valid']:break
                run=f'sonic_{attempt:03d}';report_path=ep/'validation'/run/'report.json'
                if not report_path.exists():
                    execute('run_sonic_grasp.py',['--episode',ep,'--candidate',candidate,'--run-name',run,'--no-images'],logs/f'{run}.log')
                if not report_path.exists():raise RuntimeError(f'Missing replay report {report_path}; see {logs}')
                report=read(report_path)
                if report['expert_valid']:break
                if not report['frame_coverage_complete']:break
                if attempt<args.max_replays:
                    candidate=f'corrected_{attempt:03d}'
                    if not (ep/'candidates'/candidate).exists():
                        code=execute('refine_freebase_reference.py',['--episode',ep,'--run',run,'--name',candidate,'--gain',.6],logs/f'{candidate}.log')
                        if code:break
        summary=summarize(base,cases)
        print(json.dumps(next(row for row in summary['cases'] if row['case_id']==case_id)),flush=True)
        # Save RGB for every physical replay, including failures, without changing states.
        for report_path in sorted((ep/'validation').glob('*/report.json')):
            folder=report_path.parent
            if read(report_path)['recorded_frames'] and not (folder/'vision').exists():
                code=execute('render_expert_rollout.py',['--episode',ep,'--rollout',f'validation/{folder.name}'],logs/f'{folder.name}_render.log')
                if code:raise RuntimeError(f'Rendering failed: {folder}')
        if read(ep/'metadata.json')['expert_valid']:
            chosen=read(ep/'metadata.json')['selected_validation_run']
            neutral_audit=audit(ep,f'validation/{chosen}')
            if not neutral_audit['initial_pose_geometry_valid']:raise RuntimeError(f'Neutral geometry invalid: {ep}')
            for script in ['export_expert_episode.py','verify_expert_episode.py']:
                code=execute(script,['--episode',ep],logs/f'{script}.log')
                if code:raise RuntimeError(f'{script} failed: {ep}')
        elif read(ep/'metadata.json').get('selected_validation_run'):
            last=sorted((ep/'validation').glob('*/report.json'))[-1]
            if read(last)['frame_coverage_complete']:
                from verify_expert_episode import select_saved_run
                select_saved_run(ep,last.parent.name)
                code=execute('export_expert_episode.py',['--episode',ep,'--include-failed'],logs/'export_failed.log')
                if code:raise RuntimeError(f'Failed-rollout inspection export failed: {ep}')
                code=execute('verify_expert_episode.py',['--episode',ep],logs/'verify_failed.log')
                if code!=2:raise RuntimeError(f'Failed-rollout verification did not preserve invalid status: {ep}')
        index+=1
        if index==5 and args.corners_if_stable:
            summary=summarize(base,cases)
            if len(summary['cases'])==5 and all(r['expert_valid'] and r['attempts']<=2 for r in summary['cases']):
                cases.extend((name,[origin[0]+x,origin[1]+y]) for name,x,y in [
                    ('xp02_yp02',.02,.02),('xp02_ym02',.02,-.02),('xm02_yp02',-.02,.02),('xm02_ym02',-.02,-.02)])
    summary=summarize(base,cases)
    training=[]
    for row in summary['cases']:
        if not row['expert_valid']:continue
        ep=base/row['episode_id'];meta=read(ep/'metadata.json')
        alignment=validate_images(ep/'validation'/meta['selected_validation_run'],np.load(ep/'timestamps.npy'))
        training.append(dict(episode_id=ep.name,path=ep.name,case_id=row['case_id'],
                             selected_validation_run=meta['selected_validation_run'],rgb_alignment=alignment))
    write_json(base/'pilot_xy_training_manifest.json',dict(schema_version=3,split='pilot_training_eligible',
        excluded_development_archive='pilot_development',episodes=training,
        note='Episode-level eligible pool; no frame-wise train/test split is created.'))
    failures=[dict(episode_id=row['episode_id'],run=attempt['run'],
        path=f"{row['episode_id']}/validation/{attempt['run']}",expert_valid=False,
        failure_reason=attempt['failure_reason']) for row in summary['cases']
        for attempt in row['attempt_history'] if not attempt['expert_valid']]
    write_json(base/'pilot_xy_failure_manifest.json',dict(schema_version=3,excluded_from_training=True,
        failed_final_standard_runs=failures,development_archive='pilot_development',
        development_attempts=summary['development_attempts']))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
