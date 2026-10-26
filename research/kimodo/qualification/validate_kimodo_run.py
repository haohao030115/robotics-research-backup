"""Independent read-back of recorded contact rows, state coverage and reference streams."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_export_sources(expected, recorded):
    try:
        unchanged=all(sha(p)==h for p,h in expected.items())
    except OSError:
        unchanged=False
    retained=all(recorded.get(p)==h for p,h in expected.items())
    return unchanged,retained


def validate(run):
    result=json.loads((run/'result.json').read_text())
    if result.get('physical_replay_run') is False:return dict(physical_success_recomputed=None,expert_valid_confirmed=None,checks={},status='NOT_RUN')
    provenance=json.loads((run/'provenance.json').read_text())
    rows=json.loads((run/'physics_steps.json').read_text())
    active=[r for r in rows if r['playing']]
    states=np.load(run/'simulation.npz',allow_pickle=False)
    reference=np.load(run/'input_reference.npz',allow_pickle=False)
    count=len(reference['timestamps'])
    expected_logs=['motion_playing.csv','action.csv','token_state.csv','encoder_mode.csv']
    missing=[name for name in expected_logs if not (run/'policy_logs'/name).is_file()]
    if not active:
        return dict(expert_valid_confirmed=False,physical_success_recomputed=False,
                    max_continuous_hold_seconds_recomputed=0.,checks={'complete_replay_evidence':False},
                    status='INCOMPLETE_REPLAY',missing_policy_logs=missing,runtime_failure_reasons=result.get('failure_reasons',[]))
    checks={}
    current=maximum=0.;begin=best_begin=best_end=None
    force_rows_consistent=True;timer_rows_consistent=True;physics_finite=True
    for i,r in enumerate(active):
        forces=[float(c['normal_force_n']) for c in r['contacts']]
        physics_finite &= bool(np.isfinite([r['sim_time'],r['lift_m'],*forces]).all())
        thumb=max((c['normal_force_n'] for c in r['contacts'] if 'right_hand_thumb' in c['body']),default=0.)
        finger=max((c['normal_force_n'] for c in r['contacts'] if any(k in c['body'] for k in ('right_hand_index','right_hand_middle'))),default=0.)
        table=any(c['body']=='task_table' for c in r['contacts'])
        valid=r['lift_m']>=.1 and thumb>1e-4 and finger>1e-4 and not table
        force_rows_consistent &= bool(r['qualified']==valid and r['cube_table_contact']==table and
            abs(thumb-r['thumb_force_n'])<1e-9 and abs(finger-r['opposing_finger_force_n'])<1e-9)
        if valid:
            if current==0:begin=i
            current+=.005
        else:current=0.
        if current>maximum:maximum=current;best_begin=begin;best_end=i
        timer_rows_consistent &= abs(r['hold_seconds']-current)<1e-8
    physical=maximum>=2.-1e-9
    checks['raw_contacts_match_summary']=force_rows_consistent
    checks['continuous_timer_recomputed']=timer_rows_consistent and abs(maximum-result['max_continuous_hold_seconds'])<1e-8
    checks['physical_result_matches']=physical==result['physical_success']
    checks['finite_recordings']=physics_finite and all(np.isfinite(states[k]).all() for k in states.files)
    checks['complete_state_frame_coverage']=np.array_equal(states['reference_frame'],np.arange(count))
    checks['physics_step_coverage']=bool(active) and np.allclose(np.diff([r['sim_time'] for r in active]),.005,atol=1e-8,rtol=0)
    checks['reference_clock']=np.array_equal(reference['timestamps'],np.arange(count)/50)
    checks['hand_reference_agreement']=bool(np.allclose(states['hand_applied_ref_q'],reference['hand_ref_q'][states['reference_frame']],atol=1e-8,rtol=0))
    checks['no_support_contacts']=all(not r['support_contacts'] for r in rows)
    checks['no_fall']=all(not r['fall'] for r in rows)
    checks['runtime_integrity_passed']=result['integrity_pass']
    checks['playing_at_least16s']=len(active)*.005>=16.
    checks['exact810_reference']=count==810
    checks['cube_lift_recomputed']=all(abs(r['cube_world_pos'][2]-provenance['initial_qpos'][-5]-r['lift_m'])<1e-8 for r in active)
    checks['no_external_force_or_weld']=all(r['external_force_max']==0 and r['equality_count']==0 for r in rows)
    checks['finite_original_steps']=all(np.isfinite(np.r_[r['qpos'],r['qvel'],r['actuator_control'],r['body_actual_q'],r['hand_actual_q'],r['body_target_q'],r['hand_target_q']]).all() for r in rows)
    checks['original_step_hand_sync']=all(np.allclose(r['hand_target_q'],reference['hand_ref_q'][r['frame']],atol=1e-8,rtol=0) for r in active)
    checks['original_step_body_sync']=all(np.allclose(r['body_target_q'],reference['body_ref_q'][r['frame']],atol=1e-8,rtol=0) for r in active)
    checks['final_timer_matches']=abs(current-result['final_continuous_hold_seconds'])<1e-8
    checks['kimodo_actual_provenance']=provenance['source_generation'] is True and Path(provenance['raw_kimodo_source']).exists() and Path(provenance['raw_motion_npz']).exists()
    from export_kimodo_reference import verify_generation
    spec=json.loads((run/'source_setting.json').read_text())
    try:
        task_dir=Path(spec['task_path']);gen_dir=Path(spec['generation_path'])
        verify_generation(task_dir,gen_dir,json.loads((task_dir/'task.json').read_text()),json.loads((gen_dir/'generation.json').read_text()))
        checks['actual_generation_hash_linkage']=True
    except (ValueError,KeyError,OSError):checks['actual_generation_hash_linkage']=False
    checks['cursor_fresh']=result['max_sync_age_s'] is not None and result['max_sync_age_s']<=.04
    checks['no_missing_policy_ticks']=result['missing_policy_ticks']==0 and not result['skipped_frames']
    checks['all_source_hashes_unchanged']=all(sha(p)==h for p,h in provenance['source_hashes'].items())
    checks['export_source_hashes_unchanged'],checks['export_source_hashes_retained']=check_export_sources(spec['source_hashes'],provenance['source_hashes'])
    checks['all_runtime_hashes_unchanged']=all(sha(p)==h for p,h in provenance['audit_hashes'].items())
    if missing or not (run/'target_motion.csv').is_file():
        checks['complete_policy_logs']=False
        return dict(checks={k:bool(v) for k,v in checks.items()},physical_success_recomputed=bool(physical),expert_valid_confirmed=False,
                    max_continuous_hold_seconds_recomputed=maximum,final_continuous_hold_seconds_recomputed=current,
                    status='INCOMPLETE_POLICY_EVIDENCE',missing_policy_logs=missing)
    # Validate finite model outputs and the policy's own target stream separately.
    log_dir=run/'policy_logs'
    play=np.loadtxt(log_dir/'motion_playing.csv',delimiter=',',skiprows=1,ndmin=2)
    ix=np.flatnonzero(play[:,-1]>.5)
    checks['complete_policy_ticks']=len(ix)==count
    for name,width in [('action',29),('token_state',64)]:
        values=np.loadtxt(log_dir/(name+'.csv'),delimiter=',',skiprows=1,ndmin=2)
        checks['finite_'+name]=bool(len(ix) and len(values)>ix[-1] and
            values.shape[1]>=width and np.isfinite(values[ix]).all())
    mode=np.loadtxt(log_dir/'encoder_mode.csv',delimiter=',',skiprows=1,ndmin=2)
    checks['encoder_mode_zero']=bool(len(ix) and np.all(mode[ix,-1]==0))
    sonic=Path(provenance['binary']).parents[3]
    header=sonic/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp'
    scale=header.read_text().split('g1_action_scale = {',1)[1].split('};',1)[0]
    motors=re.findall(r'//\s*(\w+_joint)\b',scale)
    perm=[motors.index(n) for n in provenance['body_joint_order']]
    targets=np.loadtxt(run/'target_motion.csv',delimiter=',',usecols=range(36),ndmin=2)
    error=float(np.max(np.abs(targets[ix,7:][:,perm]-reference['body_ref_q']))) if len(ix)==count else None
    checks['policy_reference_matches']=error is not None and error<=1e-5
    for name,item in provenance['model_bundle'].items():
        checks['current_bundle_hash_'+name]=sha(item['path'])==item['sha256']
    checks['current_binary_hash']=sha(provenance['binary'])==provenance['binary_sha256']
    for name in ['scene','robot']:
        checks['current_'+name+'_hash']=sha(provenance[name])==provenance[name+'_sha256']
    shared=result['original_shared_evaluator_report']
    checks['shared_neutral_self_contact_check']=shared['robot_self_contact_steps']==0
    checks['shared_neutral_left_arm_contact_check']=shared['left_arm_environment_contact_steps']==0
    snapshots=run/'runtime_source'
    for key,hash_ in provenance['audit_hashes'].items():
        if Path(key).name in ('replay_validate_kimodo.py','grasp_metrics.py'):
            checks['snapshot_'+Path(key).name]=(snapshots/Path(key).name).is_file() and sha(snapshots/Path(key).name)==hash_
    best=active[best_begin:best_end+1] if best_begin is not None else []
    interval=None
    if best:
        interval=dict(start_time_from_playback_s=best[0]['sim_time']-active[0]['sim_time'],
            end_time_from_playback_s=best[-1]['sim_time']-active[0]['sim_time'],
            integrated_hold_seconds=maximum,minimum_lift_m=min(r['lift_m'] for r in best),
            minimum_thumb_force_n=min(r['thumb_force_n'] for r in best),
            minimum_opposing_finger_force_n=min(r['opposing_finger_force_n'] for r in best),
            cube_table_contact_steps=sum(r['cube_table_contact'] for r in best))
    return dict(checks={k:bool(v) for k,v in checks.items()},physical_success_recomputed=bool(physical),
        expert_valid_confirmed=bool(physical and result['expert_valid'] and all(checks.values())),
        policy_reference_max_error_rad=error,joint_mapping_header_sha256=sha(header),
        longest_qualifying_interval=interval,
        artifact_sha256={n:sha(run/n) for n in ['result.json','provenance.json','physics_steps.json','simulation.npz','input_reference.npz']},
        max_continuous_hold_seconds_recomputed=maximum,final_continuous_hold_seconds_recomputed=current,scope='User-specified physical/expert criteria; not shared Rule-source dataset certification')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path)
    args=p.parse_args();out=args.run_dir/'validation.json'
    if out.exists():p.error('validation.json already exists; preserve the existing audit')
    report=validate(args.run_dir)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['expert_valid_confirmed'] else 2


if __name__=='__main__':raise SystemExit(main())
