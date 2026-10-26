"""Execute an actual Kimodo candidate with canonical SONIC and original-step grasp evidence."""
import argparse
import hashlib
import gc
import json
import os
from pathlib import Path
import pty
import select
import signal
import shutil
import tempfile
import threading
import xml.etree.ElementTree as ET
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(os.environ.get('RESEARCH_ROOT', Path(__file__).resolve().parents[3])).resolve()
PROJECT = Path(os.environ.get('KIMODO_RUNTIME_SUPPORT', Path(__file__).resolve().parents[1]/'runtime_support')).resolve()
SONIC = Path(os.environ.get('SONIC_REPO', ROOT/'GR00T-WholeBodyControl')).resolve()
FROZEN_CONFIG = Path(os.environ.get('KIMODO_FROZEN_CONFIG',
    ROOT/'workspace/kunqi/task9_formal40_20261008/kimodo_frozen_method_source/config/frozen_config.json')).resolve()
ROBOT = SONIC/'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml'
HASHES = {
    'model_encoder.onnx': '013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3',
    'model_decoder.onnx': 'c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed',
    'observation_config.yaml': '466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b',
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def other_controllers(own_pid=None):
    # Ignore this wrapper, its own launched child, and ancestors only.
    excluded={os.getpid(),own_pid};pid=os.getpid()
    while pid>1:
        try:
            status=Path(f'/proc/{pid}/status').read_text()
            pid=int(next(r.split()[1] for r in status.splitlines() if r.startswith('PPid:')))
            excluded.add(pid)
        except (OSError,StopIteration,ValueError):break
    pattern='g1_deploy_onnx_ref|run_sim_loop.py|s2_trial.py|test_ardy_grasp.py|run_sonic_grasp.py|canonical_grasp.py|replay_validate_kimodo.py'
    p=subprocess.run(['pgrep','-af',pattern],capture_output=True,text=True)
    return [line for line in p.stdout.splitlines() if 'pgrep' not in line and int(line.split()[0]) not in excluded]


def dry_run(candidate,out):
    import numpy as np
    from export_kimodo_reference import canonical
    spec=json.loads((candidate/'setting.json').read_text())
    from export_kimodo_reference import verify_generation
    task_dir=Path(spec['task_path']);generation_dir=Path(spec['generation_path'])
    verify_generation(task_dir,generation_dir,json.loads((task_dir/'task.json').read_text()),json.loads((generation_dir/'generation.json').read_text()))
    model,data,bn,hn=canonical(spec)
    reference=np.load(candidate/'reference.npz',allow_pickle=False)
    if len(reference['timestamps'])!=810:raise RuntimeError('Expected810 frames')
    source=candidate/'reference/motion'
    for key,width in [('joint_pos',29),('joint_vel',29),('body_pos',42),('body_quat',56),('body_lin_vel',42),('body_ang_vel',42)]:
        a=np.loadtxt(source/(key+'.csv'),delimiter=',',skiprows=1,ndmin=2)
        if a.shape!=(810,width) or not np.isfinite(a).all():raise RuntimeError('Invalid SONIC CSV '+key)
    if spec['body_part_indexes'] != [0,4,10,18,5,11,19,9,16,22,28,17,23,29]:raise RuntimeError('Incorrect body part indexes')
    for path,hash_ in spec['source_hashes'].items():
        if digest(path)!=hash_:raise RuntimeError('Changed raw source '+path)
    out.mkdir(parents=True)
    result=dict(status='DRY_RUN_READY',physical_replay_run=False,physical_success=None,expert_valid=None,frames=810,
        scope='CPU scene/format/provenance checks only; no policy/controller started')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--gpu',type=int,help='Parent-allocated physical GPU ID; required for --run')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.out.exists():parser.error('Choose a fresh output directory')
    if args.dry_run:return dry_run(args.candidate.resolve(),args.out.resolve())
    if args.gpu is None or args.gpu<0:parser.error('--run requires --gpu parent-allocated ID')
    busy=other_controllers()
    if busy:parser.error('Shared DDS channel busy: '+str(busy))
    os.environ['MUSA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ['SONIC_REPO']=str(SONIC)
    sys.path[:0]=[str(PROJECT/'scripts'),str(SONIC)]
    # Explicit contract robot selection, superseding the old helper's Git-pin fallback.
    # This changes only the current process and is recorded below.
    import project_paths
    project_paths.default_robot_xml=lambda:ROBOT
    import mujoco
    import numpy as np
    from scene import load_scene
    from cases import reset_case,read_cases
    from episode_schema import neutral_arms,NEUTRAL_POSE
    from expert_trajectory import orders,JointMap,robot_state
    from reference_io import export_reference
    import run_sonic_grasp as api
    if mujoco.__version__!='3.14.0':raise RuntimeError('Unexpected MuJoCo version')
    deploy=SONIC/'gear_sonic_deploy'
    loaded={}
    for name,want in HASHES.items():
        p=deploy/'policy/release'/name
        got=digest(p)
        if got!=want:raise RuntimeError('Canonical hash mismatch: '+str(p))
        loaded[name]={'path':str(p),'sha256':got}
    args.out.mkdir(parents=True)
    out=args.out.resolve()
    model,data=load_scene(supported=False,robot_path=ROBOT)
    assert (model.nq,model.nv,model.nu,model.neq)==(57,55,43,0)
    assert model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert np.allclose(model.opt.gravity,[0,0,-9.81])
    candidate=args.candidate.resolve()
    spec=json.loads((candidate/'setting.json').read_text())
    from export_kimodo_reference import verify_generation
    task_dir=Path(spec['task_path']);generation_dir=Path(spec['generation_path'])
    verify_generation(task_dir,generation_dir,json.loads((task_dir/'task.json').read_text()),json.loads((generation_dir/'generation.json').read_text()))
    case=spec['case']
    known_cases={c['id']:c for c in read_cases()['cases']}
    if case['id'] not in known_cases or case!=known_cases[case['id']]:
        raise RuntimeError('Candidate case must match the shared grasp_cases.json')
    reset_case(model,data,case)
    for side in ('left','right'):
        for part in ('hip_pitch','knee','ankle_pitch'):
            data.qpos[model.joint(f'{side}_{part}_joint').qposadr[0]]=NEUTRAL_POSE[part]
    neutral_arms(model,data)
    mujoco.mj_forward(model,data)
    feet=[i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name in
          ('left_ankle_roll_link','right_ankle_roll_link') and model.geom_contype[i]]
    rqa=int(model.joint('floating_base_joint').qposadr[0])
    data.qpos[rqa+2]-=min(data.geom_xpos[i,2]-model.geom_size[i,0] for i in feet)+.0001
    mujoco.mj_forward(model,data)
    bn,hn,motors=orders()
    body,hand=JointMap(model,bn,29),JointMap(model,hn,14)
    from grasp_metrics import GraspEvidence
    if spec['body_joint_order']!=bn or spec['hand_joint_order']!=hn:
        raise RuntimeError('Source joint order differs from canonical order')
    source=candidate/'reference/motion'
    read=lambda name:np.loadtxt(source/(name+'.csv'),delimiter=',',skiprows=1,ndmin=2)
    q=read('joint_pos'); dq=read('joint_vel')
    root=read('body_pos')[:,:3]; quat=read('body_quat')[:,:4]
    hand_values={}
    for side in ('left','right'):
        f=source/f'{side}_hand_pos.csv'
        headers=f.read_text().splitlines()[0].split(',')
        values=np.loadtxt(f,delimiter=',',skiprows=1,ndmin=2)
        for name,column in zip(headers,values.T):
            canonical=side+'_hand_'+name[len(side)+1:]+'_joint'
            hand_values[canonical]=column
    hands=np.stack([hand_values[n] for n in hn],axis=1)
    count=len(q)
    recorded_reference=np.load(candidate/'reference.npz',allow_pickle=False)
    arrays={k:recorded_reference[k] for k in recorded_reference.files}
    if count!=810:raise RuntimeError('Expected complete810 frame reference')
    for key,actual in [('body_ref_q',q),('body_ref_dq',dq),('root_ref_pos',root),('root_ref_quat',quat),('hand_ref_q',hands)]:
        if not np.allclose(arrays[key],actual,atol=1e-8,rtol=0):raise RuntimeError('NPZ/CSV mismatch '+key)
    expected_shapes={'timestamps':(count,), 'body_ref_q':(count,29),'body_ref_dq':(count,29),
        'hand_ref_q':(count,14),'hand_ref_dq':(count,14),'root_ref_pos':(count,3),'root_ref_quat':(count,4),
        'root_lin_vel':(count,3),'root_ang_vel':(count,3)}
    for k,v in arrays.items():
        if v.shape!=expected_shapes[k] or not np.isfinite(v).all():raise RuntimeError('Invalid array '+k)
    if not np.allclose(q[0],data.qpos[body.qa],atol=1e-6,rtol=0):raise RuntimeError('Noncanonical body frame zero')
    if not np.allclose(hands[0],0,atol=1e-10):raise RuntimeError('Noncanonical hand frame zero')
    if not np.allclose(root[0],data.qpos[rqa:rqa+3],atol=1e-6,rtol=0):raise RuntimeError('Noncanonical root frame zero')
    if not np.allclose(quat[0],[1,0,0,0],atol=1e-6):raise RuntimeError('Noncanonical root quaternion')
    if not np.allclose(np.linalg.norm(quat,axis=1),1,atol=1e-5):raise RuntimeError('Root quaternion not normalized')
    metadata=dict(candidate_type='kimodo_'+spec['raw_vs_adapted'],case=case,
        initial_pose={'name':NEUTRAL_POSE['name']},body_joint_order=bn,hand_joint_order=hn)
    motion=out/'reference/motion'
    # Keep every SONIC body CSV byte-for-byte, including body-part poses and velocity fields.
    shutil.copytree(source,motion)
    np.savez_compressed(out/'input_reference.npz',**arrays)
    (out/'source_setting.json').write_text(json.dumps(spec,indent=2)+'\n')
    source_files=[f for f in source.iterdir() if f.is_file()]+[candidate/'setting.json',candidate/'reference.npz',candidate/'export_metadata.json']
    raw=Path(spec['raw_source'])
    source_files += [raw,Path(spec['raw_motion']),Path(spec['generation_path'])/'generation.json',Path(spec['task_path'])/'task.json']
    source_hashes={str(f):digest(f) for f in source_files if f.is_file()}
    if not raw.exists() or not Path(spec['raw_motion']).exists():raise RuntimeError('Missing original Kimodo output')
    for path,hash_ in spec['source_hashes'].items():
        if digest(path)!=hash_:raise RuntimeError('Input hash changed '+path)
    # Preserve calibration and prior-run evidence through end-of-run checks.
    source_hashes.update(spec['source_hashes'])
    xml=ET.parse(ROBOT).getroot(); compiler=xml.find('compiler')
    assets=[]
    for kind,attr in [('mesh','meshdir'),('texture','texturedir')]:
        for elem in xml.findall('asset/'+kind):
            if elem.get('file'):assets.append((ROBOT.parent/compiler.get(attr,'')/elem.get('file')).resolve())
    audited=[ROBOT,PROJECT/'scenes/tabletop.xml',*assets,
        *[PROJECT/'scripts'/n for n in ['scene.py','run_sonic_grasp.py','hand_trajectory_controller.py',
            'expert_trajectory.py','episode_schema.py','cases.py']],
        Path(__file__),Path(__file__).with_name('grasp_metrics.py'),
        Path(__file__).resolve().parents[1]/'export/export_kimodo_reference.py']
    audit_hashes={str(f):digest(f) for f in audited}
    (out/'runtime_source').mkdir()
    for f in audited:
        if f.suffix=='.py':shutil.copy2(f,out/'runtime_source'/f.name)
    provenance=dict(task='Actual Kimodo generated body + explicit offline export + canonical SONIC physical grasp',task_id=spec['task_id'],raw_vs_adapted=spec['raw_vs_adapted'],postprocessing=spec.get('postprocessing'),generation_metadata=json.loads((Path(spec['generation_path'])/'generation.json').read_text()),task_sha256=digest(Path(spec['task_path'])/'task.json'),case=case,
        scene=str(PROJECT/'scenes/tabletop.xml'),scene_sha256=digest(PROJECT/'scenes/tabletop.xml'),
        builder=str(PROJECT/'scripts/scene.py'),robot=str(ROBOT),robot_sha256=digest(ROBOT),
        model_bundle=loaded,binary=str(deploy/'target/release/g1_deploy_onnx_ref'),
        binary_sha256=digest(deploy/'target/release/g1_deploy_onnx_ref'),
        initial_qpos=data.qpos.tolist(),body_joint_order=bn,hand_joint_order=hn,
        physics_hz=200,reference_hz=50,hand_pd={'kp':12.,'kd':.3},musa_visible_devices=str(args.gpu),physical_gpu_id=args.gpu,visible_device_index=0,
        source_generation=True,source_candidate=str(candidate),source_description=spec['source'],
        prompt=spec['prompt'],seed=spec['seed'],source_hashes=source_hashes,audit_hashes=audit_hashes,
        pre_simulated_settle_seconds=0,raw_kimodo_source=str(raw),raw_motion_npz=spec['raw_motion'],
        rule_source_success_gate='not applicable to generative Kimodo reference; no source-success flag fabricated',
        frame_zero_precision='exported source CSV precision12g; canonical tolerance1e-6',
        startup_body='canonical freebase WBC while controller loads; SONIC only during playback',
        requested_standby_ticks=5,evaluation_version='user_physical_success_20261007_v1',
        policy_live_logging='node-local temporary directory, archived after controller shutdown')
    provenance.update(mode='task9_qualification',classification='predeclared_single_trial',ardy_grasp_information_used=False,historical_replay_information_used=False,shared_generic_text_encoder_authorized=True,frozen_config_path=str(FROZEN_CONFIG),frozen_config_sha256=digest(FROZEN_CONFIG))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    cfg=SimpleNamespace(robot='baseline',pelvis_support='none',physics_hz=200,startup_body='freebase-wbc',
        band_angular_damping=10.,band_anchor='source-equilibrium',band='off',startup_support='none')
    original=api.load_scene
    api.load_scene=lambda **kw:original(**dict(kw,robot_path=ROBOT))
    runtime=api.TabletopDDS(arrays,metadata,cfg)
    assert np.allclose(runtime.data.qpos,data.qpos,atol=1e-6,rtol=0)
    assert runtime.model.opt.timestep==.005
    assert runtime.model.neq==0
    evidence=GraspEvidence(runtime.model,float(data.xpos[model.body('task_red_cube').id,2]))
    provenance['initial_qpos']=runtime.data.qpos.tolist()
    master,slave=pty.openpty()
    # Keep per-tick controller IO off the shared mount; archive it after shutdown.
    live_temp=tempfile.TemporaryDirectory(prefix='kimodo_canonical_live_')
    live=Path(live_temp.name)
    command=[str(deploy/'target/release/g1_deploy_onnx_ref'),'lo','policy/release/model_decoder.onnx',str(motion.parent),
        '--obs-config','policy/release/observation_config.yaml','--encoder-file','policy/release/model_encoder.onnx',
        '--input-type','keyboard','--policy-precision','32','--disable-crc-check','--enable-csv-logs',
        '--logs-dir',str(live/'policy_logs'),'--target-motion-logfile',str(live/'target_motion.csv'),
        '--record-input-file',str(live/'playback_cursor.csv')]
    provenance['command']=command
    provenance.update(mode='task9_qualification',classification='predeclared_single_trial',ardy_grasp_information_used=False,historical_replay_information_used=False,shared_generic_text_encoder_authorized=True,frozen_config_path=str(FROZEN_CONFIG),frozen_config_sha256=digest(FROZEN_CONFIG))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    log=(live/'controller.log').open('wb')
    controller_launch_wall=time.monotonic();play_start_wall=None;play_end_wall=None
    child=subprocess.Popen(command,cwd=deploy,stdin=slave,stdout=slave,stderr=slave,start_new_session=True)
    os.close(slave)
    monitor_stop=threading.Event()
    monitor={'foreign':[], 'checked_at':time.monotonic(), 'error':None}
    def monitor_processes():
        while not monitor_stop.is_set():
            try:
                monitor['foreign']=other_controllers(child.pid)
                monitor['checked_at']=time.monotonic()
            except Exception as exc:monitor['error']=str(exc)
            monitor_stop.wait(.5)
    monitor_thread=threading.Thread(target=monitor_processes,daemon=True)
    monitor_thread.start()
    cursor=api.PlaybackCursor(live,count)
    captured=bytearray();frames=[];records=[];applied=[];ages=[];skips=[];cursor_offsets=[];hand_commands=[]
    completed=initialized=requested=False;failure=None;foreign=[]
    start=time.monotonic();next_step=start;next_check=start
    minimum_height=999.;max_tilt=0.;startup_samples=[];step_costs=[];gc_was_enabled=gc.isenabled()
    def pump():
        while select.select([master],[],[],0)[0]:
            try:chunk=os.read(master,65536)
            except OSError:break
            if not chunk:break
            captured.extend(chunk);log.write(chunk);log.flush()
    try:
        while time.monotonic()-start<150:
            now=time.monotonic()
            foreign=monitor['foreign']
            if foreign:raise RuntimeError('Concurrent DDS process: '+str(foreign))
            if monitor['error'] or now-monitor['checked_at']>2:
                raise RuntimeError('Concurrent-process monitor unavailable')
            pump()
            if child.poll() is not None:raise RuntimeError(f'Controller exited {child.returncode}; see controller.log')
            if not initialized and b'Init Done' in captured:
                initialized=True;os.write(master,b']');print('CONTROLLER_READY',flush=True)
            if initialized and not requested and runtime.bridge.low_cmd_received and cursor.standby_ticks>=5:
                os.write(master,b't');requested=True;print('PLAY_KIMODO_GRASP_REFERENCE',flush=True)
            was_playing=cursor.started
            new=cursor.update()
            if cursor.started and not was_playing:
                play_start_wall=time.monotonic()
                next_step=play_start_wall
                gc.disable()  # defer cyclic-GC pauses during the short measured replay
            if cursor.completed:
                completed=True;play_end_wall=time.monotonic();break
            frame=cursor.frame if cursor.started else 0
            before=runtime.data.time
            step_wall_start=time.monotonic()
            hq,hcmd,band,_=runtime.step(arrays['timestamps'][frame],runtime.bridge.low_cmd_received and initialized,cursor.started)
            assert runtime.data.time>before and not band
            assert not np.any(runtime.data.xfrc_applied) and not np.any(runtime.data.qfrc_applied)
            pelvis=runtime.model.body('pelvis').id
            height=float(runtime.data.xpos[pelvis,2])
            tilt=float(np.degrees(np.arccos(np.clip(runtime.data.xmat[pelvis].reshape(3,3)[2,2],-1,1))))
            minimum_height=min(minimum_height,height);max_tilt=max(max_tilt,tilt)
            contact=evidence.step(runtime.data,frame,cursor.started)
            contact['body_target_q']=arrays['body_ref_q'][frame].tolist()
            contact['hand_target_q']=hq.tolist()
            contact['body_actual_q']=runtime.data.qpos[runtime.body.qa].tolist()
            contact['hand_actual_q']=runtime.data.qpos[runtime.hand.qa].tolist()
            if cursor.started:step_costs.append(time.monotonic()-step_wall_start)
            if not cursor.started and (not startup_samples or runtime.data.time-startup_samples[-1][0]>=.1):
                startup_samples.append([float(runtime.data.time),height,tilt])
            if new:
                skips.extend(f for f,_ in new[:-1])
                if frame!=(frames[-1]+1 if frames else 0):skips.extend(range(frames[-1]+1 if frames else 0,frame))
                state=robot_state(runtime.model,runtime.data,runtime.body,runtime.hand)
                state['sim_time']=runtime.data.time
                records.append(state);frames.append(frame);applied.append(hq)
                ages.append(max(0,time.time()-cursor.tick_time))
                cursor_offsets.append(cursor.explicit_frame-frame);hand_commands.append(hcmd)
                if frame%50==0:print(f'FRAME {frame}/{count} root_z={height:.4f} lift={contact["lift_m"]:.4f} hold={contact["hold_seconds"]:.3f}',flush=True)
            if contact['fall']:raise RuntimeError('Robot fell; no reset or retry within episode')
            next_step+=.005
            time.sleep(max(0,next_step-time.monotonic()))
        if not completed:failure='Playback did not complete'
    except Exception as e:failure=f'{type(e).__name__}: {e}'
    finally:
        if gc_was_enabled:gc.enable()
        monitor_stop.set();monitor_thread.join(timeout=3)
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        pump();log.close();cursor.close();os.close(master)
        for path in live.iterdir():
            if path.is_dir():shutil.copytree(path,out/path.name)
            else:shutil.copy2(path,out/path.name)
        live_temp.cleanup()
    target_error=None;mode_ok=False
    try:
        play=np.loadtxt(out/'policy_logs/motion_playing.csv',delimiter=',',skiprows=1,ndmin=2)
        ix=np.flatnonzero(play[:,-1]>.5)
        target=np.loadtxt(out/'target_motion.csv',delimiter=',',usecols=range(36),ndmin=2)
        modes=np.loadtxt(out/'policy_logs/encoder_mode.csv',delimiter=',',skiprows=1,ndmin=2)
        if len(ix)==count:
            target_error=float(np.max(abs(target[ix,7:][:,[motors.index(n) for n in bn]]-arrays['body_ref_q'])))
            mode_ok=bool(np.all(modes[ix,-1]==0))
    except (OSError,ValueError,IndexError):pass
    physical_duration=records[-1]['sim_time']-records[0]['sim_time'] if len(records)>1 else 0.
    coverage=frames==list(range(count))
    hand_error=float(np.max(abs(np.array(applied)-arrays['hand_ref_q'][frames]))) if frames else None
    reasons=[]
    if failure:reasons.append(failure)
    if not completed or not coverage or skips or cursor.missing_ticks:reasons.append('frame_coverage_failed')
    if not cursor_offsets or max(abs(x) for x in cursor_offsets)>2:reasons.append('explicit_cursor_disagreement')
    if target_error is None or target_error>1e-5:reasons.append('policy_target_unverified')
    if not mode_ok:reasons.append('encoder_mode_unverified')
    if hand_error is None or hand_error>1e-8:reasons.append('hand_sync_unverified')
    if abs(physical_duration-arrays['timestamps'][-1])>.1:reasons.append('physics_reference_clock_mismatch')
    if len([r for r in evidence.rows if r['playing']])*.005<16.:reasons.append('playing_less_than16s')
    if not ages or max(ages)>.04:reasons.append('reference_cursor_stale')
    finite=all(np.isfinite(v).all() for row in records for v in row.values())
    if not finite:reasons.append('nonfinite_state')
    physics_report=evidence.report()
    if evidence.fall_steps:reasons.append('robot_fell')
    if evidence.support_steps:reasons.append('robot_environment_support_contact')
    changed=[f for f,h in audit_hashes.items() if digest(f)!=h]
    changed += [f for f,h in source_hashes.items() if digest(f)!=h]
    if changed:reasons.append('source_or_runtime_file_changed_during_run')
    active=[r for r in evidence.rows if r['playing']]
    physics_steps_ok=bool(active) and all(abs(b['sim_time']-a['sim_time']-.005)<1e-8 for a,b in zip(active,active[1:]))
    if not physics_steps_ok:reasons.append('missing_physics_step_evidence')
    if runtime.command_received_steps!=len(active):reasons.append('missing_sonic_body_command')
    complete_finite=finite and all(np.isfinite(r['lift_m']) and np.isfinite(r['thumb_force_n']) and np.isfinite(r['opposing_finger_force_n']) for r in evidence.rows)
    if not complete_finite:reasons.append('nonfinite_physics_evidence')
    expert_valid=physics_report['physical_success'] and not reasons
    result=dict(**physics_report,expert_valid=expert_valid,grasp_attempted=True,
        integrity_pass=not reasons,integrity_failure_reasons=reasons,
        failure_reasons=reasons+([] if physics_report['physical_success'] else ['physical_grasp_criterion_not_met']),
        completed=completed,recorded_frames=len(frames),expected_frames=count,
        minimum_pelvis_height_m=minimum_height,maximum_tilt_degrees=max_tilt,frame_coverage_complete=coverage,
        target_error_rad=target_error,hand_target_error_rad=hand_error,encoder_mode_0_verified=mode_ok,
        max_sync_age_s=max(ages) if ages else None,physical_duration_s=physical_duration,
        finite=complete_finite,foreign_processes=foreign,initialization_sim_seconds=records[0]['sim_time'] if records else None,
        explicit_cursor_max_offset_frames=max((abs(x) for x in cursor_offsets),default=None),
        missing_policy_ticks=cursor.missing_ticks,skipped_frames=skips,
        sonic_body_command_steps=runtime.command_received_steps,changed_files=changed,
        no_runtime_body_ik=True,no_weld=True,no_external_force=True,
        original_shared_rule_evaluator_certification=False,
        original_shared_evaluator_report=runtime.evaluator.report(runtime.data),
        body_tracking_rmse_rad=float(np.sqrt(np.mean((np.array([r['body_q'] for r in records])-q[frames])**2))) if frames else None)
    result['sonic_initialization_wall_seconds']=None if play_start_wall is None else play_start_wall-controller_launch_wall
    result['replay_wall_seconds']=None if play_start_wall is None else (play_end_wall or time.monotonic())-play_start_wall
    result['physics_and_evidence_max_wall_seconds']=max(step_costs,default=None)
    result['physics_and_evidence_p99_wall_seconds']=float(np.quantile(step_costs,.99)) if step_costs else None
    schema=dict(physics_steps='Every original200Hz physics step, startup and playing separated by playing boolean',fields=dict(sim_time='s',frame='50Hz cursor',cube_world_pos='m3',cube_world_quat='wxyz4',qpos='57',qvel='55',actuator_control='43',body_actual_q='29 named canonical',hand_actual_q='14 named canonical',body_target_q='29 named canonical',hand_target_q='14 named canonical',external_force_max='0 required',equality_count='0 required',contacts='raw contact force/body/position/normal/distance',hold_seconds='continuous original-step timer'),body_joint_order=bn,hand_joint_order=hn)
    (out/'recording_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    (out/'physics_steps.json').write_text(json.dumps(evidence.rows)+'\n')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'simulation.npz',reference_frame=np.asarray(frames,dtype=np.int64),hand_applied_ref_q=np.asarray(applied).reshape(-1,14),startup_samples=startup_samples,
        explicit_cursor_offset_frames=cursor_offsets,hand_actuator_command=hand_commands,
        **({k:np.asarray([row[k] for row in records]) for k in records[0]} if records else {}))
    print(json.dumps(result,indent=2),flush=True)
    print('OUTPUT_DIRECTORY',out,flush=True)
    return 0 if result['expert_valid'] else 2


if __name__=='__main__':sys.exit(main())
