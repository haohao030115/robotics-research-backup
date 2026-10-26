"""Execute a named ARDY candidate with canonical SONIC and original-step grasp evidence."""
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
PROJECT = Path(os.environ.get('DL_GROUP2_ROOT', ROOT / 'dl-group2')).resolve()
SONIC = Path(os.environ.get('SONIC_REPO', ROOT / 'GR00T-WholeBodyControl')).resolve()
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
    # DDS is isolated by a fresh network namespace, not by timing with other jobs.
    current=os.readlink('/proc/self/ns/net')
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in (os.getpid(),own_pid): continue
        try:
            if os.readlink(proc/'ns/net')!=current: continue
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if any(x in cmd for x in ('g1_deploy_onnx_ref','sonic_ardy_isolated','run_sim_loop.py','replay_ardy_isolated.py')):
                found.append(proc.name+' '+cmd)
        except (OSError,PermissionError): pass
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Choose a fresh output directory')
    busy=other_controllers()
    if busy:parser.error('Shared DDS channel busy: '+str(busy))
    gpu=os.environ.get('ARDY_PHYSICAL_GPU','7')
    if gpu not in ('4','5','6','7'):parser.error('Use physical GPU 4--7')
    os.environ['MUSA_VISIBLE_DEVICES']=gpu
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    if os.readlink('/proc/self/ns/net')==os.readlink('/proc/1/ns/net'):
        parser.error('Run via replay_isolated.sh to protect other DDS users')
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
    from dds_runtime_logging import configure_dds_logging
    dds_logs=configure_dds_logging(out/'dds_logs')
    model,data=load_scene(supported=False,robot_path=ROBOT)
    assert (model.nq,model.nv,model.nu,model.neq)==(57,55,43,0)
    assert model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert np.allclose(model.opt.gravity,[0,0,-9.81])
    candidate=args.candidate.resolve()
    spec=json.loads((candidate/'setting.json').read_text())
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
    arrays=dict(timestamps=np.arange(count)/50,body_ref_q=q,body_ref_dq=dq,
        hand_ref_q=hands,hand_ref_dq=np.gradient(hands,.02,axis=0),
        root_ref_pos=root,root_ref_quat=quat)
    expected_shapes={'timestamps':(count,), 'body_ref_q':(count,29),'body_ref_dq':(count,29),
        'hand_ref_q':(count,14),'hand_ref_dq':(count,14),'root_ref_pos':(count,3),'root_ref_quat':(count,4)}
    for k,v in arrays.items():
        if v.shape!=expected_shapes[k] or not np.isfinite(v).all():raise RuntimeError('Invalid array '+k)
    if not np.allclose(q[0],data.qpos[body.qa],atol=1e-6,rtol=0):raise RuntimeError('Noncanonical body frame zero')
    if not np.allclose(hands[0],0,atol=1e-10):raise RuntimeError('Noncanonical hand frame zero')
    if not np.allclose(root[0],data.qpos[rqa:rqa+3],atol=1e-6,rtol=0):raise RuntimeError('Noncanonical root frame zero')
    if not np.allclose(quat[0],[1,0,0,0],atol=1e-6):raise RuntimeError('Noncanonical root quaternion')
    if not np.allclose(np.linalg.norm(quat,axis=1),1,atol=1e-5):raise RuntimeError('Root quaternion not normalized')
    metadata=dict(candidate_type='ardy_world_wrist_ik_postprocessing',case=case,
        initial_pose={'name':NEUTRAL_POSE['name']},body_joint_order=bn,hand_joint_order=hn)
    motion=out/'reference/motion'
    # Keep every SONIC body CSV byte-for-byte, including body-part poses and velocity fields.
    shutil.copytree(source,motion)
    np.savez_compressed(out/'input_reference.npz',**arrays)
    (out/'source_setting.json').write_text(json.dumps(spec,indent=2)+'\n')
    source_files=[f for f in source.iterdir() if f.is_file()]+[candidate/'setting.json',candidate/'constraints.json']
    raw=Path(spec['raw_source'])
    source_files += [raw,raw.with_suffix('.npz')]
    source_hashes={str(f):digest(f) for f in source_files if f.is_file()}
    if not raw.exists() or not raw.with_suffix('.npz').exists():raise RuntimeError('Missing original ARDY output')
    xml=ET.parse(ROBOT).getroot(); compiler=xml.find('compiler')
    assets=[]
    for kind,attr in [('mesh','meshdir'),('texture','texturedir')]:
        for elem in xml.findall('asset/'+kind):
            if elem.get('file'):assets.append((ROBOT.parent/compiler.get(attr,'')/elem.get('file')).resolve())
    audited=[ROBOT,PROJECT/'scenes/tabletop.xml',*assets,
        *[PROJECT/'scripts'/n for n in ['scene.py','run_sonic_grasp.py','hand_trajectory_controller.py',
            'expert_trajectory.py','episode_schema.py','cases.py']],
        Path(__file__),Path(__file__).with_name('grasp_metrics.py')]
    audit_hashes={str(f):digest(f) for f in audited}
    (out/'runtime_source').mkdir()
    for f in audited:
        if f.suffix=='.py':shutil.copy2(f,out/'runtime_source'/f.name)
    provenance=dict(task='ARDY generated body + world-wrist IK postprocessing + canonical SONIC physical grasp',case=case,
        scene=str(PROJECT/'scenes/tabletop.xml'),scene_sha256=digest(PROJECT/'scenes/tabletop.xml'),
        builder=str(PROJECT/'scripts/scene.py'),robot=str(ROBOT),robot_sha256=digest(ROBOT),
        model_bundle=loaded,binary=str(deploy/'target/release/g1_deploy_onnx_ref'),
        binary_sha256=digest(deploy/'target/release/g1_deploy_onnx_ref'),
        initial_qpos=data.qpos.tolist(),body_joint_order=bn,hand_joint_order=hn,
        physics_hz=200,reference_hz=50,hand_pd={'kp':12.,'kd':.3},musa_visible_devices=gpu,network_namespace=os.readlink('/proc/self/ns/net'),dds_logging=dds_logs,
        source_generation=False,source_candidate=str(candidate),source_description=spec['source'],
        prompt=spec['prompt'],seed=spec['seed'],source_hashes=source_hashes,audit_hashes=audit_hashes,
        pre_simulated_settle_seconds=0,raw_ardy_source=str(raw),
        rule_source_success_gate='not applicable to generative ARDY reference; no source-success flag fabricated',
        frame_zero_precision='exported source CSV precision12g; canonical tolerance1e-6',
        startup_body='canonical freebase WBC while controller loads; SONIC only during playback',
        requested_standby_ticks=5,evaluation_version='user_physical_success_20261007_v1',
        policy_live_logging='node-local temporary directory, archived after controller shutdown')
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
    live_temp=tempfile.TemporaryDirectory(prefix='ardy_canonical_live_')
    live=Path(live_temp.name)
    command=[str(Path(__file__).resolve().parents[1]/'bin/sonic_ardy_isolated'),'lo','policy/release/model_decoder.onnx',str(motion.parent),
        '--obs-config','policy/release/observation_config.yaml','--encoder-file','policy/release/model_encoder.onnx',
        '--input-type','keyboard','--policy-precision','32','--disable-crc-check','--enable-csv-logs',
        '--logs-dir',str(live/'policy_logs'),'--target-motion-logfile',str(live/'target_motion.csv'),
        '--record-input-file',str(live/'playback_cursor.csv')]
    provenance['command']=command
    provenance['isolated_controller_alias']=command[0]
    provenance['isolated_controller_sha256']=digest(command[0])
    assert provenance['isolated_controller_sha256']==provenance['binary_sha256']
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    log=(live/'controller.log').open('wb')
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
                os.write(master,b't');requested=True;print('PLAY_ARDY_GRASP_REFERENCE',flush=True)
            was_playing=cursor.started
            new=cursor.update()
            if cursor.started and not was_playing:
                next_step=time.monotonic()
                gc.disable()  # defer cyclic-GC pauses during the short measured replay
            if cursor.completed:completed=True;break
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
    result['physics_and_evidence_max_wall_seconds']=max(step_costs,default=None)
    result['physics_and_evidence_p99_wall_seconds']=float(np.quantile(step_costs,.99)) if step_costs else None
    (out/'physics_steps.json').write_text(json.dumps(evidence.rows)+'\n')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'simulation.npz',reference_frame=frames,hand_applied_ref_q=applied,startup_samples=startup_samples,
        explicit_cursor_offset_frames=cursor_offsets,hand_actuator_command=hand_commands,
        **({k:np.asarray([row[k] for row in records]) for k in records[0]} if records else {}))
    print(json.dumps(result,indent=2),flush=True)
    print('OUTPUT_DIRECTORY',out,flush=True)
    return 0 if result['completed'] and result['finite'] else 2

if __name__=='__main__':sys.exit(main())
