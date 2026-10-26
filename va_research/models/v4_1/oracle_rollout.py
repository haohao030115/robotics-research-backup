"""V4 body with privileged geometry expert-hand oracle diagnostic rollout."""
import argparse
import copy
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import queue
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from .dataset import CanonicalV4, dump, read, sha
from .evaluation import load
from .inference import Inference, current_state
from .runtime import PlanBuffer


def vision_worker(blob, checkpoint, requests, results, ready, output):
    try:
        import mujoco
        torch.set_num_threads(1)
        model=pickle.loads(blob);data=mujoco.MjData(model)
        renderer=mujoco.Renderer(model,height=240,width=320)
        from OpenGL import GL
        proof = dict(pid=os.getpid(), parent_pid=os.getppid(), network_namespace=os.readlink('/proc/self/ns/net'),
            gl_vendor=GL.glGetString(GL.GL_VENDOR).decode(), gl_renderer=GL.glGetString(GL.GL_RENDERER).decode(),
            gl_version=GL.glGetString(GL.GL_VERSION).decode(), camera='head_rgb', resolution=[320,240],
            environment={k:os.environ.get(k) for k in ('MUJOCO_GL','PYOPENGL_PLATFORM','__EGL_VENDOR_LIBRARY_FILENAMES','LIBGL_DRIVERS_PATH','LD_LIBRARY_PATH')})
        proof['hardware_verified'] = 'Moore Threads' in proof['gl_vendor'] and 'MTT S4000' in proof['gl_renderer']
        dump(Path(output)/'renderer_worker.json',proof)
        if not proof['hardware_verified']:raise RuntimeError('Actual rendering worker is not Moore Threads / MTT S4000; refusing rollout')
        print('V4_RENDERER_VERIFIED '+json.dumps(proof),flush=True)
        network,_=load(checkpoint);policy=Inference(network)
        for _ in range(3):policy.predict(np.zeros((240,320,3),np.uint8),np.zeros(68),np.zeros(28))
        mujoco.mj_forward(model,data);renderer.update_scene(data,camera='head_rgb');renderer.render()
        ready.set()
        while True:
            r=requests.get()
            if r is None:break
            render_start=time.monotonic()
            data.qpos[:]=r.pop('qpos');data.qvel[:]=r.pop('qvel');mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera='head_rgb');rgb=renderer.render().copy()
            render_end=time.monotonic()
            p,seconds=policy.predict(rgb,r.pop('body'),r.pop('hand'))
            r.update(prediction=p,finished=time.monotonic(),inference_seconds=seconds,render_start=render_start,render_end=render_end,render_seconds=render_end-render_start)
            dump(Path(output)/('worker_plan_%06d.json'%r['start_tick']),{k:v for k,v in r.items() if k!='prediction'})
            results.put(r)
        renderer.close()
    except Exception:results.put(dict(error=traceback.format_exc()));ready.set()


def run(checkpoint, output):
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from execution.sonic_mujoco.common import SONIC, PROJECT, ROBOT, BUNDLE, HASHES
    from execution.sonic_mujoco.native_scene import SnapshotReader
    from execution.sonic_mujoco.native_replay import configure_dds
    from va_research.sonic.runtime import NativeTokenSonic, TimestampedHand
    from va_research.sonic.release import ExternalTokenAdapter
    import mujoco
    import zmq
    sys.path[:0]=[str(SONIC),str(PROJECT/'scripts')]
    import run_sonic_grasp as api
    torch.set_num_threads(1)
    _,saved=load(checkpoint);ds=CanonicalV4(saved['manifest'])
    metrics=read(Path(checkpoint).parent/'offline_metrics.json')
    if not metrics['offline_gate']:raise ValueError('Offline overfit gate failed')
    if ds.provenance != saved['provenance']:raise ValueError('Canonical inputs changed')
    out=Path(output);out.mkdir(exist_ok=False)
    lock=(root/'outputs/canonical_va/native_dds.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,h in HASHES.items():
        if sha(BUNDLE/name)!=h:raise ValueError('Frozen SONIC changed: '+name)
    native_meta=ds.runtime['runtime']
    binary=SONIC/'gear_sonic_deploy/target/release/g1_deploy_onnx_ref'
    if sha(binary)!=native_meta['original_build']['original_binary_sha256']:
        raise ValueError('Normal frozen native binary changed')
    os.environ.pop('CANONICAL_TRACE_DIR',None)
    configure_dds(out)
    with np.load(ds.root/'source_reference/reference.npz') as z:
        initial={k:z[k][:2].copy() for k in ('body_ref_q','body_ref_dq','root_ref_pos','root_ref_quat','hand_ref_q')}
        length=len(z['time_s'])
    initial.update(timestamps=np.array([0.,.02]),hand_ref_dq=np.zeros((2,14)))
    # Freeze initial hand references for startup only; learned targets replace them before active ticks.
    initial['hand_ref_q'][:]=initial['hand_ref_q'][0]
    original=api.load_scene
    api.load_scene=lambda **kw:original(**dict(kw,robot_path=ROBOT,scene_path=PROJECT/'scenes/tabletop.xml'))
    args=SimpleNamespace(robot='baseline',pelvis_support='none',physics_hz=200,startup_body='freebase-wbc',
        band_angular_damping=10.,band_anchor='source-equilibrium',band='off',startup_support='none')
    try:runtime=api.TabletopDDS(initial,dict(ds.source,case=native_meta['case'],initial_pose={'name':'neutral_standing_v1'}),args)
    finally:api.load_scene=original
    reader=SnapshotReader(runtime.model,runtime.data)
    # Privileged diagnostic only: expert closure trajectory and geometry trigger.
    expert_refs=ds.refs["hand_ref_q"]
    onset=int(np.flatnonzero(ds.phase_all>=1)[0]) if np.any(ds.phase_all>=1) else len(expert_refs)
    closed=int(np.argmax(ds.progress_all))
    open_ref=np.asarray(expert_refs[0]).copy(); close_ref=np.asarray(expert_refs[min(closed,len(expert_refs)-1)]).copy()
    hold_ref=np.asarray(expert_refs[-1]).copy()
    # Thresholds are derived from the successful canonical expert at its closure onset.
    try:
        with np.load(ds.root/"canonical_replay/post_states.npz",allow_pickle=False) as ez:
            ep=np.asarray(ez["right_wrist_pose"]); cp=np.asarray(ez["cube_pose"])
        n=min(onset,len(ep),len(cp)); expert_dist=float(np.linalg.norm(ep[n-1,:3]-cp[n-1,:3]))
        readiness_distance=expert_dist*1.20
        readiness_orientation=.35
    except Exception:
        readiness_distance=.12; readiness_orientation=.50
    trigger_log=[]; oracle_triggered=False; oracle_hold=False
    def oracle_hand_target(frame):
        nonlocal oracle_triggered, oracle_hold
        wrist=runtime.data.body("right_wrist_yaw_link").xpos.copy()
        cube=runtime.data.body("task_cube").xpos.copy()
        dist=float(np.linalg.norm(wrist-cube))
        # Relative orientation is diagnostic geometry; body orientation is available in xquat.
        wq=runtime.data.body("right_wrist_yaw_link").xquat; cq=runtime.data.body("task_cube").xquat
        orient=float(2*np.arccos(np.clip(abs(float(np.dot(wq,cq))),0.,1.)))
        ready=(dist<=readiness_distance and orient<=readiness_orientation and frame>=max(0,onset-10))
        if not oracle_triggered and ready:
            oracle_triggered=True; trigger_log.append(dict(tick=int(frame),distance_m=dist,orientation_rad=orient,threshold_distance_m=readiness_distance,threshold_orientation_rad=readiness_orientation,event="trigger"))
        if oracle_triggered:
            if frame<closed: target=expert_refs[min(frame,len(expert_refs)-1)]
            else: target=hold_ref; oracle_hold=True
            return np.asarray(target).copy()
        trigger_log.append(dict(tick=int(frame),distance_m=dist,orientation_rad=orient,threshold_distance_m=readiness_distance,threshold_orientation_rad=readiness_orientation,event="not_ready"))
        return open_ref.copy()
    if not np.array_equal(runtime.data.qpos,np.array(native_meta['initial_qpos'])) or not np.array_equal(runtime.data.qvel,np.array(native_meta['initial_qvel'])):
        raise ValueError('Canonical reset differs')
    boot=out/'bootstrap/initial_pose';boot.mkdir(parents=True)
    for name,value in dict(joint_pos=initial['body_ref_q'][0],joint_vel=np.zeros(29),body_pos=initial['root_ref_pos'][0],
                           body_quat=initial['root_ref_quat'][0],body_lin_vel=np.zeros(3),body_ang_vel=np.zeros(3)).items():
        np.savetxt(boot/(name+'.csv'),np.tile(value,(length+1,1)),delimiter=',',header=','.join(f'{name}_{i}' for i in range(len(value))),comments='')
    (boot/'metadata.txt').write_text(f'Metadata for: initial_pose\nBody part indexes:\n[0]\n\nTotal timesteps: {length+1}\n')
    (boot/'info.txt').write_text('V4 startup pose only\nsample_rate_hz: 50\n')
    c=dict(sonic=dict(binary=str(binary),repository=str(SONIC),decoder=str(BUNDLE/'model_decoder.onnx'),observation_config=str(BUNDLE/'observation_config.yaml')),
           rollout=dict(network_interface='lo',zmq_port=5594,zmq_output_port=5595))
    # This is the authored mapping relation, not a searched/re-estimated offset.
    leads={r['future_execution_indices'][0]-r['anchor_execution_index'] for r in ds.mapping}
    if len(leads)!=1:raise ValueError('Nonuniform authored canonical mapping')
    lead=leads.pop()
    dump(out/'provenance.json',dict(checkpoint_sha256=sha(checkpoint),canonical=ds.provenance,
        canonical_physics=native_meta, native_runtime=dict(binary=str(binary),sha256=sha(binary),instrumented=False,encoder_enabled=False),sonic_hashes=HASHES,authored_target_lead_ticks=lead,
        startup_observation_anchor=-lead,hand_pd_unchanged=True,expert_future_injected=False,
        policy_inputs='Current RGB + 68D body + 28D hand',strict=True))
    hand=TimestampedHand(runtime.hand_controller);runtime.hand_controller=hand
    context=mp.get_context('spawn');requests=context.Queue();results=context.Queue();ready=context.Event()
    worker=context.Process(target=vision_worker,args=(pickle.dumps(runtime.model),str(checkpoint),requests,results,ready,str(out)));worker.start()
    def submit(anchor):
        body,h=current_state(runtime)
        requests.put(dict(anchor=anchor,start_tick=anchor+lead,requested=time.monotonic(),simulation_time=float(runtime.data.time),
            qpos=runtime.data.qpos.copy(),qvel=runtime.data.qvel.copy(),body=body,hand=h))
    scheduler=PlanBuffer();predictions=[];records=[];timeline=[];native=None;pub=None;zmqctx=None
    tokenlog=actionlog=None;failure=None;frame=-1;pending=None;first=None;last_request=None
    def accept(r,available):
        if 'error' in r:raise RuntimeError(r['error'])
        r['received_monotonic']=time.monotonic()
        predictions.append(r);p=r['prediction']
        scheduler.add(r['start_tick'],p['body_tokens'],p['hand_ref'],available)
        r['activated_monotonic']=time.monotonic()
        r['available_tick']=available
    try:
        if not ready.wait(60):raise RuntimeError('Vision initialization timeout')
        if not results.empty():accept(results.get(),0)
        proof=read(out/'renderer_worker.json')
        if not proof['hardware_verified']:raise RuntimeError('GPU renderer gate failed')
        zmqctx=zmq.Context();pub=zmqctx.socket(zmq.PUB);pub.setsockopt(zmq.LINGER,0);pub.bind('tcp://*:5594')
        adapter=ExternalTokenAdapter(str(SONIC));native=NativeTokenSonic(c,boot.parent,out)
        dump(out/'native_launch.json',dict(command=native.command,pid=native.process.pid,instrumented=False,encoder_enabled=False,renderer_verified_before_launch=proof))
        tokenlog=api.CSVFollower(out/'policy_logs/token_state.csv');actionlog=api.CSVFollower(out/'policy_logs/action.csv')
        initialized=enabled=started=False;preload=None;next_step=time.monotonic();end=next_step+180
        token_rows=[];action_rows=[]
        while time.monotonic()<end:
            native.pump()
            if native.process.poll() is not None:raise RuntimeError(f'Native SONIC exited with code {native.process.returncode}')
            if not worker.is_alive():raise RuntimeError('Vision worker exited')
            if not initialized and b'Init Done' in native.captured:
                native.key('\n');initialized=True
            if initialized and not enabled and b'ZMQ STREAMING MODE: ENABLED' in native.captured:
                enabled=True;submit(-lead);accept(results.get(timeout=60),0)
                pending=scheduler.take(0,0);first=pending;preload=time.monotonic()
            if enabled and not started:
                send_start=time.monotonic();pub.send(adapter.message(first['token'],0))
                first.update(send_start=send_start,send_end=time.monotonic())
                if time.monotonic()-preload>.2 and b'Protocol v4: Received 64D token' in native.captured:
                    native.key(']');started=True;next_step=time.monotonic()
            while not results.empty():accept(results.get_nowait(),frame+1)
            if started and pending is None and frame+1<length:
                expected=((frame+1)//5)*5
                pending=scheduler.take(frame+1,expected)
                send_start=time.monotonic();pub.send(adapter.message(pending['token'],frame+1))
                pending.update(send_start=send_start,send_end=time.monotonic())
            token_rows.extend(tokenlog.read());action_rows.extend(actionlog.read())
            if len(token_rows)>1 or len(action_rows)>1:raise RuntimeError('Missed native control snapshot')
            new=False
            if token_rows and action_rows:
                tr,ar=token_rows.pop(0),action_rows.pop(0)
                actual=int(tr[0])
                if tr[:5]!=ar[:5] or actual!=frame+1:raise RuntimeError('Noncontiguous native control records')
                if pending is None or pending['intended_tick']!=actual or not np.allclose(tr[5:],pending['token'],rtol=0,atol=1e-6):
                    scheduler.counts['deadline_misses']+=1
                    timeline.append(dict(actual_tick=actual,intended_tick=None if pending is None else pending['intended_tick'],actual_consumed_token=tr[5:],deadline_miss=True))
                    raise RuntimeError(f'Native token consumption mismatch at tick {actual}; strict abort')
                frame=actual; oracle_target=oracle_hand_target(frame); hand.set(frame,oracle_target); entry=pending; entry["oracle_hand_target"]=oracle_target; entry["oracle_state"]="HOLD" if oracle_hold else ("CLOSING" if oracle_triggered else "OPEN"); pending=None; new=True
                timeline.append(dict(actual_tick=frame,**{k:v for k,v in entry.items() if k not in ('token','hand_target')},
                    actual_consumed_token=tr[5:],hand_target_applied=hand.target.tolist(),native_time_ms=tr[1],native_consumed_monotonic=tr[3]/1000,observed_monotonic=time.monotonic()))
            now=time.monotonic()
            if now<next_step and not new:
                time.sleep(min(.001,next_step-now));continue
            applied,_,band,physical=runtime.step(max(frame,0)/50,initialized and runtime.bridge.low_cmd_received,frame>=0)
            if band or np.any(runtime.data.xfrc_applied) or np.any(runtime.data.qfrc_applied):raise RuntimeError('Unexpected support/external force')
            if new:
                if not np.array_equal(applied,hand.target):raise RuntimeError('Hand target application mismatch')
                snapshot=reader.snapshot(f'v4:{frame}',frame)
                snapshot.update(frame=frame,contacts_evaluator=copy.deepcopy(runtime.evaluator.last_contacts),physical=physical)
                records.append(snapshot)
                if (frame+lead)%5==0 and frame+lead<length:
                    submit(frame);last_request=frame
                if frame%50==0:print(f'V4 tick={frame} lift={physical["lift_m"]:.4f}',flush=True)
                if frame==length-1:break
            if runtime.evaluator.fall_steps:raise RuntimeError('Robot fell')
            next_step+=.005
        if frame!=length-1:failure='Incomplete rollout'
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        (out/'failure_traceback.txt').write_text(traceback.format_exc())
    finally:
        if native is not None:native.close()
        requests.put(None);worker.join(timeout=5)
        if worker.is_alive():worker.terminate();worker.join()
        for log in (tokenlog,actionlog):
            if log is not None:log.close()
        if pub is not None:pub.close()
        if zmqctx is not None:zmqctx.term()
    report=runtime.evaluator.report(runtime.data)
    report.update(failure=failure,rollout_complete=frame==length-1 and failure is None,no_fall=runtime.evaluator.fall_steps==0,
        recorded_ticks=len(records),expected_ticks=length,scheduler=scheduler.counts,
        strict_runtime_pass=failure is None and frame==length-1 and scheduler.counts['uncovered_ticks']==0)
    report['oracle']=dict(triggered=oracle_triggered,readiness_not_reached=not oracle_triggered,onset_expert_tick=onset,closed_expert_tick=closed,readiness_distance_m=readiness_distance,readiness_orientation_rad=readiness_orientation); dump(out/'report.json',report);dump(out/'control_timeline.json',timeline);dump(out/'oracle_trigger_timeline.json',trigger_log)
    dump(out/'scheduler_events.json',[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()} for r in scheduler.log])
    if records:
        np.savez_compressed(out/'physics.npz',**{k:np.asarray([r[k] for r in records]) for k in records[0] if k not in ('physical','contacts_evaluator')})
        dump(out/'physical_metrics.json',[r['physical'] for r in records]);dump(out/'contacts.json',[r['contacts_evaluator'] for r in records])
    if predictions:
        np.savez_compressed(out/'predictions.npz',**{k:np.asarray([r['prediction'][k] for r in predictions]) for k in predictions[0]['prediction']},
            start_tick=[r['start_tick'] for r in predictions],anchor=[r['anchor'] for r in predictions])
        dump(out/'inference_timing.json',[{k:v for k,v in r.items() if k!='prediction'} for r in predictions])
    print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    report=run(a.checkpoint,a.output)
    if not report['strict_runtime_pass']:raise SystemExit(2)


if __name__=='__main__':main()
