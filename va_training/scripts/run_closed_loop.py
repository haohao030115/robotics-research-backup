#!/usr/bin/env python3
"""Fresh physics via unchanged native SONIC ZMQ streaming and existing Hand PD.
VA observes live snapshots; no recorded state sequence is executed.
"""
import argparse,json,os,pty,queue,re,select,shlex,signal,subprocess,threading,time,traceback
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import mujoco
import torch,torch_musa
import zmq
ROOT=Path('/workspace/group2');sys.path.insert(0,str(ROOT/'va_training'));sys.path.insert(0,str(ROOT/'dl-group2/scripts'))
from models.inference import ModelOnlyPolicy
from run_sonic_grasp import TabletopDDS,CSVFollower
from expert_trajectory import load_reference,orders,robot_state,write_json,save_arrays
from hand_trajectory_controller import HandTrajectoryController
from reference_buffer import FutureReferenceBuffer

class NativeSonic:
    def __init__(self,reference,out):
        deploy=ROOT/'GR00T-WholeBodyControl/gear_sonic_deploy'
        self.master,slave=pty.openpty();self.log=(out/'sonic.log').open('wb');self.captured=bytearray();self.partial=b'';self.merge_acks=[]
        self.command=[str(deploy/'target/release/g1_deploy_onnx_ref'),'lo','policy/release/model_decoder.onnx',str(reference),
          '--obs-config','policy/release/observation_config.yaml','--encoder-file','policy/release/model_encoder.onnx',
          '--input-type','zmq','--zmq-host','localhost','--zmq-port','5586','--zmq-topic','pose',
          '--output-type','zmq','--zmq-out-port','5587','--disable-crc-check','--logs-dir',str(out/'policy_logs'),
          '--enable-csv-logs','--target-motion-logfile',str(out/'target_motion.csv'),'--record-input-file',str(out/'playback_cursor.csv')]
        self.process=subprocess.Popen(self.command,cwd=deploy,stdin=slave,stdout=slave,stderr=slave,start_new_session=True);os.close(slave)
    def pump(self):
        while select.select([self.master],[],[],0)[0]:
            try:data=os.read(self.master,65536)
            except OSError:break
            if not data:break
            self.captured.extend(data);self.log.write(data);self.log.flush()
            lines=(self.partial+data).split(b'\n');self.partial=lines.pop()
            for line in lines:
                match=re.search(rb'Merged streamed data: .*window \[(\d+)\.\.(\d+)\]',line)
                if match:self.merge_acks.append((int(match[2]),time.monotonic()))
    def key(self,k):os.write(self.master,k.encode())
    def close(self):
        self.pump()
        if self.process.poll() is None:
            self.key('O')
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGTERM);self.process.wait(timeout=3)
        self.pump();self.log.close();os.close(self.master)

class VisionWorker:
    def __init__(self,model,policy):
        self.model,self.policy=model,policy;self.requests=queue.Queue(maxsize=1);self.results=queue.Queue();self.stop=threading.Event();self.ready=threading.Event();self.error=None
        self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
    def submit(self,frame,data,body,hand):
        request=(frame,data.qpos.copy(),data.qvel.copy(),body.copy(),hand.copy(),time.monotonic())
        try:self.requests.put_nowait(request)
        except queue.Full:
            try:self.requests.get_nowait()
            except queue.Empty:pass
            self.requests.put_nowait(request)
    def work(self):
        renderer=None
        try:
            data=mujoco.MjData(self.model);renderer=mujoco.Renderer(self.model,height=240,width=320);self.ready.set()
            while not self.stop.is_set():
                try:frame,qpos,qvel,body,hand,requested=self.requests.get(timeout=.1)
                except queue.Empty:continue
                start=time.monotonic();data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(self.model,data)
                renderer.update_scene(data,camera='head_rgb');rgb=renderer.render().copy();render_end=time.monotonic()
                with torch.inference_mode():
                    image,bq,hq,single=self.policy.prepare(rgb,body,hand)
                    torch.musa.synchronize();prepare_end=time.monotonic()
                    output=self.policy.policy.predict(image,bq,hq)
                    torch.musa.synchronize();forward_end=time.monotonic()
                    prediction=output[0].cpu().numpy().copy()
                    if prediction.shape!=(50,43) or not np.isfinite(prediction).all():
                        raise RuntimeError('Invalid VA prediction')
                finish=time.monotonic()
                self.results.put({'anchor':frame,'prediction':prediction,'body_q':body,'hand_q':hand,'rgb':rgb,
                    'requested':requested,'finished':finish,'render_seconds':render_end-start,
                    'preprocessing_seconds':prepare_end-render_end,'forward_seconds':forward_end-prepare_end,
                    'postprocessing_seconds':finish-forward_end,'total_seconds':finish-requested})
        except Exception:self.error=traceback.format_exc();self.ready.set()
        finally:
            if renderer is not None:renderer.close()
    def close(self):self.stop.set();self.thread.join(timeout=30)

def packet(anchor,body,root_quat):
    arrays={'joint_pos':np.asarray(body,dtype='<f4'),'joint_vel':np.asarray(np.gradient(body,.02,axis=0),dtype='<f4'),
        'body_quat':np.tile(np.asarray(root_quat,dtype='<f8'),(len(body),1,1)),
        'frame_index':np.arange(anchor,anchor+len(body),dtype='<i8'),'catch_up':np.array([False],dtype=np.bool_)}
    dtype={'f':'f','i':'i','b':'bool'}
    fields=[{'name':k,'dtype':('bool' if v.dtype.kind=='b' else dtype[v.dtype.kind]+str(v.dtype.itemsize*8)),'shape':list(v.shape)} for k,v in arrays.items()]
    header=json.dumps({'v':1,'endian':'le','count':len(body),'fields':fields},separators=(',',':')).encode();assert len(header)<=1280
    return b'pose'+header.ljust(1280,b'\0')+b''.join(v.tobytes() for v in arrays.values())

class BufferedHand:
    """Keep the existing Hand PD; choose its target using only SONIC's cursor."""
    def __init__(self,original,buffer):
        self.original,self.buffer=original,buffer
        self.last_cursor=-1;self.last_target=None
    def apply(self,data,reference_time):
        if not len(self.buffer.q):
            return self.original.apply(data,0.)
        self.last_cursor=self.buffer.cursor
        self.last_target=self.buffer.hand().copy()
        self.original.q[:]=self.last_target
        return self.original.apply(data,0.)

def latency_summary(samples,updates,handoffs):
    values={k:[s[k] for s in samples] for k in [
        'render_seconds','preprocessing_seconds','forward_seconds','postprocessing_seconds','total_seconds']}
    values.update(reference_buffer_update_seconds=updates,sonic_zmq_handoff_seconds=handoffs)
    return {k:{'median_ms':float(np.median(v))*1000,'max_ms':max(v)*1000,'samples':len(v)}
            for k,v in values.items() if v}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--episode',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--latency-only',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    reference,meta=load_reference(args.episode/'candidates/actual')
    # Only the legal initial pose is used. Full expert arrays are not accessible
    # to the runtime; its future body/hand inputs come exclusively from VA.
    initial={k:v[:2].copy() for k,v in reference.items()};del reference
    body_names,hand_names,motor_names=orders()
    text=(ROOT/'GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp').read_text()
    permutation=[int(x) for x in re.search(r'mujoco_to_isaaclab\s*=\s*\{([^}]+)',text).group(1).split(',')]
    stream_names=[motor_names[i] for i in permutation];to_stream=[body_names.index(n) for n in stream_names]
    policy,checkpoint=ModelOnlyPolicy.from_checkpoint(args.checkpoint,'musa');assert policy.policy.config['horizon']==50
    reference_dir=out/'bootstrap';folder=reference_dir/'initial_pose';folder.mkdir(parents=True)
    for name in ['joint_pos.csv','joint_vel.csv','body_pos.csv','body_quat.csv','body_lin_vel.csv','body_ang_vel.csv']:
        (folder/name).write_text('\n'.join((args.episode/'candidates/actual'/name).read_text().splitlines()[:2])+'\n')
    for name in ['metadata.txt','info.txt']:(folder/name).write_bytes((args.episode/'candidates/actual'/name).read_bytes())
    runtime_args=SimpleNamespace(robot='baseline',pelvis_support='none',physics_hz=200,startup_body='freebase-wbc',
      band_angular_damping=10.,band_anchor='source-equilibrium',band='off',startup_support='none')
    # Warm MUSA allocations/kernels before starting the native control loop.
    warm_start=time.monotonic()
    for _ in range(3):
        policy.predict(np.zeros((240,320,3),np.uint8),initial['body_ref_q'][0],initial['hand_ref_q'][0])
    torch.musa.synchronize();warm_seconds=time.monotonic()-warm_start
    runtime=TabletopDDS(initial,meta,runtime_args);worker=VisionWorker(runtime.model,policy)
    buffer=FutureReferenceBuffer()
    runtime.hand_controller=BufferedHand(runtime.hand_controller,buffer)
    context=zmq.Context();publisher=context.socket(zmq.PUB);publisher.bind('tcp://*:5586')
    if args.latency_only:
        samples=[];updates=[];handoffs=[]
        try:
            for anchor in range(0,30,5):
                worker.submit(anchor,runtime.data,runtime.data.qpos[runtime.body.qa],runtime.data.qpos[runtime.hand.qa])
                result=worker.results.get(timeout=60)
                start=time.monotonic();payload=buffer.replenish(anchor,result['prediction']);updates.append(time.monotonic()-start)
                start=time.monotonic();publisher.send(packet(payload[0],payload[1][:,:29][:,to_stream],initial['root_ref_quat'][0]));handoffs.append(time.monotonic()-start)
                samples.append(result)
            measured=latency_summary(samples[1:],updates[1:],handoffs[1:])
            measured['warmup_seconds']=warm_seconds
            write_json(out/'latency.json',measured);print(json.dumps(measured,indent=2))
        finally:
            worker.close();publisher.close();context.term()
        return
    native=NativeSonic(reference_dir,out);play=CSVFollower(out/'policy_logs/motion_playing.csv')
    targets=CSVFollower(out/'target_motion.csv',header=False)
    from collections import deque
    play_pending=deque();target_pending=deque()
    target_to_body=[motor_names.index(n) for n in body_names]
    frame=-1;last_index=None;initialized=False;stream_enabled=False;initial_requested=False;control_requested=False;playing=False
    deadline=time.monotonic()+120;next_step=time.monotonic();next_request=0;records={};frames=[];contacts=[];chunks=[];skipped=[];ages=[];hand_errors=[];body_errors=[];updates=[];handoffs=[];transmissions=[];applied_references=[];cursor_trace=[];stall_ticks=0;failure=None;step_seconds=[]
    try:
        while time.monotonic()<deadline:
            native.pump()
            if native.process.poll() is not None:raise RuntimeError('native SONIC process exited')
            if worker.error:raise RuntimeError(worker.error)
            if not initialized and b'Init Done' in native.captured:
                native.key('\n');initialized=True
            if initialized and not stream_enabled and b'ZMQ STREAMING MODE: ENABLED' in native.captured:
                stream_enabled=True
            if stream_enabled and not initial_requested:
                worker.submit(0,runtime.data,runtime.data.qpos[runtime.body.qa],runtime.data.qpos[runtime.hand.qa]);initial_requested=True;next_request=5
            # Native consumed body targets are authoritative. Read before
            # stitching so the commit boundary includes the newest active tick.
            play_pending.extend(play.read());target_pending.extend(targets.read());new=[]
            while play_pending and target_pending:
                row=play_pending.popleft();target=target_pending.popleft()
                index=int(row[0]);active=row[-1]>.5
                if active:
                    if not playing:playing=True;next_step=time.monotonic()
                    if last_index is not None and index!=last_index+1:
                        raise RuntimeError('Missing SONIC control feedback')
                    last_index=index
                    body=np.asarray(target[7:36])[target_to_body]
                    advanced,error=buffer.consume(body);body_errors.append(error)
                    frame=buffer.cursor
                    if advanced and frame<800:new.append((frame,row[2]/1000))
                    elif not advanced:stall_ticks+=1
            if frame>=800:break
            while not worker.results.empty():
                result=worker.results.get_nowait();anchor=result['anchor'];prediction=result['prediction'];assert prediction.shape==(50,43) and np.isfinite(prediction).all()
                start=time.monotonic();payload=buffer.replenish(anchor,prediction);updates.append(time.monotonic()-start)
                chunks.append(result)
                if payload is None:continue
                packet_start,values=payload
                start=time.monotonic();publisher.send(packet(packet_start,values[:,:29][:,to_stream],initial['root_ref_quat'][0]));handoffs.append(time.monotonic()-start)
                transmissions.append({'capture_anchor':anchor,'apply_at':frame,'packet_start':packet_start,'packet_end':buffer.end,'send_monotonic':start})
                print(f'VA capture={anchor} cursor={frame} packet=[{packet_start},{buffer.end}] latency={result["total_seconds"]:.3f}s',flush=True)
                if not control_requested:
                    native.key(']');control_requested=True
            if playing and frame>=next_request:
                worker.submit(frame,runtime.data,runtime.data.qpos[runtime.body.qa],runtime.data.qpos[runtime.hand.qa]);next_request=(frame//5+1)*5
            ref_time=max(frame,0)*.02
            start=time.monotonic()
            hq,hcmd,band,physical=runtime.step(ref_time,control_requested and runtime.bridge.low_cmd_received,playing)
            step_seconds.append(time.monotonic()-start)
            if playing:
                hand_errors.append(float(np.max(np.abs(hq-buffer.hand()))))
                cursor_trace.append((buffer.cursor,runtime.hand_controller.last_cursor))
            if new:
                if frames and frame!=frames[-1]+1:skipped.extend(range(frames[-1]+1,frame))
                for f,_ in new[:-1]:skipped.append(f)
                state=robot_state(runtime.model,runtime.data,runtime.body,runtime.hand)
                for k,v in state.items():records.setdefault(k,[]).append(v)
                
                for k,v in [('qpos',runtime.data.qpos),('qvel',runtime.data.qvel),('ctrl',runtime.data.ctrl)]:
                    if k not in state:records.setdefault(k,[]).append(v.copy())
                records.setdefault('sim_timestamps',[]).append(runtime.data.time);frames.append(frame);ages.append(max(0,time.time()-new[-1][1]));contacts.append({'frame':frame,**physical,'contacts':runtime.evaluator.last_contacts})
                applied_references.append(buffer.q[frame].copy())
                if frame%50==0:print(f'physics frame={frame} lift={physical["lift_m"]:.4f} hold={runtime.evaluator.hold:.3f}',flush=True)
            if playing and runtime.data.xpos[runtime.model.body('pelvis').id,2]<.3:raise RuntimeError('robot_fell')
            if not np.isfinite(runtime.data.qpos).all() or not np.isfinite(runtime.data.qvel).all():raise RuntimeError('nonfinite_physics')
            next_step+=.005;delay=next_step-time.monotonic()
            if delay>0:time.sleep(delay)
        if frame<800 and failure is None:failure='timeout_before_800_frames'
    except Exception as exc:failure=f'{type(exc).__name__}: {exc}';print(failure,flush=True)
    finally:
        native.close();worker.close();play.close();targets.close();publisher.close();context.term()
    arrays={k:np.asarray(v) for k,v in records.items()};save_arrays(out,arrays);np.save(out/'reference_frame.npy',np.asarray(frames,dtype=np.int64));np.save(out/'timestamps.npy',np.asarray(frames)*.02)
    write_json(out/'contacts.json',contacts)
    write_json(out/'transmissions.json',transmissions)
    np.save(out/'reference_buffer.npy',buffer.q)
    np.save(out/'reference_source_anchor.npy',buffer.source)
    np.save(out/'applied_reference.npy',np.asarray(applied_references))
    np.save(out/'body_hand_cursor.npy',np.asarray(cursor_trace))
    for result in chunks:
        result.pop('rgb')
    if chunks:
        np.savez_compressed(out/'va_chunks.npz',anchor=[c['anchor'] for c in chunks],predictions=[c['prediction'] for c in chunks],body_q=[c['body_q'] for c in chunks],hand_q=[c['hand_q'] for c in chunks])
    report=runtime.evaluator.report(runtime.data);complete=frames==list(range(800));duration=float(np.ptp(arrays.get('sim_timestamps',[]))) if len(frames)>1 else 0
    timing=complete and not skipped and abs(duration-15.98)<=.1 and max(ages,default=999)<=.04
    log=native.captured.decode(errors='replace')
    resets=len(re.findall(r'Catch-up: Reset to frame 0',log))
    waits=len(re.findall(r'Motion streamed completed and waiting',log))
    skew=max((abs(body-hand) for body,hand in cursor_trace),default=999)
    scheduling=bool(complete and stall_ticks==0 and resets==1 and waits==0 and buffer.stale_chunks==0)
    synced=bool(skew==0 and max(body_errors,default=999)<=1e-5 and max(hand_errors,default=999)<=1e-8)
    latency=latency_summary(chunks,updates,handoffs)
    sent={p['packet_end']:p['send_monotonic'] for p in transmissions}
    ack_seconds=[ack-sent[end] for end,ack in native.merge_acks if end in sent]
    if ack_seconds:
        latency['sonic_merge_ack_seconds']={'median_ms':float(np.median(ack_seconds))*1000,'max_ms':max(ack_seconds)*1000,'samples':len(ack_seconds)}
    write_json(out/'latency.json',latency)
    report.update(episode_completed=frame>=800,recorded_frames=len(frames),expected_frames=800,frame_coverage_complete=complete,skipped_frames=sorted(set(skipped)),failure=failure,
       physics_hz=200,sonic_control_hz=50,replan_requested_hz=10,execute_frames=5,horizon=50,prediction_count=len(chunks),physical_duration_seconds=duration,
       sync_age_max_seconds=max(ages,default=None),timing_alignment_pass=timing,hand_interface_max_error=max(hand_errors,default=None),
       latency_breakdown=latency,warmup_seconds=warm_seconds,prediction_latency_seconds=[c['total_seconds'] for c in chunks],
       inference_deadline_misses=sum(c['total_seconds']>.1 for c in chunks),reference_source='live VA predictions only; two-frame initial-pose bootstrap',
       unpredicted_root_orientation='held at initial world quaternion; no expert future root trajectory',runtime_command=native.command,
       checkpoint=str(args.checkpoint),SONIC_source_modified=False,hand_pd={'kp':12.,'kd':.3},
       reference_scheduling_pass=scheduling,body_hand_sync_pass=synced,max_body_hand_timeline_skew_frames=skew,
       body_reference_max_error=max(body_errors,default=None),native_cursor_resets=resets,native_lookahead_waits=waits,stall_ticks=stall_ticks,
       buffer_lookahead_frames=46,buffer_endpoint_hold_guard_frames=110,buffer_commit_lead_frames=buffer.COMMIT_LEAD,
       stale_chunks=buffer.stale_chunks,control_step_max_seconds=max(step_seconds,default=None),
       closed_loop_pass=bool(complete and timing and scheduling and synced and failure is None and report['physical_grasp_success'] and not report['fall_steps']),
       grasp_pass=bool(report['physical_grasp_success'] and not report['fall_steps'] and not report['robot_self_contact_steps'] and not report['left_arm_environment_contact_steps']))
    write_json(out/'report.json',report);write_json(out/'metadata.json',meta)
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(list,dict))},indent=2))
if __name__=='__main__':main()
