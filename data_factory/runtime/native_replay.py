"""Native controller + unchanged TabletopDDS.step; passive transport/snapshot recorder."""
import fcntl
import gc
import os
import pty
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np
from .common import ROOT,SONIC,PROJECT,ROBOT,BUNDLE,HASHES,dump,read,digest,require,table,jsonl
from .native_build import BUILD_ROOT
from .native_scene import SnapshotReader,mujoco

sys.path[:0]=[str(PROJECT/'scripts'),str(SONIC)]
import run_sonic_grasp as api
from expert_trajectory import orders
from reference_io import export_reference

DECISION_DTYPE=np.dtype([('header','<u8',(10,)),('encoder_input','<f4',(1762,)),('decoder_input','<f4',(994,)),
    ('token','<f4',(64,)),('action','<f4',(29,)),('previous_action','<f4',(29,)),('command','<f4',(5,29))])


def export_native_reference(a,m,out):
    bn,hn,_=orders()
    arrays=dict(a,timestamps=a['time_s'],hand_ref_dq=np.zeros_like(a['hand_ref_q']))
    metadata=dict(candidate_type='canonical_expert_reference',root_velocities=(np.gradient(a['root_ref_pos'],.02,axis=0),np.zeros_like(a['root_ref_pos'])))
    export_reference(out,arrays,bn,hn,metadata)
    # Lossless decimal serialization of the original numeric values; no motion adaptation.
    for file,key in [('joint_pos','body_ref_q'),('joint_vel','body_ref_dq'),('body_pos','root_ref_pos'),('body_quat','root_ref_quat'),('hand_ref_q','hand_ref_q')]:
        path=out/(file+'.csv'); header=path.read_text().splitlines()[0]
        np.savetxt(path,a[key],delimiter=',',header=header,comments='',fmt='%.17g')
        require(np.array_equal(np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2),a[key]),'CSV conversion changed '+key)
    # Native mode 0 uses joint positions/velocities and root orientation only.
    return arrays


def configure_dds(out):
    from unitree_sdk2py.core import channel
    from xml.sax.saxutils import escape
    channel.ChannelConfigHasInterface=channel.ChannelConfigHasInterface.replace('/tmp/cdds.LOG',escape(str(out/'dds_python.log')))
    os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain Id="any"><Tracing><Verbosity>config</Verbosity><OutputFile>'+escape(str(out/'dds_native.log'))+'</OutputFile></Tracing></Domain></CycloneDDS>'


def replay(a,m,out,gpu=0):
    replay_started=time.perf_counter(); active_started=None
    out.mkdir(); (out/'rgb').mkdir()
    lock=(ROOT/'outputs/canonical_va/native_dds.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    process_list=subprocess.run(['pgrep','-af','g1_deploy_onnx_ref|replay_validate_kimodo.py|run_sonic_grasp.py'],capture_output=True,text=True).stdout.splitlines()
    # Other jobs may run in the host namespace. Only controllers in OUR network namespace share DDS.
    own_net=os.readlink('/proc/self/ns/net');foreign=[]
    for line in process_list:
        pid=int(line.split()[0])
        if pid==os.getpid() or 'pgrep' in line:continue
        try:
            if os.readlink(f'/proc/{pid}/ns/net')==own_net:foreign.append(line)
        except FileNotFoundError:pass
    require(not foreign,'Shared DDS already occupied: '+str(foreign))
    for name,h in HASHES.items(): require(digest(BUNDLE/name)==h,'Frozen SONIC model changed')
    binary=BUILD_ROOT/'target/release/g1_deploy_onnx_ref'
    require(binary.is_file(),'Native instrumented build required: python -m canonical_va.native_build')
    build=read(BUILD_ROOT/'build_validation.json'); require(digest(binary)==build['sha256'],'Instrumented executable changed')
    os.environ['MUSA_VISIBLE_DEVICES']=str(gpu); os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['SONIC_REPO']=str(SONIC)
    arrays=export_native_reference(a,m,out/'reference/motion')
    configure_dds(out)
    from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge as bridge_module
    OriginalBridge=bridge_module.UnitreeSdk2Bridge
    receipts=[]; selection={}; step_context={'enabled':False,'step':-1}
    class LoggedBridge(OriginalBridge):
        @property
        def low_cmd(self):
            value=self._canonical_low_cmd
            # Getter runs under the ORIGINAL step() low_cmd_lock; this is the exact selected message.
            if step_context['enabled']:
                selection['message']=value
            return value
        @low_cmd.setter
        def low_cmd(self,value):
            self._canonical_low_cmd=value
            if hasattr(self,'low_cmd_lock'):
                receipts.append(dict(command_number=int(value.reserve[0]),publication_id=int(value.reserve[1]),receipt_time=time.monotonic_ns()))
    bridge_module.UnitreeSdk2Bridge=LoggedBridge
    original_load=api.load_scene
    api.load_scene=lambda **kw: original_load(**dict(kw,robot_path=ROBOT,scene_path=PROJECT/'scenes/tabletop.xml'))
    cfg=SimpleNamespace(robot='baseline',pelvis_support='none',physics_hz=200,startup_body='freebase-wbc',
        band_angular_damping=10.,band_anchor='source-equilibrium',band='off',startup_support='none')
    try:
        runtime=api.TabletopDDS(arrays,dict(m,initial_pose={'name':'neutral_standing_v1'}),cfg)
    finally:
        api.load_scene=original_load; bridge_module.UnitreeSdk2Bridge=OriginalBridge
    reader=SnapshotReader(runtime.model,runtime.data)
    original_write=runtime.bridge.low_state_puber.Write
    observation_sequence={'value':0}
    def tagged_write(message,*args,**kw):
        message.reserve[0]=observation_sequence['value']
        return original_write(message,*args,**kw)
    runtime.bridge.low_state_puber.Write=tagged_write
    runtime_config=dict(backend='native_dds',network_namespace=own_net,binary=str(binary),binary_sha256=digest(binary),original_build=read(BUILD_ROOT/'sources.json'),
        controller='original G1Deploy Control/GatherObservations/CreatePolicyCommand/LowCommandWriter',
        simulator='dl-group2/scripts/run_sonic_grasp.py:TabletopDDS.step',
        startup='original freebase-wbc, native InitControl, five observed standby ticks then keyboard t',
        physics_hz=200,control_hz=50,command_publisher_hz=500,hand_pd={'kp':12.,'kd':.3},camera=reader.camera,
        case=m['case'],physical_gpu=gpu,initial_qpos=runtime.data.qpos.tolist(),initial_qvel=runtime.data.qvel.tolist(),
        command_identity='MotorCommand bookkeeping -> LowCmd.reserve[0]; publication sequence -> reserve[1]',
        observation_identity='LowState.reserve[0] tags exact published physical snapshot; original tick unmodified',
        timing='Asynchronous native DDS; no forced four-step interval or lockstep barrier',
        rendering='Offline exact integration-state restore; no live rendering stalls')
    dump(out/'metadata.json',dict(episode_id=m['episode_id'],producer_type=m['producer_type'],runtime=runtime_config,
        reference_hz=50,rgb_hz=10,observation_semantic='PRE_ACTION published physical state actually selected by native GatherRobotStateToLogger; physics may advance during inference'))
    source_files=[PROJECT/'scripts'/x for x in ('run_sonic_grasp.py','generate_freebase_expert.py','hand_trajectory_controller.py','scene.py')]
    dump(out/'provenance.json',dict(source_reference_sha256=m['source_reference_sha256'],runtime=runtime_config,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in source_files},
        implementation_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'canonical_va').glob('*.py')}))
    for rel,h in read(out/'provenance.json')['implementation_hashes'].items():
        source=ROOT/rel;require(digest(source)==h,'Recorder changed before launch')
        archived=out/'recording_source'/rel;archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,archived)
    shutil.copy2(BUILD_ROOT/'instrumentation.patch',out/'instrumentation.patch')
    live_tmp=tempfile.TemporaryDirectory(prefix='canonical_native_'); live=__import__('pathlib').Path(live_tmp.name)
    env=os.environ.copy(); env['CANONICAL_TRACE_DIR']=str(live)
    # Explicit simulator mode: the native audio warning thread remains
    # available, but its startup-only volume request is not required because
    # this isolated DDS namespace has no audio service.
    env['CANONICAL_SIMULATOR_RUNTIME']='1'
    command=[str(binary),'lo',str(BUNDLE/'model_decoder.onnx'),str(out/'reference'),
        '--obs-config',str(BUNDLE/'observation_config.yaml'),'--encoder-file',str(BUNDLE/'model_encoder.onnx'),
        '--input-type','keyboard','--policy-precision','32','--disable-crc-check','--enable-csv-logs','--logs-dir',str(live/'policy_logs'),
        '--target-motion-logfile',str(live/'target_motion.csv'),'--record-input-file',str(live/'playback_cursor.csv')]
    dump(out/'launch.json',dict(command=command,environment={'MUSA_VISIBLE_DEVICES':str(gpu),'CANONICAL_TRACE_DIR':str(live)}))
    sidecar_root=out.parent.parent
    startup_native_file=sidecar_root/'native_startup_events.jsonl'
    startup_event_file=sidecar_root/'startup_event_timeline.json'
    try: startup_native_file.unlink()
    except FileNotFoundError: pass
    env['CANONICAL_STARTUP_TIMING_FILE']=str(startup_native_file)
    master,slave=pty.openpty(); child=subprocess.Popen(command,cwd=SONIC/'gear_sonic_deploy',env=env,stdin=slave,stdout=slave,stderr=slave,start_new_session=True)
    startup_events={}; native_event_lines=[]; native_detail_records=[]; startup_launch_ns=time.monotonic_ns()
    def startup_event(name, source, physics_step=0, sim_time=0., event_ns=None, observed_ns=None):
        if name in startup_events:return
        startup_events[name]=dict(event=name,monotonic_ns=int(event_ns or time.monotonic_ns()),
            observed_monotonic_ns=int(observed_ns or time.monotonic_ns()),physics_steps=int(physics_step),
            simulated_physics_seconds=float(sim_time),source=source)
    startup_event('native_process_launch','canonical_va/native_replay.py:132',0,0.,startup_launch_ns,startup_launch_ns)
    os.close(slave); log=(live/'controller.log').open('wb'); captured=bytearray()
    cursor=api.PlaybackCursor(live,len(a['time_s'])); raw=[]; complete=False; initialized=requested=False; failure=None
    start=time.monotonic(); next_step=start; step=0; was_gc=gc.isenabled()
    def pump():
        while select.select([master],[],[],0)[0]:
            try: chunk=os.read(master,65536)
            except OSError: break
            if not chunk: break
            captured.extend(chunk); log.write(chunk); log.flush()
    def consume_native_events():
        if not startup_native_file.is_file(): return
        lines=startup_native_file.read_text().splitlines()
        for line in lines[len(native_event_lines):]:
            native_event_lines.append(line)
            parts=line.split('\t',2)
            if len(parts)<2: continue
            name,stamp=parts[:2]
            try: stamp_ns=int(stamp)
            except ValueError: continue
            if len(parts)==3:
                native_detail_records.append(dict(event=name,monotonic_ns=stamp_ns,detail=parts[2]))
            mapped={
                'native_constructor_enter':('native_constructor_enter','native g1_deploy_onnx_ref.cpp:2222'),
                'channel_factory_init_done':('channel_factory_init_done','native g1_deploy_onnx_ref.cpp:2236'),
                'dex3_initialize_start':('dex3_initialize_start','native g1_deploy_onnx_ref.cpp:2240'),
                'dex3_initialize_done':('dex3_initialize_done','native g1_deploy_onnx_ref.cpp:2242'),
                'audio_thread_initialize_start':('audio_thread_initialize_start','native g1_deploy_onnx_ref.cpp:2252'),
                'audio_thread_initialize_done':('audio_thread_initialize_done','native g1_deploy_onnx_ref.cpp:2254'),
                'motion_switcher_client_construct_start':('motion_switcher_client_construct_start','native g1_deploy_onnx_ref.cpp:2320'),
                'motion_switcher_client_construct_done':('motion_switcher_client_construct_done','native g1_deploy_onnx_ref.cpp:2322'),
                'motion_switcher_release_done':('motion_switcher_release_done','native g1_deploy_onnx_ref.cpp:2307'),
                'lowstate_channel_init_done':('lowstate_channel_init_done','native g1_deploy_onnx_ref.cpp:2317'),
                'motion_data_loaded':('motion_data_loaded','native g1_deploy_onnx_ref.cpp:2333'),
                'policy_init_start':('policy_init_start','native g1_deploy_onnx_ref.cpp:2352'),
                'policy_init_done':('policy_init_done','native g1_deploy_onnx_ref.cpp:2356'),
                'policy_graph_capture_done':('policy_graph_capture_done','native g1_deploy_onnx_ref.cpp:2366'),
                'encoder_graph_capture_done':('encoder_graph_capture_done','native g1_deploy_onnx_ref.cpp:2422'),
                'native_threads_started':('native_threads_started','native g1_deploy_onnx_ref.cpp:2653'),
                'first_native_lowstate_callback':('first_lowstate_received','native g1_deploy_onnx_ref.cpp:2704'),
                'init_control_first_valid':('init_control_first_valid','native g1_deploy_onnx_ref.cpp:2808'),
                'init_control_valid_call_50':('init_control_valid_call_50','native g1_deploy_onnx_ref.cpp:2809'),
                'init_control_valid_call_100':('init_control_valid_call_100','native g1_deploy_onnx_ref.cpp:2810'),
                'init_control_valid_call_150':('init_control_valid_call_150','native g1_deploy_onnx_ref.cpp:2811'),
                'native_init_done':('native_init_done','native g1_deploy_onnx_ref.cpp:2834'),
            }.get(name)
            if mapped:
                startup_event(mapped[0],mapped[1],step,float(runtime.data.time),stamp_ns,time.monotonic_ns())
    try:
        while time.monotonic()-start<180:
            pump()
            consume_native_events()
            require(child.poll() is None,'Native controller exited: see controller.log')
            if not initialized and b'Init Done' in captured:
                initialized=True; os.write(master,b']'); print('NATIVE_CONTROLLER_READY',flush=True)
            if initialized and not requested and runtime.bridge.low_cmd_received and cursor.standby_ticks>=5:
                if cursor.standby_ticks>=5:
                    startup_event('five_standby_observations','dl-group2/scripts/run_sonic_grasp.py:102',step,float(runtime.data.time))
                os.write(master,b't'); requested=True; startup_event('playback_request_sent','canonical_va/native_replay.py:148',step,float(runtime.data.time)); print('NATIVE_PLAY_REFERENCE',flush=True)
            playing_before=cursor.started; new=cursor.update()
            if cursor.started and not playing_before:
                startup_event('first_playing_tick','canonical_va/native_replay.py:150',step,float(runtime.data.time))
                active_started=time.perf_counter(); next_step=time.monotonic(); gc.disable()
            if cursor.completed: complete=True; break
            frame=cursor.frame if cursor.started else 0
            pre=reader.integration_state(); pre_time=float(runtime.data.time)
            observation_sequence['value']=step+1
            step_context.update(enabled=True,step=step); selection.clear()
            begin=time.monotonic_ns()
            hq,hcmd,band,metrics=runtime.step(float(a['time_s'][frame]),runtime.bridge.low_cmd_received and initialized,cursor.started)
            end=time.monotonic_ns(); step_context['enabled']=False
            msg=selection.get('message'); cid=int(msg.reserve[0]) if msg is not None else 0; pub=int(msg.reserve[1]) if msg is not None else 0
            raw.append(dict(sim_step_id=step,observation_sequence=step+1,pre_integration_state=pre,post_integration_state=reader.integration_state(),
                pre_time=pre_time,post_time=float(runtime.data.time),begin_ns=begin,end_ns=end,command_number=cid,publication_id=pub,
                applied_ctrl=runtime.data.ctrl.copy(),hand_ref_q=hq.copy(),hand_reference_index=frame,playing=cursor.started,
                applied_motor_values=np.array([[v.q,v.dq,v.kp,v.kd,v.tau] for v in msg.motor_cmd[:29]]) if msg is not None else np.zeros((29,5))))
            require(not band and not np.any(runtime.data.xfrc_applied) and not np.any(runtime.data.qfrc_applied),'Unexpected support/external force')
            if new and frame%100==0: print(f'NATIVE_FRAME {frame} sim={runtime.data.time:.3f} lift={metrics.get("lift_m",0):.4f} hold={metrics.get("hold_seconds",0):.3f}',flush=True)
            require(runtime.evaluator.fall_steps==0,'Robot fell; no reset/retry')
            step+=1; next_step+=.005; time.sleep(max(0,next_step-time.monotonic()))
        if not complete: failure='Native playback incomplete'
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
    finally:
        if was_gc: gc.enable()
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
        pump(); log.close(); os.close(master); cursor.close()
        consume_native_events()
        if 'five_standby_observations' not in startup_events and cursor.standby_ticks>=5:
            startup_event('five_standby_observations','dl-group2/scripts/run_sonic_grasp.py:102',step,float(runtime.data.time))
        ordered=sorted(startup_events.values(),key=lambda x:x['monotonic_ns'])
        names=[x['event'] for x in ordered]
        for prev,cur in zip(ordered,ordered[1:]):
            cur['interval_from_previous']=dict(wall_seconds=(cur['monotonic_ns']-prev['monotonic_ns'])/1e9,
                simulated_seconds=cur['simulated_physics_seconds']-prev['simulated_physics_seconds'],
                physics_steps=cur['physics_steps']-prev['physics_steps'])
        dump(startup_event_file,dict(events=ordered,event_order=names,
            native_detail_records=native_detail_records,
            monotonic_clock='CLOCK_MONOTONIC/steady_clock nanoseconds',
            native_event_file=str(startup_native_file),
            semantics='Native events are exact native steady-clock timestamps; physics fields are nearest Python observation at event discovery.'))
        for path in live.iterdir():
            if path.is_dir(): shutil.copytree(path,out/path.name)
            else: shutil.copy2(path,out/path.name)
        live_tmp.cleanup()
        if raw: table(out/'native_physics.npz',raw)
        jsonl(out/'native_receipts.jsonl',receipts)
        native_finished=time.perf_counter()
        metrics=runtime.evaluator.report(runtime.data)
        physical=dict(physical_success=bool(metrics['physical_grasp_success'] and not metrics['fall_steps'] and complete),
            cube_max_lift=metrics['max_lift_m'],qualifying_hold_duration=metrics['max_continuous_hold_seconds'],
            opposing_contact_valid=False,cube_table_contact_during_hold=False,fall_detected=bool(metrics['fall_steps']),
            replay_complete=complete,canonical_evaluator=metrics,camera_valid=True,failure=failure,
            physics_steps=sum(x['playing'] for x in raw),startup_physics_steps=sum(not x['playing'] for x in raw),
            startup_simulation_seconds=next((x['pre_time'] for x in raw if x['playing']),None),
            cursor_missing_ticks=cursor.missing_ticks)
        physical['stage_timings']=dict(startup=(active_started or native_finished)-replay_started,native_rollout=native_finished-(active_started or native_finished))
        effective=[x['body'] for x in metrics['final_cube_contacts'] if x['normal_force_n']>1e-4]
        physical['opposing_contact_valid']=any(m.get('task_arm','right')+'_hand_thumb' in x for x in effective) and any(m.get('task_arm','right')+'_hand_index' in x or m.get('task_arm','right')+'_hand_middle' in x for x in effective)
        dump(out/'replay_validation.json',physical)
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
    from .native_truth import materialize
    return materialize(out,a,m,physical)
