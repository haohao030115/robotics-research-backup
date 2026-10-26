"""V3 native closed-loop integration. Existing SONIC process/reset are unchanged.

Derived from the V2 native transport loop; policy, capture schedule and exact
slot validation are independent. In strict mode stale consumption aborts.
"""
from collections import deque
import copy
from pathlib import Path
import time
import sys
import numpy as np
import zmq
from sonic.release import bundle_provenance, digest, ExternalTokenAdapter
from sonic.runtime import create_runtime, NativeTokenSonic, TimestampedHand, runtime_provenance
from sonic.token_handoff import TokenHandoff
from utils.config import dump_json
from .latency import command_alignment
from .training import load_checkpoint
from .inference import CurrentPolicy, latency_report
from .runtime import CurrentVisionWorker, ProcessVisionWorker, TwoTickBuffer, current_state

def run(config, checkpoint_path, episode, output):
    sonic = bundle_provenance(config['sonic'])
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    model, saved = load_checkpoint(checkpoint_path, config['training']['device'])
    policy = CurrentPolicy(model)
    if saved['provenance']['sonic'] != sonic:
        raise ValueError('Checkpoint SONIC provenance differs')
    strict = config['rollout'].get('strict_timing', True)
    render_shape = (config['rollout'].get('render_height',120), config['rollout'].get('render_width',160))
    if min(render_shape)<1 or render_shape[1]*3!=render_shape[0]*4:
        raise ValueError('Render size must preserve the original 4:3 camera aspect')
    config['rollout']['render_height'],config['rollout']['render_width']=render_shape
    provenance = {'body_source': 'pred', 'hand_source': 'pred', 'va_config': config, 'sonic': sonic, 'runtime': runtime_provenance(config),
        'dataset': episode.provenance, 'body_joint_order': episode.body_order, 'hand_joint_order': episode.hand_order,
        'checkpoint': str(Path(checkpoint_path).resolve()), 'checkpoint_sha256': digest(checkpoint_path),
        'timing_contract': config['timing'],
        'v3_source_sha256': {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    dump_json(output/'provenance.json', provenance)
    # A passive timestamp wrapper preserves the existing native bridge callback.
    sys.path.insert(0, config['sonic']['repository'])
    from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
    dds_events = []
    original_callback = UnitreeSdk2Bridge.LowCmdHandler
    def measured_callback(bridge, message):
        received = time.monotonic()
        original_callback(bridge, message)
        dds_events.append((received, [m.q for m in message.motor_cmd[:29]]))
    UnitreeSdk2Bridge.LowCmdHandler = measured_callback
    try:
        runtime, bootstrap, follower = create_runtime(config, episode, output)
    finally:
        UnitreeSdk2Bridge.LowCmdHandler = original_callback
    hand = TimestampedHand(runtime.hand_controller); runtime.hand_controller = hand
    worker_kind = config['rollout'].get('vision_worker','process')
    if worker_kind not in ('process','thread'):raise ValueError('Invalid vision worker')
    worker_class = ProcessVisionWorker if worker_kind=='process' else CurrentVisionWorker
    worker = worker_class(runtime.model, policy, render_shape)
    native = None; context = None; pub = None; token_log = action_log = None
    records = []; sync = []; frames = []; latencies = []; requests = []; failure = None
    control_times = []; previous_action = np.zeros(29); frame = -1
    buffer = TwoTickBuffer(); stale_ticks = 0
    handoff = TokenHandoff(); step_dt = 1/config['rollout']['physics_hz']
    length = len(episode.times)
    try:
        if not worker.ready.wait(timeout=30) or worker.error:
            raise RuntimeError(worker.error or 'Renderer initialization timed out')
        first_entry = None
        context = zmq.Context(); pub = context.socket(zmq.PUB); pub.setsockopt(zmq.LINGER, 0)
        pub.bind(f'tcp://*:{config["rollout"]["zmq_port"]}')
        adapter = ExternalTokenAdapter(config['sonic']['repository'])
        native = NativeTokenSonic(config, bootstrap, output)
        token_log = follower(output/'policy_logs/token_state.csv'); action_log = follower(output/'policy_logs/action.csv')
        token_pending, action_pending = deque(), deque()
        initialized = enabled = started = False; preload_start = None
        next_step = time.monotonic(); deadline = next_step+config['rollout']['timeout_seconds']
        while time.monotonic() < deadline:
            native.pump()
            if native.process.poll() is not None:
                raise RuntimeError('Native SONIC exited')
            if worker.error:
                raise RuntimeError(worker.error)
            if not initialized and b'Init Done' in native.captured:
                native.key('\n'); initialized = True
            if initialized and not enabled and b'ZMQ STREAMING MODE: ENABLED' in native.captured:
                enabled = True
                worker.submit(-1, runtime.data, current_state(runtime))
                initial = worker.results.get(timeout=60)
                buffer.add(initial,-1,start=0)
                first_entry = buffer.take(0); handoff.propose(first_entry)
                latencies.append(initial)
                preload_start = time.monotonic()
            if enabled and not started:
                pub.send(adapter.message(first_entry['token'], 0))
                if time.monotonic()-preload_start > .2 and b'Protocol v4: Received 64D token' in native.captured:
                    native.key(']'); started = True
            while not worker.results.empty():
                result = worker.results.get_nowait(); latencies.append(result)
                buffer.add(result,frame)
            if handoff.pending is None and frame >= 0:
                entry = buffer.take(frame+1)
                if entry is not None:
                    handoff.propose(entry)
                    dispatch_start = time.monotonic()
                    pub.send(adapter.message(entry['token'],entry['execution_index']))
                    entry['enqueued'] = time.monotonic()
                    entry['dispatch_ms'] = (entry['enqueued']-dispatch_start)*1000
            token_pending.extend(token_log.read()); action_pending.extend(action_log.read())
            if len(token_pending) > 1 or len(action_pending) > 1:
                raise RuntimeError('Missed control snapshot; actual tick coverage unavailable')
            new = None
            if token_pending and action_pending:
                token_row, action_row = token_pending.popleft(), action_pending.popleft()
                if token_row[:5] != action_row[:5]:
                    raise RuntimeError('Native token/action metadata mismatch')
                if int(token_row[0]) != frame+1:
                    raise RuntimeError('Non-contiguous native control ticks')
                frame += 1
                if frame >= length:
                    break
                current_hand = handoff.consume(frame, token_row[5:])
                stale_ticks += int(handoff.current['execution_index'] != frame)
                if strict and handoff.current['execution_index'] != frame:
                    raise RuntimeError(f'V3 deadline missed at tick {frame}; refusing stale slot')
                if handoff.current.get('consumed_monotonic') is None:
                    handoff.current['consumed_monotonic'] = token_row[3]/1000
                    handoff.current['observed_monotonic'] = time.monotonic()

                hand.set(frame, current_hand); sync.append((frame, hand.index))
                previous_action = np.asarray(action_row[5:]); control_times.append(token_row[1]/1000)
                new = frame; frames.append(frame)
            if enabled and not started:
                time.sleep(step_dt)
                next_step = time.monotonic()
                continue
            if new is None and time.monotonic() < next_step:
                # Poll transport more often than physics without adding steps.
                time.sleep(max(0, min(.001, next_step-time.monotonic())))
                continue
            applied_hand, _, _, physical = runtime.step(max(frame, 0)/config['data']['control_hz'],
                started and runtime.bridge.low_cmd_received, frame >= 0)
            if frame >= 0 and not np.allclose(applied_hand, hand.target, atol=1e-8, rtol=0):
                raise RuntimeError('Independent hand controller did not apply the acknowledged target')
            if new is not None:
                if handoff.current.get('hand_applied_monotonic') is None:
                    handoff.current['hand_applied_monotonic'] = time.monotonic()
                state_now = current_state(runtime)
                state_now['previous_action'] = previous_action.copy() # diagnostic log only; never a policy input
                from expert_trajectory import robot_state
                measured = robot_state(runtime.model, runtime.data, runtime.body, runtime.hand)
                records.append({'measured': measured, 'contacts': copy.deepcopy(runtime.evaluator.last_contacts), 'frame': frame, 'sim_time': runtime.data.time, 'state': state_now,
                                'token': np.asarray(token_row[5:]), 'hand': hand.target.copy(),
                    'prediction_anchor':handoff.current['anchor'], 'slot':handoff.current['chunk_index'],
                    'intended_execution_index':handoff.current['execution_index'],
                    'native_consumption_monotonic':token_row[3]/1000, 'hand_applied_monotonic':time.monotonic()})
                if frame % 2 == 1 and frame+2 < length:
                    worker.submit(frame, runtime.data, current_state(runtime))
                    requests.append(frame)
                if frame % int(config['data']['control_hz']) == 0:
                    print(f'V3 frame={frame} lift={physical["lift_m"]:.4f} hold={runtime.evaluator.hold:.3f}', flush=True)
            if frame >= 0 and runtime.data.xpos[runtime.model.body('pelvis').id, 2] < .3:
                raise RuntimeError('Robot fell')
            next_step += step_dt; delay = next_step-time.monotonic()
            if delay > 0:
                time.sleep(min(delay, .001))
        if frames != list(range(length)):
            failure = 'Incomplete control coverage'
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    finally:
        if native is not None:
            native.close()
        worker.close()
        while not worker.results.empty():
            latencies.append(worker.results.get_nowait())
        for log in (token_log, action_log):
            if log is not None:
                log.close()
        if pub is not None:
            pub.close()
        if context is not None:
            context.term()
    if records:
        dump_json(output/'control_timeline.json',[{k:r[k] for k in ('frame','sim_time','prediction_anchor','slot','intended_execution_index','native_consumption_monotonic','hand_applied_monotonic')} for r in records])
        np.savez_compressed(output/'physics.npz', **{k: [r['measured'][k] for r in records] for k in records[0]['measured']})
        dump_json(output/'contacts.json', [r['contacts'] for r in records])
        np.savez_compressed(output/'live_observations.npz', reference_frame=frames,
            sim_timestamps=[r['sim_time'] for r in records], token=[r['token'] for r in records],
            hand_ref=[r['hand'] for r in records], **{k: [r['state'][k] for r in records] for k in records[0]['state']})
    # JSON timing records exclude NumPy prediction arrays, saved separately.
    if latencies:
        np.savez_compressed(output/'predictions.npz', capture_anchor=[r['anchor'] for r in latencies],
            intended_execution_index=[r['execution_index'] for r in latencies],
            token=[r['token'] for r in latencies], hand_ref=[r['hand'] for r in latencies])
    dump_json(output/'timing.json', [{k: v for k, v in r.items() if k not in ('token', 'hand')} for r in latencies])
    dump_json(output/'handoffs.json', [{k: v for k, v in r.items() if k not in ('token', 'hand')}
                                     for r in handoff.applied])
    timing_rows = [r['latency_ms'] for r in latencies]
    for entry in handoff.applied:
        if 'dispatch_ms' in entry:
            timing_rows.append({'sonic_dispatch': entry['dispatch_ms']})
        if entry.get('consumed_monotonic') is not None:
            timing_rows.append({'observation_to_token_consumption': (entry['consumed_monotonic']-entry['requested'])*1000})
    alignment, command_matches = command_alignment(output, handoff.applied, dds_events, config)
    dump_json(output/'execution_alignment.json', {'rows':alignment, 'matching':command_matches})
    command_rows = [{'observation_to_command':r['observation_to_command_ms'],
                     'token_consumption_to_body_command':r['token_consumption_to_body_command_ms']}
                    for r in alignment if 'observation_to_command_ms' in r and r['observation_index']>=0]
    timing_rows.extend(command_rows)
    dump_json(output/'latency.json', {'units':'milliseconds','stages':latency_report(timing_rows),
        'observation_to_command_definition':'capture to max(value-matched body DDS receipt, paired hand application); steady state only',
        'command_matching':command_matches,
        'observation_to_command_by_slot':{str(slot):latency_report([{'observation_to_command':r['observation_to_command_ms']} for r in alignment if r['observation_index']>=0 and r['slot']==slot and 'observation_to_command_ms' in r]) for slot in (0,1)},
        'cold_start_included_in_forward_stages':True,
        'steady_state_stages':latency_report([r['latency_ms'] for r in latencies if r['anchor']>=0]+command_rows)})
    report = runtime.evaluator.report(runtime.data)
    if records:
        cube = np.array([r['measured']['cube_pos'] for r in records])
        report['maximum_cube_height_m'] = float(cube[:,2].max())
        report['final_cube_displacement_m'] = (cube[-1]-cube[0]).tolist()
    complete = frames == list(range(length))
    live = [r for r in latencies if r['anchor'] >= 0]
    period = float(np.median(np.diff(control_times))) if len(control_times) > 1 else None
    skew = max((abs(a-b) for a, b in sync), default=None)
    report.update(body_source='pred', hand_source='pred', episode=episode.id, configured_visual_hz=config['rollout']['visual_hz'],
        configured_control_hz=config['data']['control_hz'], actual_control_hz=None if period is None else 1/period,
        recorded_frames=len(frames), expected_frames=length, frame_coverage_complete=complete,
        prediction_count=len(live), dropped_visual_requests=worker.dropped_requests,
        visual_deadline_misses=sum(r['total_seconds'] > 1/config['rollout']['visual_hz'] for r in live),
        execution_deadline_misses=sum(r['applied_index'] > r['execution_index'] for r in handoff.applied if r['anchor'] >= 0),
        latency_median_ms=None if not live else float(np.median([r['total_seconds'] for r in live]))*1000,
        latency_max_ms=None if not live else max(r['total_seconds'] for r in live)*1000,
        body_hand_synchronized=skew == 0, max_body_hand_skew_ticks=skew,
        project_token=config['rollout']['project_token'], token_chunk_size=config['model']['token_chunk_size'],
        buffer_stale_control_ticks=stale_ticks, buffer_expired_tokens=buffer.expired, buffer_overwrites=buffer.overwrites,
        buffer_empty_polls=buffer.empty_polls,
        execution_policy='Two tick slots, capture -1,1,3,...; strict abort on stale consumption' if strict else 'DIAGNOSTIC ONLY: report stale consumption and continue',
        policy_observation='current RGB and 96-D current proprio only; no previous action or history',
        startup_capture_anchor=-1, no_expert_future_used=True, encoder_bypassed=True,
        failure=failure, renderer=worker.renderer_info, provenance=provenance,
        timing_contract_pass=bool(complete and stale_ticks == 0 and buffer.expired == 0 and len(latencies) == length//2 and len(handoff.applied) == length),
        closed_loop_pass=bool(complete and failure is None and stale_ticks == 0 and skew == 0 and report['physical_grasp_success'] and report['fall_steps'] == 0))
    dump_json(output/'report.json', report)
    print({k: report[k] for k in ('episode', 'recorded_frames', 'closed_loop_pass', 'max_lift_m', 'max_continuous_hold_seconds', 'buffer_stale_control_ticks', 'failure')}, flush=True)
    return report
