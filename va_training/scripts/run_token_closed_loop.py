"""Gated Stage C: live RGB/state -> VA V2 -> native external-token SONIC + hand PD."""
import argparse
from collections import deque
import copy
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
import numpy as np
import torch
import zmq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.tokens import load_token_episodes
from models.token_inference import TokenModelPolicy
from models.observation_history import ObservationHistory, simulator_state
from sonic.release import bundle_provenance, require_pass, digest, ExternalTokenAdapter
from sonic.runtime import create_runtime, NativeTokenSonic, TimestampedHand, runtime_provenance
from sonic.token_handoff import TokenHandoff
from utils.config import dump_json
from utils.runtime import initialize, cleanup


def synchronize(device):
    if device.type == 'musa':
        torch.musa.synchronize(device)


class LiveVisionWorker:
    def __init__(self, model, policy, project):
        self.model, self.policy, self.project = model, policy, project
        self.requests = queue.Queue(maxsize=1); self.results = queue.Queue()
        self.stop = threading.Event(); self.ready = threading.Event(); self.error = None
        self.renderer_info = {}; self.dropped_requests = 0
        self.thread = threading.Thread(target=self.work, daemon=True); self.thread.start()

    def submit(self, anchor, data, state, mask, target_offset):
        request = {'anchor': anchor, 'execution_index': anchor+target_offset, 'qpos': data.qpos.copy(),
                   'qvel': data.qvel.copy(), 'state': state, 'mask': mask, 'requested': time.monotonic()}
        try:
            self.requests.put_nowait(request)
        except queue.Full:
            try:
                self.requests.get_nowait(); self.dropped_requests += 1
            except queue.Empty:
                pass
            self.requests.put_nowait(request)

    def work(self):
        renderer = None
        try:
            import mujoco
            from OpenGL import GL
            data = mujoco.MjData(self.model)
            renderer = mujoco.Renderer(self.model, height=240, width=320)
            self.renderer_info = {k: GL.glGetString(v).decode() for k, v in
                [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}
            self.ready.set()
            device = self.policy.policy.image_mean.device
            warm = False
            while not self.stop.is_set():
                try:
                    request = self.requests.get(timeout=.1)
                except queue.Empty:
                    continue
                start = time.monotonic(); data.qpos[:] = request.pop('qpos'); data.qvel[:] = request.pop('qvel')
                mujoco.mj_forward(self.model, data); renderer.update_scene(data, camera='head_rgb')
                rgb = renderer.render().copy(); rendered = time.monotonic()
                if not warm:
                    for _ in range(3):
                        self.policy.predict(rgb, request['state'], request['mask'], self.project)
                    synchronize(device); warm = True
                infer_start = time.monotonic()
                output = self.policy.predict(rgb, request.pop('state'), request.pop('mask'), self.project)
                synchronize(device); finish = time.monotonic()
                request.update(token=output['sonic_token'].cpu().numpy().copy(),
                    hand=output['hand_ref'].cpu().numpy().copy(), finished=finish,
                    render_seconds=rendered-start, inference_seconds=finish-infer_start,
                    total_seconds=finish-request['requested'])
                self.results.put(request)
        except Exception:
            self.error = traceback.format_exc(); self.ready.set()
        finally:
            if renderer is not None:
                renderer.close()

    def close(self):
        self.stop.set(); self.thread.join(timeout=30)


def run(config, checkpoint_path, episode, output, gates):
    # Gate failures occur before process launch or physics stepping.
    sonic = bundle_provenance(config['sonic'])
    oracle = require_pass(gates['oracle'], 'oracle_pass', sonic)
    overfit = require_pass(gates['overfit'], 'overfit_pass', sonic)
    offline = require_pass(gates['offline'], 'offline_evaluation_pass', sonic)
    if digest(checkpoint_path) != offline['checkpoint_sha256']:
        raise ValueError('Closed loop checkpoint differs from the Stage B evaluated checkpoint')
    if config['model']['token_chunk_size'] != 1 or config['data']['target_offset'] != 1:
        raise ValueError('Initial Stage C scheduler supports K=1 and next-control targets only')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    policy, saved = TokenModelPolicy.from_checkpoint(checkpoint_path, config['training']['device'])
    if saved['provenance']['sonic'] != sonic:
        raise ValueError('Checkpoint SONIC provenance differs')
    provenance = {'va_config': config, 'sonic': sonic, 'runtime': runtime_provenance(config),
        'dataset': episode.provenance, 'body_joint_order': episode.body_order, 'hand_joint_order': episode.hand_order,
        'checkpoint': str(Path(checkpoint_path).resolve()), 'checkpoint_sha256': digest(checkpoint_path),
        'gates': {k: {'path': str(Path(p).resolve()), 'sha256': digest(p)} for k, p in gates.items()}}
    dump_json(output/'provenance.json', provenance)
    runtime, bootstrap, follower = create_runtime(config, episode, output)
    hand = TimestampedHand(runtime.hand_controller); runtime.hand_controller = hand
    history = ObservationHistory(config['model']['history_length'])
    # Startup uses a live initial snapshot, with capture anchor -1 and first
    # execution tick 0; after that all snapshots are post-step control snapshots.
    history.append(-1, simulator_state(runtime, np.zeros(29)))
    worker = LiveVisionWorker(runtime.model, policy, config['rollout']['project_token'])
    native = None; context = None; pub = None; token_log = action_log = None
    records = []; sync = []; frames = []; latencies = []; requests = []; failure = None
    control_times = []; previous_action = np.zeros(29); frame = -1
    handoff = TokenHandoff(); step_dt = 1/config['rollout']['physics_hz']
    visual_stride = int(config['data']['control_hz']/config['rollout']['visual_hz'])
    length = len(episode.times)
    try:
        if not worker.ready.wait(timeout=30) or worker.error:
            raise RuntimeError(worker.error or 'Renderer initialization timed out')
        state, mask = history.snapshot(); worker.submit(-1, runtime.data, state, mask, 1)
        initial = worker.results.get(timeout=60)
        handoff.propose(initial); latencies.append(initial)
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
                enabled = True; preload_start = time.monotonic()
            if enabled and not started:
                pub.send(adapter.message(initial['token'], 0))
                if time.monotonic()-preload_start > .2 and b'Protocol v4: Received 64D token' in native.captured:
                    native.key(']'); started = True
            while not worker.results.empty():
                result = worker.results.get_nowait(); latencies.append(result)
                if handoff.pending is not None:
                    result['discard_reason'] = 'native handoff still pending'
                    continue
                # A late prediction is applied on a later observed native tick;
                # it is never relabeled as an on-time next-tick prediction.
                handoff.propose(result)
                pub.send(adapter.message(result['token'], result['execution_index']))
            token_pending.extend(token_log.read()); action_pending.extend(action_log.read())
            if len(token_pending) > 1 or len(action_pending) > 1:
                raise RuntimeError('Missed control snapshot; abort rather than fabricate history')
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
                hand.set(frame, current_hand); sync.append((frame, hand.index))
                previous_action = np.asarray(action_row[5:]); control_times.append(token_row[1]/1000)
                new = frame; frames.append(frame)
            applied_hand, _, _, physical = runtime.step(max(frame, 0)/config['data']['control_hz'],
                started and runtime.bridge.low_cmd_received, frame >= 0)
            if frame >= 0 and not np.allclose(applied_hand, hand.target, atol=1e-8, rtol=0):
                raise RuntimeError('Independent hand controller did not apply the acknowledged target')
            if new is not None:
                current_state = simulator_state(runtime, previous_action)
                history.append(frame, current_state)
                records.append({'frame': frame, 'sim_time': runtime.data.time, 'state': current_state,
                                'token': np.asarray(token_row[5:]), 'hand': hand.target.copy()})
                if frame % visual_stride == 0 and frame+1 < length:
                    state, mask = history.snapshot(); worker.submit(frame, runtime.data, state, mask, 1)
                    requests.append(frame)
                if frame % int(config['data']['control_hz']) == 0:
                    print(f'V2 frame={frame} lift={physical["lift_m"]:.4f} hold={runtime.evaluator.hold:.3f}', flush=True)
            if frame >= 0 and runtime.data.xpos[runtime.model.body('pelvis').id, 2] < .3:
                raise RuntimeError('Robot fell')
            next_step += step_dt; delay = next_step-time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if frames != list(range(length)):
            failure = 'Incomplete control coverage'
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    finally:
        if native is not None:
            native.close()
        worker.close()
        for log in (token_log, action_log):
            if log is not None:
                log.close()
        if pub is not None:
            pub.close()
        if context is not None:
            context.term()
    if records:
        np.savez_compressed(output/'live_observations.npz', reference_frame=frames,
            sim_timestamps=[r['sim_time'] for r in records], token=[r['token'] for r in records],
            hand_ref=[r['hand'] for r in records], **{k: [r['state'][k] for r in records] for k in records[0]['state']})
    # JSON timing records exclude NumPy prediction arrays, saved separately.
    if latencies:
        np.savez_compressed(output/'predictions.npz', capture_anchor=[r['anchor'] for r in latencies],
            intended_execution_index=[r['execution_index'] for r in latencies],
            token=[r['token'] for r in latencies], hand_ref=[r['hand'] for r in latencies])
    dump_json(output/'handoffs.json', [{k: v for k, v in r.items() if k not in ('token', 'hand')}
                                     for r in handoff.applied])
    report = runtime.evaluator.report(runtime.data); complete = frames == list(range(length))
    live = [r for r in latencies if r['anchor'] >= 0]
    period = float(np.median(np.diff(control_times))) if len(control_times) > 1 else None
    skew = max((abs(a-b) for a, b in sync), default=None)
    report.update(episode=episode.id, configured_visual_hz=config['rollout']['visual_hz'],
        configured_control_hz=config['data']['control_hz'], actual_control_hz=None if period is None else 1/period,
        recorded_frames=len(frames), expected_frames=length, frame_coverage_complete=complete,
        prediction_count=len(live), dropped_visual_requests=worker.dropped_requests,
        visual_deadline_misses=sum(r['total_seconds'] > 1/config['rollout']['visual_hz'] for r in live),
        execution_deadline_misses=sum(r['applied_index'] > r['execution_index'] for r in handoff.applied if r['anchor'] >= 0),
        latency_median_ms=None if not live else float(np.median([r['total_seconds'] for r in live]))*1000,
        latency_max_ms=None if not live else max(r['total_seconds'] for r in live)*1000,
        body_hand_synchronized=skew == 0, max_body_hand_skew_ticks=skew,
        project_token=config['rollout']['project_token'], token_chunk_size=1,
        execution_policy='10 Hz visual decisions; latest acknowledged token/hand pair held at 50 Hz',
        previous_action_semantics='native pre-action previous-action log at post-step capture anchor',
        startup_capture_anchor=-1, no_expert_future_used=True, encoder_bypassed=True,
        failure=failure, renderer=worker.renderer_info, provenance=provenance,
        closed_loop_pass=bool(complete and failure is None and skew == 0 and report['physical_grasp_success'] and report['fall_steps'] == 0))
    dump_json(output/'report.json', report)
    print({k: report[k] for k in ('episode', 'recorded_frames', 'closed_loop_pass', 'max_lift_m', 'max_continuous_hold_seconds', 'failure')}, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--episode-id', required=True)
    p.add_argument('--oracle-report', required=True); p.add_argument('--overfit-report', required=True)
    p.add_argument('--offline-report', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'musa'], default='musa')
    p.add_argument('--project-token', action='store_true')
    args = p.parse_args(); runtime = initialize(args.device, 38)
    try:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        config = copy.deepcopy(saved['config']); config['training']['device'] = args.device
        config['rollout']['project_token'] = args.project_token
        data_config = copy.deepcopy(config['data']); data_config['episode_ids'] = [args.episode_id]
        episode = load_token_episodes(data_config)[0]
        report = run(config, args.checkpoint, episode, args.output,
            {'oracle': args.oracle_report, 'overfit': args.overfit_report, 'offline': args.offline_report})
        if not report['closed_loop_pass']:
            raise SystemExit(2)
    finally:
        cleanup(runtime)


if __name__ == '__main__':
    main()
