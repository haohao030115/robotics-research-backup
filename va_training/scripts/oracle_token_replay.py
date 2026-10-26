"""Replay original native tokens through the external-token release controller."""
import argparse
from collections import deque
from pathlib import Path
import sys
import time
import numpy as np
import zmq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.tokens import load_token_episodes
from sonic.release import ExternalTokenAdapter, bundle_provenance
from sonic.runtime import NativeTokenSonic, create_runtime, TimestampedHand, runtime_provenance
from utils.config import dump_json
from utils.v2_config import load_v2_config


def run_oracle(config, episode, output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    sonic = bundle_provenance(config['sonic'])
    provenance = {'va_config': config, 'sonic': sonic, 'runtime': runtime_provenance(config),
                  'dataset': episode.provenance, 'body_joint_order': episode.body_order,
                  'hand_joint_order': episode.hand_order}
    dump_json(output/'provenance.json', provenance)
    runtime, bootstrap, follower = create_runtime(config, episode, output)
    hand = TimestampedHand(runtime.hand_controller); runtime.hand_controller = hand
    adapter = ExternalTokenAdapter(config['sonic']['repository'])
    context = zmq.Context(); pub = context.socket(zmq.PUB); pub.setsockopt(zmq.LINGER, 0)
    pub.bind(f'tcp://*:{config["rollout"]["zmq_port"]}')
    native = NativeTokenSonic(config, bootstrap, output)
    tokens = follower(output/'policy_logs/token_state.csv')
    pending = deque(); consumed = []; physical = []; sync = []; errors = []; control_times = []
    frame = -1; sent = 0; initialized = enabled = started = False; failure = None
    preloaded_at = None; last_index = None
    step_dt = 1/config['rollout']['physics_hz']; next_step = time.monotonic()
    deadline = next_step+config['rollout']['timeout_seconds']
    try:
        while time.monotonic() < deadline:
            native.pump()
            if native.process.poll() is not None:
                raise RuntimeError('Native SONIC exited; inspect sonic.log')
            if not initialized and b'Init Done' in native.captured:
                native.key('\n'); initialized = True
            if initialized and not enabled and b'ZMQ STREAMING MODE: ENABLED' in native.captured:
                enabled = True; preloaded_at = time.monotonic()
            if enabled and not started:
                pub.send(adapter.message(episode.tokens[0], 0))
                if time.monotonic()-preloaded_at > .2 and b'Protocol v4: Received 64D token' in native.captured:
                    native.key(']'); started = True
            pending.extend(tokens.read())
            if len(pending) > 1:
                raise RuntimeError('Missed native control tick; do not invent intermediate physics snapshots')
            if pending:
                row = pending.popleft(); index = int(row[0])
                if last_index is not None and index != last_index+1:
                    raise RuntimeError('Non-contiguous native control indices')
                last_index = index; frame += 1
                if frame >= len(episode.tokens):
                    break
                error = float(np.max(np.abs(np.asarray(row[5:])-episode.tokens[frame])))
                errors.append(error)
                if error > 1e-6:
                    raise RuntimeError(f'Consumed token differs at frame {frame}: {error}; v4 does not schedule frame_index')
                consumed.append(frame); control_times.append(row[1]/1000)
                hand.set(frame, episode.hand_ref[frame]); sync.append((frame, hand.index))
                if frame+1 < len(episode.tokens):
                    pub.send(adapter.message(episode.tokens[frame+1], frame+1)); sent += 1
                if frame % int(config['data']['control_hz']) == 0:
                    print(f'oracle frame={frame} token_error={error:.2g}', flush=True)
            playing = frame >= 0
            hq, _, _, result = runtime.step(max(frame, 0)/config['data']['control_hz'],
                started and runtime.bridge.low_cmd_received, playing)
            if playing:
                if np.max(abs(hq-hand.target)) > 1e-8:
                    raise RuntimeError('Hand target mismatch')
                physical.append(runtime.data.time)
                if runtime.data.xpos[runtime.model.body('pelvis').id, 2] < .3:
                    raise RuntimeError('Robot fell')
            if len(consumed) == len(episode.tokens):
                # Allow the final commanded interval to finish before freezing.
                if runtime.data.time-physical[0] >= (len(consumed)-1)/config['data']['control_hz']:
                    break
            next_step += step_dt
            delay = next_step-time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if len(consumed) != len(episode.tokens):
            failure = 'Incomplete oracle token coverage'
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    finally:
        native.close(); tokens.close(); pub.close(); context.term()
    report = runtime.evaluator.report(runtime.data)
    complete = consumed == list(range(len(episode.tokens)))
    skew = max((abs(a-b) for a, b in sync), default=None)
    period = float(np.median(np.diff(control_times))) if len(control_times) > 1 else None
    report.update(episode=episode.id, token_count=len(consumed), expected_token_count=len(episode.tokens),
        token_timestamp_alignment='active archived control row t -> consumed live control tick t',
        token_alignment_pass=complete and max(errors, default=1) <= 1e-6,
        sonic_execution_hz=None if period is None else 1/period, configured_sonic_hz=config['data']['control_hz'],
        hand_synchronized=skew == 0, max_body_hand_skew_ticks=skew,
        no_fall=report['fall_steps'] == 0, lift_success=report['max_lift_m'] >= .1,
        hold_success=report['max_continuous_hold_seconds'] >= 2.,
        failure=failure, frame_coverage_complete=complete, encoder_bypassed=True,
        native_command=native.command, provenance=provenance,
        oracle_pass=bool(complete and failure is None and skew == 0 and report['physical_grasp_success'] and report['fall_steps'] == 0))
    dump_json(output/'report.json', report)
    np.save(output/'body_hand_execution_indices.npy', np.asarray(sync, dtype=np.int64))
    print({k: report[k] for k in ('episode', 'token_count', 'oracle_pass', 'max_lift_m', 'max_continuous_hold_seconds', 'failure')}, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--episode-id', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args(); config = load_v2_config(args.config)
    config['data']['episode_ids'] = [args.episode_id]
    episode = load_token_episodes(config['data'])[0]
    if not run_oracle(config, episode, args.output)['oracle_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
