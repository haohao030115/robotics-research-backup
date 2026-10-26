"""Native process orchestration and existing MuJoCo/DDS/hand controller reuse."""
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from .release import digest


class NativeTokenSonic:
    def __init__(self, config, bootstrap, output):
        self.config = config
        self.master, slave = pty.openpty()
        self.log = (output/'sonic.log').open('wb')
        self.captured = bytearray()
        s, r = config['sonic'], config['rollout']
        # No --encoder-file: this process must consume external tokens only.
        self.command = [s['binary'], r['network_interface'], s['decoder'], str(bootstrap),
            '--obs-config', s['observation_config'], '--input-type', 'zmq',
            '--zmq-host', 'localhost', '--zmq-port', str(r['zmq_port']), '--zmq-topic', 'pose',
            '--output-type', 'zmq', '--zmq-out-port', str(r['zmq_output_port']),
            '--disable-crc-check', '--logs-dir', str(output/'policy_logs'), '--enable-csv-logs']
        try:
            self.process = subprocess.Popen(self.command, cwd=Path(s['repository'])/'gear_sonic_deploy',
                stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
        except BaseException:
            os.close(self.master); self.log.close()
            raise
        finally:
            os.close(slave)

    def pump(self):
        while select.select([self.master], [], [], 0)[0]:
            try:
                data = os.read(self.master, 65536)
            except OSError:
                break
            if not data:
                break
            self.captured.extend(data); self.log.write(data); self.log.flush()

    def key(self, key):
        os.write(self.master, key.encode())

    def close(self):
        self.pump()
        if self.process.poll() is None:
            self.key('O')
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL); self.process.wait()
        self.pump(); self.log.close(); os.close(self.master)


def create_runtime(config, episode, output):
    repository = Path(config['sonic']['repository'])
    project = Path(__file__).resolve().parents[2]/'dl-group2/scripts'
    sys.path.insert(0, str(project)); sys.path.insert(0, str(repository))
    import run_sonic_grasp
    import scene
    # Explicit asset selection is needed because this MUSA checkout has a newer
    # source commit than dl-group2's pinned submodule. No upstream XML is edited.
    scene.ROBOT = Path(config['sonic']['robot_xml'])
    from expert_trajectory import load_reference
    meta = __import__('json').loads((episode.path/'metadata.json').read_text())
    ref, _ = load_reference(episode.path/'candidates'/meta['selected_candidate'])
    initial = {k: v[:2].copy() for k, v in ref.items()}
    # The native reader still needs an initial motion, even in external-token
    # mode. Only the legal initial pose is replicated; no expert future is used.
    bootstrap = output/'bootstrap'; folder = bootstrap/'initial_pose'; folder.mkdir(parents=True)
    n = len(episode.times)+1
    values = {'joint_pos': initial['body_ref_q'][0], 'joint_vel': np.zeros(29),
              'body_pos': initial['root_ref_pos'][0], 'body_quat': initial['root_ref_quat'][0],
              'body_lin_vel': np.zeros(3), 'body_ang_vel': np.zeros(3)}
    for name, row in values.items():
        np.savetxt(folder/(name+'.csv'), np.tile(row, (n, 1)), delimiter=',',
                   header=','.join(f'{name}_{i}' for i in range(len(row))), comments='')
    (folder/'metadata.txt').write_text(f'Metadata for: initial_pose\nBody part indexes:\n[0]\n\nTotal timesteps: {n}\n')
    (folder/'info.txt').write_text('VA V2 initialization only; external tokens supply body motion\nsample_rate_hz: 50\n')
    args = SimpleNamespace(robot='baseline', pelvis_support='none', physics_hz=config['rollout']['physics_hz'],
        startup_body='freebase-wbc', band_angular_damping=10., band_anchor='source-equilibrium',
        band='off', startup_support='none')
    runtime = run_sonic_grasp.TabletopDDS(initial, meta, args)
    if list(runtime.body.names) != episode.body_order or list(runtime.hand.names) != episode.hand_order:
        raise ValueError('Dataset/native joint orders differ')
    return runtime, bootstrap, run_sonic_grasp.CSVFollower


class TimestampedHand:
    def __init__(self, controller):
        self.controller = controller
        self.index = -1
        self.target = controller.q[0].copy()

    def set(self, index, target):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (14,) or not np.isfinite(target).all() or index < self.index:
            raise ValueError('Invalid/out-of-order hand target')
        self.index, self.target = index, target.copy()

    def apply(self, data, reference_time):
        self.controller.q[:] = self.target
        return self.controller.apply(data, 0.)


def runtime_provenance(config):
    return {k: {'path': config['sonic'][k], 'sha256': digest(config['sonic'][k])}
            for k in ('binary', 'robot_xml')}
