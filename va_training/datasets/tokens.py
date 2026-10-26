"""Native logged-token labels with explicit post-step -> next-control alignment."""
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .episodes import image_tensor
from sonic.release import digest

STATE_WIDTHS = {'body_q': 29, 'body_dq': 29, 'hand_q': 14, 'hand_dq': 14,
                'base_quat': 4, 'base_ang_vel': 3, 'previous_action': 29,
                'projected_gravity': 3}
METADATA_COLUMNS = ['index', 'time_ms', 'time_realtime_ms', 'time_monotonic_ms', 'ros_timestamp']


def read_log(path, columns):
    path = Path(path)
    with path.open() as stream:
        header = next(csv.reader(stream))
    if header[:5] != METADATA_COLUMNS or len(set(header)) != len(header):
        raise ValueError(f'{path}: unexpected/duplicate metadata columns')
    indexes = [header.index(c) for c in columns]
    values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if values.shape[1] != len(header) or not np.isfinite(values).all():
        raise ValueError(f'{path}: invalid log values')
    if not np.array_equal(values[:, 0], np.arange(len(values))):
        raise ValueError(f'{path}: missing/non-contiguous control indices')
    if len(values) > 1 and not (np.diff(values[:, 1]) > 0).all():
        raise ValueError(f'{path}: non-monotonic control timestamps')
    return values[:, :5], values[:, indexes]


def projected_gravity(quat):
    quat = np.asarray(quat)
    if quat.shape[-1] != 4 or not np.isfinite(quat).all():
        raise ValueError('Expected wxyz quaternion')
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if not np.allclose(norm, 1, atol=1e-5):
        raise ValueError('Base quaternion is not unit length')
    w, x, y, z = np.moveaxis(quat / norm, -1, 0)
    return np.stack([2*(w*y-x*z), -2*(w*x+y*z), -1+2*(x*x+y*y)], axis=-1)


@dataclass
class TokenEpisode:
    id: str
    source_episode_id: str
    split: str | None
    path: Path
    image_path: Path
    times: np.ndarray
    rgb_times: np.ndarray
    anchors: np.ndarray
    state: dict
    tokens: np.ndarray
    hand_ref: np.ndarray
    control_metadata: np.ndarray
    body_order: list
    hand_order: list
    provenance: dict


def load_token_episodes(config):
    episodes, seen_ids, seen_paths = [], set(), set()
    for manifest in config['manifests']:
        manifest = Path(manifest).resolve()
        entries = json.loads(manifest.read_text())['episodes']
        for entry in entries:
            ident = entry.get('id', entry['episode_id'])
            if config.get('episode_ids') and ident not in config['episode_ids']:
                continue
            p = (manifest.parent / entry['episode_path']).resolve()
            if ident in seen_ids or p in seen_paths:
                raise ValueError('Duplicate episode ID/path: possible split leakage')
            seen_ids.add(ident); seen_paths.add(p)
            meta = json.loads((p/'metadata.json').read_text())
            if entry.get('expert_valid') is not True or meta.get('expert_valid') is not True:
                raise ValueError(f'{ident}: unvalidated expert')
            if meta.get('quaternion_order') != 'wxyz':
                raise ValueError(f'{ident}: expected wxyz')
            t = np.load(p/'timestamps.npy', allow_pickle=False)
            n = len(t)
            if t.shape != (n,) or n < 2 or not np.isfinite(t).all() or not np.allclose(np.diff(t), 1/config['control_hz'], atol=1e-8, rtol=0):
                raise ValueError(f'{ident}: invalid reference clock')
            def array(name, width):
                a = np.load(p/(name+'.npy'), allow_pickle=False)
                if a.shape != (n, width) or not np.isfinite(a).all():
                    raise ValueError(f'{ident}: invalid {name}')
                return a
            body_order, hand_order = meta['body_joint_order'], meta['hand_joint_order']
            if len(set(body_order)) != 29 or len(set(hand_order)) != 14 or set(body_order) & set(hand_order):
                raise ValueError('Invalid joint order')
            if episodes and (body_order != episodes[0].body_order or hand_order != episodes[0].hand_order):
                raise ValueError('Episode joint orders differ')
            state = {k: array(k, w) for k, w in STATE_WIDTHS.items() if k in ('body_q', 'body_dq', 'hand_q', 'hand_dq')}
            state['base_quat'] = array('root_quat', 4)
            # Recorded root_ang_vel is WORLD-frame. Rotate it into the base frame,
            # matching the local angular-velocity convention used by live MuJoCo.
            world = array('root_ang_vel', 3)
            q = state['base_quat']; w, v = q[:, :1], q[:, 1:]
            state['base_ang_vel'] = world - 2*w*np.cross(v, world) + 2*np.cross(v, np.cross(v, world))
            state['projected_gravity'] = projected_gravity(q)
            run = p/'validation'/meta['selected_validation_run']
            frame = np.load(run/'reference_frame.npy', allow_pickle=False)
            if not np.array_equal(frame, np.arange(n)) or not np.array_equal(np.load(run/'timestamps.npy'), t):
                raise ValueError('Selected replay has incomplete/reordered reference coverage')
            for key in ('body_q', 'body_dq', 'hand_q', 'hand_dq', 'root_quat'):
                if not np.array_equal(np.load(p/(key+'.npy')), np.load(run/(key+'.npy'))):
                    raise ValueError('Published state differs from selected replay')
            logs = run/'policy_logs'
            pm, play = read_log(logs/'motion_playing.csv', ['playing_0'])
            tm, token = read_log(logs/'token_state.csv', [f'token_{i}' for i in range(64)])
            am, action = read_log(logs/'action.csv', [f'act_{i}' for i in range(29)])
            mm, mode = read_log(logs/'encoder_mode.csv', ['encoder_mode_0'])
            if not (np.array_equal(pm, tm) and np.array_equal(pm, am) and np.array_equal(pm, mm)):
                raise ValueError('Native token/action/play/mode control metadata disagree')
            active = np.flatnonzero(play[:, 0] > .5)
            if len(active) != n or not np.array_equal(active, np.arange(active[0], active[0]+n)):
                raise ValueError('Native active control rows do not map 1:1 to reference frames')
            if not np.all(mode[active] == 0):
                raise ValueError('Expert tokens were not produced in release G1 mode')
            token = token[active].astype(np.float32)
            if not np.allclose(token*16, np.rint(token*16), atol=1e-6) or token.min() < -1 or token.max() > 15/16:
                raise ValueError('Native token does not lie on release FSQ grid')
            # Native action.csv[t] is the PREVIOUS action captured before control t.
            # This exact observable is retained at post-step anchor t, avoiding
            # a future-log-row dependency in the VA observation.
            state['previous_action'] = action[active]
            image_path = (manifest.parent/entry['rgb_path']).resolve()
            with np.load(image_path, allow_pickle=False) as rgb:
                anchors = rgb['reference_frame'].copy()
                rgb_times = rgb['timestamps'].copy()
                if anchors.ndim != 1 or not np.issubdtype(anchors.dtype, np.integer) or len(anchors) == 0 or not (np.diff(anchors) > 0).all() or anchors[0] < 0 or anchors[-1] >= n:
                    raise ValueError('Invalid RGB/reference indices')
                if not np.array_equal(anchors, entry['rgb_reference_frame_indices']) or not np.array_equal(rgb_times, t[anchors]):
                    raise ValueError('RGB/reference mapping or timestamps disagree')
                if not np.array_equal(rgb['sim_timestamps'], np.load(run/'sim_timestamps.npy')[anchors]):
                    raise ValueError('RGB and measured state simulation clocks disagree')
                images = rgb['head_rgb']
                if images.shape != (len(anchors), 240, 320, 3) or images.dtype != np.uint8:
                    raise ValueError('Expected native uint8 head RGB')
            for name, width in [('body_ref_q', 29), ('body_ref_dq', 29), ('hand_ref_dq', 14), ('root_ref_quat', 4)]:
                array(name, width)
            hand_ref = array('hand_ref_q', 14)
            if not np.array_equal(np.load(run/'hand_applied_ref_q.npy'), hand_ref):
                raise ValueError('Expert body/hand execution is not synchronized')
            prov = {'manifest': str(manifest), 'manifest_sha256': digest(manifest),
                    'token_label_source': str(logs/'token_state.csv'), 'token_sha256': digest(logs/'token_state.csv'),
                    'selected_validation_run': meta['selected_validation_run'],
                    'first_active_control_index': int(pm[active[0], 0]), 'inactive_rows_removed': len(play)-n}
            episodes.append(TokenEpisode(ident, entry.get('source_episode_id', ident), entry.get('split'),
                p, image_path, t, rgb_times, anchors, state, token, hand_ref, tm[active], body_order, hand_order, prov))
    if not episodes or (config.get('episode_ids') and set(config['episode_ids']) != {e.id for e in episodes}):
        raise ValueError('No episodes or requested episode IDs not found')
    return episodes


class TokenDataset(Dataset):
    def __init__(self, episodes, image_size=128, history_length=3, token_chunk_size=1, target_offset=1):
        if history_length < 1 or token_chunk_size < 1 or target_offset < 1:
            raise ValueError('Positive history/K and causal target_offset>=1 required')
        self.episodes, self.image_size = episodes, image_size
        self.history_length, self.k, self.target_offset = history_length, token_chunk_size, target_offset
        self.index = [(i, j, int(t)) for i, e in enumerate(episodes) for j, t in enumerate(e.anchors)
                      if t+target_offset+token_chunk_size <= len(e.times)]
        self._cache_id, self._cache = None, None
        if episodes and not self.index:
            raise ValueError('No complete causal token targets')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, j, t = self.index[index]; e = self.episodes[i]
        if self._cache_id != i:
            with np.load(e.image_path, allow_pickle=False) as z:
                self._cache = z['head_rgb'].copy()
            self._cache_id = i
        raw = np.arange(t-self.history_length+1, t+1)
        history = np.maximum(raw, 0)
        target = np.arange(t+self.target_offset, t+self.target_offset+self.k)
        token, hand = e.tokens[target], e.hand_ref[target]
        if self.k == 1:
            token, hand = token[0], hand[0]
        return {'image': image_tensor(self._cache[j], self.image_size),
                'state': {k: torch.from_numpy(np.asarray(v[history], dtype=np.float32).copy()) for k, v in e.state.items()},
                'history_mask': torch.from_numpy(raw >= 0),
                'target_sonic_token': torch.from_numpy(token.copy()),
                'target_hand_ref': torch.from_numpy(np.asarray(hand, dtype=np.float32).copy()),
                'episode_id': e.id, 'source_episode_id': e.source_episode_id,
                'state_index': t, 'rgb_index': j, 'execution_index': int(target[0]),
                'observation_timestamp': float(e.times[t]), 'execution_timestamp': float(e.times[target[0]])}


def validation_report(episodes, dataset):
    return {'validation_pass': True, 'alignment': 'post_step_anchor_t -> native_control_t+target_offset',
            'target_offset': dataset.target_offset, 'token_chunk_size': dataset.k,
            'history_length': dataset.history_length, 'history_padding': 'repeat first row with invalid mask; no cross-episode history',
            'previous_action_semantics': 'native pre-action logger previous_action at anchor t',
            'episode_boundaries_checked': True, 'samples': len(dataset),
            'episodes': [{'id': e.id, 'source_episode_id': e.source_episode_id, 'rgb_frames': len(e.anchors),
                'state_frames': len(e.times), 'reference_frames': len(e.hand_ref), 'token_frames': len(e.tokens),
                'token_shape': list(e.tokens.shape), 'finite': True, 'rgb_mapping_pass': True,
                'control_mapping_pass': True, 'state_reference_period_seconds': float(np.median(np.diff(e.times))),
                'native_control_period_seconds': float(np.median(np.diff(e.control_metadata[:, 1]))/1000),
                'body_joint_order': e.body_order, 'hand_joint_order': e.hand_order,
                **e.provenance} for e in episodes]}
