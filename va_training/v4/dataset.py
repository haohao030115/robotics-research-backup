"""Read canonical IDs as authored; never search or infer target offsets."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gravity(quat):
    w, v = quat[..., :1], -quat[..., 1:]
    z = np.array([0., 0., -1.])
    return z * (2*w*w-1) + 2*w*np.cross(v, z) + 2*v*(v*z).sum(-1, keepdims=True)


def progress_labels(ref, names):
    """Projection onto source's initial-to-maximum right-hand reference displacement."""
    initial = ref[0]
    right = np.array([n.startswith('right_') for n in names])
    peak = np.argmax(np.square((ref-initial)*right).sum(-1))
    delta = ref[peak]-initial
    moving = np.array([n.startswith('right_') for n in names]) & (np.abs(delta) > .05)
    if not moving.any():
        raise ValueError('No moving right-hand reference joints for closure labels')
    direction = delta * moving
    progress = np.clip(((ref-initial)*direction).sum(-1)/(direction@direction), 0., 1.)
    phase = np.where(progress <= .05, 0, np.where(progress >= .95, 2, 1))
    return progress, phase, moving, initial, direction


class CanonicalV4:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        m = read(self.manifest)
        if len(m['episodes']) != 1 or not m['episodes'][0]['expert_valid']:
            raise ValueError('V4 overfit requires exactly one valid canonical episode')
        self.root = self.manifest.parent
        self.folder = self.root/m['episodes'][0]['path']
        self.metadata = read(self.folder/'metadata.json')
        self.source = read(self.root/'source_reference/metadata.json')
        self.runtime = read(self.root/'canonical_replay/metadata.json')
        self.mapping = [json.loads(x) for x in (self.folder/'sample_mapping.jsonl').read_text().splitlines() if x]
        self.schema = {}
        def load(path):
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            self.schema[str(path.relative_to(self.root))] = {
                k: dict(shape=list(a.shape), dtype=str(a.dtype), finite=bool(np.isfinite(a).all()) if a.dtype.kind in 'fci' else None)
                for k, a in arrays.items()}
            if any(v['finite'] is False for v in self.schema[str(path.relative_to(self.root))].values()):
                raise ValueError('Nonfinite canonical array: '+str(path))
            return arrays
        a = load(self.folder/'samples.npz')
        enc = load(self.root/'canonical_replay/sonic_encoder.npz')
        refs = load(self.root/'canonical_replay/references.npz')
        obs = load(self.root/'canonical_replay/observations.npz')
        self.hand_names = self.source['hand_joint_names']
        self.body_names = self.source['body_joint_names']
        if self.source.get('root_quat_order', self.source.get('quaternion_convention')) != 'wxyz':
            raise ValueError('Explicit wxyz quaternion convention required')
        if self.metadata['control_hz'] != 50:
            raise ValueError('50 Hz required')
        def index(values):
            if len(set(values.tolist())) != len(values):
                raise ValueError('Nonunique canonical IDs')
            return {v: i for i, v in enumerate(values)}
        ti, hi, oi = index(enc['token_id']), index(refs['hand_ref_id']), index(obs['observation_id'])
        self.progress_all, self.phase_all, self.moving, self.initial, self.direction = progress_labels(refs['hand_ref_q'], self.hand_names)
        self.refs = refs
        body, hand, rgb, bt, ht, phases, progress, keep, starts = [], [], [], [], [], [], [], [], []
        self.rgb_sizes = set()
        for i, row in enumerate(self.mapping):
            if str(a['sample_id'][i]) != row['sample_id']:
                raise ValueError('Sample identity mismatch')
            t, h = ti[row['body_token_ids'][0]], hi[row['hand_ref_ids'][0]]
            if [ti[x] for x in row['body_token_ids']] != list(range(t, t+5)) or [hi[x] for x in row['hand_ref_ids']] != list(range(h, h+5)):
                raise ValueError('Canonical target IDs are not contiguous')
            if not np.array_equal(a['body_target'][i], enc['current_post_fsq_64d_token'][t:t+5]) or not np.array_equal(a['hand_target'][i], refs['hand_ref_q'][h:h+5]):
                raise ValueError('Canonical stored target prefix mismatch')
            if t+10 > len(enc['token_id']) or h+50 > len(refs['hand_ref_id']):
                continue
            q = a['state_history_root_quat'][i, -1]
            body.append(np.concatenate([a['state_history_body_q'][i,-1], a['state_history_body_dq'][i,-1], q,
                                        a['state_history_root_velocity'][i,-1,3:], gravity(q)]))
            hand.append(np.concatenate([a['state_history_hand_q'][i,-1], a['state_history_hand_dq'][i,-1]]))
            with Image.open(self.folder/row['rgb_path']) as image:
                self.rgb_sizes.add(image.size)
                rgb.append(np.asarray(image.convert('RGB').resize((128,128), Image.Resampling.BILINEAR)).transpose(2,0,1)/255.)
            bt.append(enc['current_post_fsq_64d_token'][t:t+10])
            ht.append(refs['hand_ref_q'][h:h+50])
            current = oi[row['observation_id']]
            if refs['observation_id'][current] != row['observation_id']:
                raise ValueError('Auxiliary current-reference identity mismatch')
            phases.append(self.phase_all[current]); progress.append(self.progress_all[current])
            keep.append(i); starts.append(h)
        self.keep, self.starts = np.array(keep), np.array(starts)
        self.data = {k: torch.tensor(np.asarray(v), dtype=torch.long if k == 'phase' else torch.float32)
                     for k,v in dict(rgb=rgb, body=body, hand=hand, body_tokens=bt, hand_ref=ht, phase=phases, progress=progress).items()}
        if not len(keep):
            raise ValueError('No complete V4 horizons')
        self.provenance = {str(p.relative_to(self.root)): sha(p) for p in
            [self.manifest, self.folder/'samples.npz', self.folder/'sample_mapping.jsonl', self.root/'canonical_replay/sonic_encoder.npz', self.root/'canonical_replay/references.npz']}

    def report(self):
        return dict(manifest=str(self.manifest), arrays=self.schema, current_observation_samples=len(self.mapping),
            usable_v4_samples=len(self.keep), dropped_tail_samples=len(self.mapping)-len(self.keep),
            rgb_resolution=[list(s) for s in sorted(self.rgb_sizes)], model_rgb_resolution=[128,128], control_hz=50,
            token_shape=list(self.data['body_tokens'].shape), hand_reference_shape=list(self.data['hand_ref'].shape),
            body_joint_order=self.body_names, hand_joint_order=self.hand_names, sonic_variant='policy/release native post-FSQ',
            finite_checks=True, target_horizons_available=True, target_padding=False,
            inputs='Only last/current state entry; root_velocity[3:6] angular velocity; gravity R(wxyz)^T [0,0,-1]. No history input.',
            target_extension='Existing body_token_ids / hand_ref_ids select start; extend same contiguous canonical arrays, exact stored five-slot prefix verified.',
            phase_definition='Current canonical hand reference projected onto initial-to-maximum-norm moving right-joint displacement (abs peak-initial > .05 rad), clipped [0,1]. OPEN <= .05; HOLD >= .95; otherwise CLOSING. Derived reference label, not physical task-phase ground truth.',
            provenance=self.provenance)
