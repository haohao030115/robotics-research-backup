"""Current post-step observation r -> native execution ticks r+1,r+2.

Existing images are sparse 10-Hz snapshots, not a 25-Hz stream. Each is an
independent training decision; never reuse/interpolate a stale image as current.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.episodes import image_tensor

FIELDS = ('body_q', 'body_dq', 'hand_q', 'hand_dq', 'base_quat', 'base_ang_vel', 'projected_gravity')
WIDTHS = (29, 29, 14, 14, 4, 3, 3)


def current_proprio(state):
    arrays = [np.asarray(state[k], dtype=np.float32) for k in FIELDS]
    if any(a.shape != (w,) for a, w in zip(arrays, WIDTHS)):
        raise ValueError('Expected one current row for each V3 field')
    result = np.concatenate(arrays)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite current state')
    return result


def targets(anchor):
    return np.array([anchor+1, anchor+2], dtype=np.int64)


class V3Dataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes
        self.index = [(i, j, int(r)) for i, e in enumerate(episodes) for j, r in enumerate(e.anchors) if r+2 < len(e.times)]
        # Resize once, keep uint8 to bound memory; identical path in live inference.
        self.images = []
        for e in episodes:
            with np.load(e.image_path, allow_pickle=False) as z:
                self.images.append(torch.stack([preprocess_uint8(rgb) for rgb in z['head_rgb']]))
        if episodes and not self.index:
            raise ValueError('No complete two-tick targets')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, j, r = self.index[index]
        e = self.episodes[i]; ix = targets(r)
        return dict(image=self.images[i][j].float()/255,
                    proprio=torch.from_numpy(current_proprio({k: e.state[k][r] for k in FIELDS})),
                    target_sonic_token=torch.from_numpy(e.tokens[ix].astype(np.float32)),
                    target_hand_ref=torch.from_numpy(e.hand_ref[ix].astype(np.float32)),
                    episode_id=e.id, observation_index=r, execution_indices=torch.from_numpy(ix),
                    observation_timestamp=float(e.times[r]), rgb_timestamp=float(e.rgb_times[j]),
                    proprio_timestamp=float(e.times[r]), target_timestamps=torch.from_numpy(e.times[ix].copy()))


def preprocess_uint8(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected uint8 HWC RGB')
    return (image_tensor(rgb, 128)*255).round().clamp(0, 255).to(torch.uint8)


def alignment_report(episodes):
    rows = []
    for e in episodes:
        sim = np.load(e.path/'validation'/e.provenance['selected_validation_run']/'sim_timestamps.npy')
        for j, r in enumerate(e.anchors):
            if r+2 >= len(e.times):
                continue
            ix = targets(int(r))
            rows.append(dict(episode=e.id, observation_index=int(r), decision_execution_index=int(r+1),
                observation_timestamp=float(e.times[r]), rgb_timestamp=float(e.rgb_times[j]),
                proprio_timestamp=float(e.times[r]), observation_sim_timestamp=float(sim[r]),
                slot_target_indices=ix.tolist(), slot_target_timestamps=e.times[ix].tolist(),
                actual_expert_execution_indices=e.control_metadata[ix, 0].astype(int).tolist(),
                actual_expert_execution_time_ms=e.control_metadata[ix, 1].tolist(),
                actual_expert_execution_monotonic_ms=e.control_metadata[ix, 3].tolist()))
    return dict(convention='post-step r -> pre-action native rows r+1,r+2; decision t=r+1',
        clocks='Reference timestamps are labels, not capture wall time. Native monotonic and simulation clocks are reported separately; do not subtract across clocks.',
        runtime_anchors=[-1, 1, 3, 5], runtime_first_execution_indices=[0, 2, 4, 6],
        training_rgb_hz=10, runtime_rgb_hz=25, history_length=0, control_hz=50,
        prediction_horizon=2, execution_horizon=2, stride=2, samples=len(rows), rows=rows)
