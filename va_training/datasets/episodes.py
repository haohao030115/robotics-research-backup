"""Explicit NPZ schema adapter. No inferred keys, clocks or reference semantics."""
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class Episode:
    id: str
    q: np.ndarray
    ref: np.ndarray
    state_times: np.ndarray
    image_path: str
    image_key: str
    rgb_times: np.ndarray
    anchors: np.ndarray
    instruction: str
    split: str | None
    sources: tuple
    source_episode_id: str


def _clock(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or len(value) == 0 or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected finite nonempty timestamps in seconds')
    if not (np.diff(value) > 0).all():
        raise ValueError(f'{name}: timestamps must strictly increase')
    return value


def _matrix(value, length, width, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (length, width) or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected finite [{length},{width}], got {value.shape}')
    return value


def load_episodes(config):
    path = Path(config['manifest']).resolve()
    manifest = json.loads(path.read_text())
    records = manifest['episodes'] if isinstance(manifest, dict) else manifest
    if not isinstance(records, list):
        raise ValueError('training_manifest.json must contain an episodes list')
    keys = config['keys']
    if config.get('timestamp_unit') != 'seconds':
        raise ValueError('timestamp_unit must explicitly be seconds; convert other clocks first')
    episodes, ids, sources = [], set(), set()
    for entry in records:
        def source(canonical, legacy):
            if canonical in entry and legacy in entry and entry[canonical] != entry[legacy]:
                raise ValueError(f'conflicting {canonical}/{legacy} manifest paths')
            value = entry.get(canonical, entry.get(legacy))
            if not isinstance(value, str) or not value:
                raise ValueError(f'manifest requires {canonical}')
            return value
        episode_source = source('episode_path', 'states')
        rgb_source = source('rgb_path', 'images')
        ident = str(entry.get('id', episode_source))
        if ident in ids:
            raise ValueError(f'duplicate episode id: {ident}')
        ids.add(ident)
        state_path = (path.parent / episode_source).resolve()
        if state_path.is_dir():
            filename = config.get('episode_state_filename')
            if not filename:
                raise ValueError('directory episode_path requires explicit episode_state_filename')
            state_path = (state_path / filename).resolve()
        image_path = (path.parent / rgb_source).resolve()
        current_sources = set((state_path, image_path))
        if current_sources & sources:
            raise ValueError('episode files reused under different IDs; possible split leakage')
        sources.update(current_sources)
        with np.load(state_path, allow_pickle=False) as data:
            times = _clock(data[keys['state_timestamps']], ident + ':state')
            n = len(times)
            q = np.concatenate((_matrix(data[keys['body_q']], n, 29, 'body_q'),
                                _matrix(data[keys['hand_q']], n, 14, 'hand_q')), axis=1)
            ref = np.concatenate((_matrix(data[keys['body_ref_q']], n, 29, 'body_ref_q'),
                                  _matrix(data[keys['hand_ref_q']], n, 14, 'hand_ref_q')), axis=1)
            for field, dim in [('body_dq', 29), ('hand_dq', 14)]:
                if keys.get(field):
                    _matrix(data[keys[field]], n, dim, field)
        dt = 1 / config['control_hz']
        if not np.allclose(np.diff(times), dt, rtol=0,
                           atol=config['state_period_tolerance_seconds']):
            raise ValueError(f'{ident}: state timestamps do not match control_hz')
        with np.load(image_path, allow_pickle=False) as data:
            inline_mapping = entry.get('rgb_reference_frame_indices')
            mapping_key = keys['rgb_state_indices']
            file_mapping = data[mapping_key] if mapping_key in data else None
            if inline_mapping is not None and file_mapping is not None:
                if not np.array_equal(np.asarray(inline_mapping), file_mapping):
                    raise ValueError('manifest and images.npz reference-frame mappings disagree')
            mapping = np.asarray(inline_mapping) if inline_mapping is not None else file_mapping
            rgb_times = _clock(data[keys['rgb_timestamps']], ident + ':rgb')
            rgb_times = rgb_times + config.get('rgb_time_offset_seconds', 0.0)
            rgb = np.asarray(data[keys['head_rgb']]).copy()
            if rgb.ndim != 4 or rgb.shape[0] != len(rgb_times) or rgb.shape[-1] != 3:
                raise ValueError(f'{ident}: head RGB must be [N,height,width,3]')
            if rgb.dtype != np.uint8 and (not np.issubdtype(rgb.dtype, np.floating)
                                         or not np.isfinite(rgb).all()
                                         or rgb.min() < 0 or rgb.max() > 1):
                raise ValueError(f'{ident}: RGB must be uint8 or finite float in [0,1]')
            alignment = config['alignment']
            if alignment == 'reference_frame_mapping':
                if config.get('mapping_semantics') != 'rgb_capture_state_index':
                    raise ValueError('mapping semantics must explicitly be rgb_capture_state_index')
                if mapping is None:
                    raise ValueError('rgb_reference_frame_indices mapping is required')
                mapping = np.asarray(mapping)
                if mapping.shape != rgb_times.shape or not np.issubdtype(mapping.dtype, np.integer):
                    raise ValueError('mapping must contain one integer state index per RGB frame')
                anchors = mapping.astype(np.int64)
                if (anchors < 0).any() or (anchors >= n).any():
                    raise ValueError('reference-frame mapping out of range')
            elif alignment == 'nearest_timestamp':
                right = np.searchsorted(times, rgb_times).clip(0, n - 1)
                left = (right - 1).clip(0, n - 1)
                anchors = np.where(abs(times[left] - rgb_times) <= abs(times[right] - rgb_times), left, right)
            else:
                raise ValueError('unknown alignment mode')
        errors = abs(times[anchors] - rgb_times)
        if (errors > config['alignment_tolerance_seconds']).any():
            raise ValueError(f'{ident}: RGB/state timestamp mismatch max={errors.max():.6f}s')
        if (np.diff(anchors) < 0).any():
            raise ValueError(f'{ident}: mapping reverses time')
        split = entry.get('split')
        if split not in (None, 'train', 'val'):
            raise ValueError('split must be train or val')
        episodes.append(Episode(ident, q, ref, times, str(image_path), keys['head_rgb'], rgb_times, anchors,
                                entry.get('instruction', ''), split,
                                (str(state_path), str(image_path)), str(entry.get('source_episode_id', ident))))
    if not episodes:
        raise ValueError('manifest contains no episodes')
    return episodes


def split_episodes(episodes, config):
    explicit = [e.split is not None for e in episodes]
    if any(explicit) and not all(explicit):
        raise ValueError('provide splits for every episode or none')
    if all(explicit):
        train = [e for e in episodes if e.split == 'train']
        val = [e for e in episodes if e.split == 'val']
    else:
        seed = config['split_seed']
        groups = sorted({e.source_episode_id for e in episodes},
                        key=lambda group: hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest())
        ratio = config['val_fraction']
        if not 0 <= ratio < 1:
            raise ValueError('val_fraction must be in [0,1)')
        n_val = min(len(groups)-1, max(1, round(len(groups)*ratio))) if ratio > 0 and len(groups)>1 else 0
        val_groups = set(groups[:n_val])
        val = [e for e in episodes if e.source_episode_id in val_groups]
        train = [e for e in episodes if e.source_episode_id not in val_groups]
    if not train:
        raise ValueError('no training episodes')
    assert not ({e.id for e in train} & {e.id for e in val})
    if {e.source_episode_id for e in train} & {e.source_episode_id for e in val}:
        raise ValueError('source episode appears in both train and val; frame leakage')
    return train, val


def image_tensor(rgb, size):
    value = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float()
    if rgb.dtype == np.uint8:
        value = value / 255.0
    return F.interpolate(value[None], size=(size, size), mode='bilinear', align_corners=False)[0]


class EpisodeDataset(Dataset):
    def __init__(self, episodes, horizon, image_size, rgb_cache_episodes=1):
        self.episodes, self.horizon, self.image_size = episodes, horizon, image_size
        self.rgb_cache_episodes = max(1, rgb_cache_episodes)
        self._cache = OrderedDict()
        if horizon < 1:
            raise ValueError('horizon must be positive')
        self.index = [(i, j, int(a)) for i, e in enumerate(episodes)
                      for j, a in enumerate(e.anchors) if int(a) + horizon <= len(e.q)]
        if episodes and not self.index:
            raise ValueError('no RGB frames have a complete future action horizon')
        for i, episode in enumerate(episodes):
            if not any(k == i for k, _, _ in self.index):
                raise ValueError(f'{episode.id}: no complete action chunks')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, frame, anchor = self.index[index]
        e = self.episodes[i]
        if i not in self._cache:
            with np.load(e.image_path, allow_pickle=False) as archive:
                self._cache[i] = np.asarray(archive[e.image_key]).copy()
            while len(self._cache) > self.rgb_cache_episodes:
                self._cache.popitem(last=False)
        self._cache.move_to_end(i)
        return {'image': image_tensor(self._cache[i][frame], self.image_size),
                'q': torch.from_numpy(e.q[anchor].copy()),
                'target': torch.from_numpy(e.ref[anchor:anchor+self.horizon].copy()),
                'instruction': e.instruction, 'episode_id': e.id,
                'state_index': anchor, 'rgb_index': frame}
