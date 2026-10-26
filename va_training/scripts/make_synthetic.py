"""Clearly labeled schema-adapter fixture, never presented as expert robot data."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import dump_json


def create_fixture(directory, episodes=3, states=80, size=32):
    directory = Path(directory)
    if (directory/'manifest.json').exists():
        raise ValueError('fixture already exists; choose a new directory')
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    generator = np.random.default_rng(38)
    for i in range(episodes):
        time = np.arange(states, dtype=np.float64)/50
        # Current actual q differs from reference q; tests must catch interchange.
        q = generator.normal(0, 0.3, (states, 43)).astype(np.float32)
        offset = generator.normal(0, 0.2, 43).astype(np.float32)
        ref = np.broadcast_to(offset, (states, 43)).copy()
        indices = np.arange(0, states, 5)
        rgb = generator.integers(0, 256, (len(indices), size, size, 3), dtype=np.uint8)
        np.savez(directory/f'states_{i}.npz', timestamps=time,
                 body_q=q[:, :29], hand_q=q[:, 29:], body_ref_q=ref[:, :29], hand_ref_q=ref[:, 29:],
                 body_dq=np.zeros((states, 29), np.float32), hand_dq=np.zeros((states, 14), np.float32))
        np.savez(directory/f'images_{i}.npz', head_rgb=rgb, timestamps=time[indices],
                 rgb_reference_frame_indices=indices)
        records.append({'id': f'SYNTHETIC_{i}', 'states': f'states_{i}.npz', 'images': f'images_{i}.npz',
                        'split': 'val' if i == episodes-1 and episodes > 1 else 'train',
                        'instruction': 'Pick up the red cube and lift it from the table.'})
    dump_json(directory/'manifest.json', {'synthetic': True, 'episodes': records})
    return directory/'manifest.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'tests/fixtures/synthetic'))
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError('episodes must be positive')
    manifest = create_fixture(args.output, args.episodes)
    baseline = json.loads((ROOT/'configs/baseline.json').read_text())
    baseline['data']['manifest'] = str(manifest.resolve())
    baseline['model'].update(image_size=32, d_model=32, vision_channels=[8, 16],
                             state_hidden=32, fusion_hidden=64, depth=2, heads=4, mlp_ratio=2,
                             sampling_steps=4)
    baseline['training'].update(epochs=30, max_steps=40, batch_size_per_rank=4,
                               num_workers=0, lr=0.003, log_every=10,
                               output_dir=str(ROOT/'outputs/synthetic_baseline'))
    dump_json(Path(args.output)/'baseline_smoke.json', baseline)
    baseline['model_type'] = 'flow'
    baseline['training'].update(max_steps=30, lr=0.001, output_dir=str(ROOT/'outputs/synthetic_flow'))
    dump_json(Path(args.output)/'flow_smoke.json', baseline)
    print('SYNTHETIC ONLY:', manifest)


if __name__ == '__main__':
    main()
