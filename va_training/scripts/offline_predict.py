"""Load one dataset sample and checkpoint, export only abstract trajectories."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets import load_episodes, split_episodes, EpisodeDataset
from models.inference import ModelOnlyPolicy, split_action_chunk
from utils.runtime import initialize, cleanup
from utils.config import dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='musa', choices=['musa', 'cpu'])
    parser.add_argument('--manifest')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'all'])
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--output', required=True, help='Output prefix, without .npz/.json')
    args = parser.parse_args()
    runtime = initialize(args.device, 38)
    if runtime.distributed:
        raise ValueError('offline inference uses one process')
    try:
        interface, checkpoint = ModelOnlyPolicy.from_checkpoint(args.checkpoint, runtime.device)
        config = checkpoint['config']
        if args.manifest:
            config['data']['manifest'] = str(Path(args.manifest).resolve())
        episodes = load_episodes(config['data'])
        train, val = split_episodes(episodes, config['data'])
        selected = {'train': train, 'val': val, 'all': episodes}[args.split]
        data = EpisodeDataset(selected, config['model']['horizon'], config['model']['image_size'])
        if not 0 <= args.sample < len(data):
            raise ValueError('sample index out of range or selected split empty')
        sample = data[args.sample]
        model = interface.policy
        noise = torch.randn(1, config['model']['horizon'], 43, device=runtime.device) \
            if config['model_type'] == 'flow' else None
        prediction = interface.predict(sample['image'], sample['q'][:29], sample['q'][29:],
                                       sample['instruction'], args.steps, noise)
        image, body, hand, _ = interface.prepare(sample['image'], sample['q'][:29], sample['q'][29:])
        q = torch.cat((body, hand), -1)
        with torch.inference_mode():
            normalized = model.sample_normalized(image, q, sample['instruction'], args.steps, noise) \
                if config['model_type'] == 'flow' else model(image, q, sample['instruction'])
        stats = checkpoint['normalization_stats']['action']
        mean = torch.tensor(stats['mean'], device=runtime.device)
        std = torch.tensor(stats['std'], device=runtime.device)
        torch.testing.assert_close((prediction[None]-mean)/std, normalized, atol=2e-5, rtol=2e-5)
        body_ref, hand_ref = split_action_chunk(prediction)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        values = {k: v.detach().cpu().numpy() for k, v in {
            'action_chunk': prediction, 'body_ref': body_ref, 'hand_ref': hand_ref,
            'current_q': sample['q'], 'target_reference': sample['target']}.items()}
        np.savez_compressed(str(out)+'.npz', **values)
        raw_manifest = json.loads(Path(config['data']['manifest']).read_text())
        synthetic = raw_manifest.get('synthetic', False) if isinstance(raw_manifest, dict) else False
        report = {'checkpoint': args.checkpoint, 'model_type': config['model_type'], 'synthetic': synthetic,
                  'episode_id': sample['episode_id'], 'rgb_index': sample['rgb_index'],
                  'state_index': sample['state_index'], 'shape': list(prediction.shape),
                  'finite': True, 'inverse_normalization': 'PASS',
                  'body_ref_shape': list(body_ref.shape), 'hand_ref_shape': list(hand_ref.shape),
                  'controller_dispatch': False, 'trajectory': prediction.cpu().tolist()}
        dump_json(str(out)+'.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'trajectory'}, indent=2))
    finally:
        cleanup(runtime)


if __name__ == '__main__':
    main()
