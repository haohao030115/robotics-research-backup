"""Short reproducible schema-v3/single-GPU/DDP save-resume verification.

All inputs are synthetic; this script does not benchmark or dispatch actions.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_synthetic import create_fixture
from utils.config import dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'outputs/tonight/verification'))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if (output/'readiness.json').exists():
        raise ValueError('verification already complete; use a fresh --output')
    output.mkdir(parents=True, exist_ok=True)
    manifest = create_fixture(output/'data')
    content = json.loads(manifest.read_text())
    for entry in content['episodes']:
        entry['episode_path'] = entry.pop('states')
        entry['rgb_path'] = entry.pop('images')
    manifest = manifest.parent/'training_manifest.json'
    dump_json(manifest, content)
    configs = {}
    for kind in ('baseline', 'flow'):
        base = json.loads((ROOT/f'configs/{kind}.json').read_text())
        assert base['model']['language_enabled'] is False
        base['data']['manifest'] = str(manifest)
        base['model'].update(image_size=32, d_model=32, vision_channels=[8, 16],
                             state_hidden=32, fusion_hidden=64, depth=2, heads=4,
                             mlp_ratio=2, sampling_steps=4)
        base['training'].update(epochs=4, max_steps=3, batch_size_per_rank=4,
                               num_workers=0, log_every=1, validation_sampling_batches=1)
        configs[kind] = base

    def run(command, log, timeout=240):
        print('RUN:', ' '.join(command), flush=True)
        with (output/log).open('w') as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
        print('PASS:', log, flush=True)

    results = {}
    for kind, base in configs.items():
        results[kind] = {}
        for world in (1, 8):
            name = f'{kind}_gpu{world}'
            cfg = copy.deepcopy(base)
            cfg['training']['output_dir'] = str(output/name)
            config_path = output/f'{name}.json'
            dump_json(config_path, cfg)
            launch = [sys.executable, '-B'] if world == 1 else [sys.executable, '-B', '-m',
                'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=8']
            command = launch + [f'train_{kind}.py', '--config', str(config_path)]
            checkpoint = output/name/'checkpoints/latest.pt'
            run(command + ['--stop-after-steps', '1'], name+'_stop.log')
            stopped = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert stopped['global_step'] == 1 and len(stopped['rng_states']) == world
            run(command + ['--resume', str(checkpoint)], name+'_resume.log')
            restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert restored['global_step'] == 3
            assert restored['scheduler']['last_epoch'] == 3
            assert len(restored['rng_states']) == world
            assert restored['resume_progress']['world_size'] == world
            assert all(torch.isfinite(v).all().item() for v in restored['model'].values())
            metrics = [json.loads(line) for line in (output/name/'logs/metrics.jsonl').read_text().splitlines()]
            assert any(r['phase'] == 'epoch' and r['val'] for r in metrics)
            results[kind][str(world)] = {'training': 'PASS', 'save_resume': 'PASS',
                'logging': 'PASS', 'validation': 'PASS', 'steps': 3,
                'checkpoint': str(checkpoint), 'stop_progress': stopped['resume_progress']}
        single = output/f'{kind}_gpu1/checkpoints/latest.pt'
        run([sys.executable, '-B', 'scripts/offline_predict.py', '--checkpoint', str(single),
             '--output', str(output/f'{kind}_prediction')], kind+'_inference.log')
        inference = json.loads((output/f'{kind}_prediction.json').read_text())
        assert inference['shape'] == [20, 43] and inference['inverse_normalization'] == 'PASS'
        results[kind]['inference'] = {k: inference[k] for k in
            ('shape', 'finite', 'inverse_normalization', 'body_ref_shape', 'hand_ref_shape')}
    run([sys.executable, '-B', 'scripts/check_dataset.py', '--config', str(output/'baseline_gpu1.json'),
         '--output', str(output/'schema_v3_sanity.json')], 'dataset_sanity.log')
    dump_json(output/'readiness.json', {'synthetic_only': True, 'manifest': str(manifest),
                                      'baseline': results['baseline'], 'flow': results['flow']})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
