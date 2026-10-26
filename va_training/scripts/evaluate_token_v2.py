"""Standalone offline evaluation using checkpoint-recorded source episode splits."""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.checkpoint import load_policy
from utils.runtime import initialize, cleanup
from utils.config import dump_json
from datasets.tokens import load_token_episodes
from trainers.token_training import dataset_for, evaluate_tokens
from sonic.release import bundle_provenance, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--device', choices=['cpu', 'musa'], default='musa')
    args = p.parse_args(); runtime = initialize(args.device, 38)
    try:
        policy, checkpoint = load_policy(args.checkpoint, runtime.device)
        if checkpoint['config']['model_type'] != 'token_v2':
            raise ValueError('Expected VA V2 checkpoint')
        config = checkpoint['config']; sonic = bundle_provenance(config['sonic'])
        if sonic != checkpoint['provenance']['sonic']:
            raise ValueError('Frozen SONIC bundle differs from training provenance')
        split = __import__('json').loads((Path(args.checkpoint).parents[1]/'split.json').read_text())
        ids = set(split[args.split]); episodes = [e for e in load_token_episodes(config['data']) if e.id in ids]
        if not episodes:
            raise ValueError('Requested checkpoint split is empty')
        data = dataset_for(episodes, config)
        report = evaluate_tokens(policy, DataLoader(data, batch_size=config['training']['batch_size_per_rank']), runtime.device)
        report.update(split=args.split, episode_ids=sorted(ids), checkpoint=args.checkpoint,
                      checkpoint_sha256=digest(args.checkpoint), provenance=checkpoint['provenance'])
        dump_json(args.output, report)
        print({k: report[k] for k in ('samples', 'token_mae', 'token_mse', 'fsq_level_accuracy', 'hand_mae')})
    finally:
        cleanup(runtime)


if __name__ == '__main__':
    main()
