import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import load_episodes, split_episodes, EpisodeDataset
from utils.runtime import initialize, cleanup
from utils.checkpoint import load_policy
from trainers.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='musa', choices=['musa', 'cpu'])
    parser.add_argument('--manifest', help='Override source manifest, retaining checkpoint split settings')
    parser.add_argument('--output', help='Write evaluation metrics as JSON')
    args = parser.parse_args()
    runtime = initialize(args.device, 38)
    try:
        policy, checkpoint = load_policy(args.checkpoint, runtime.device)
        config = checkpoint['config']
        if args.manifest:
            config['data']['manifest'] = str(Path(args.manifest).resolve())
        episodes = load_episodes(config['data'])
        _, val = split_episodes(episodes, config['data'])
        train_ids = set(checkpoint['normalization_stats']['train_episode_ids'])
        if train_ids & {e.id for e in val}:
            raise ValueError('evaluation episodes overlap checkpoint training episodes')
        if not val:
            raise ValueError('no validation episodes; evaluation requires a held-out split')
        data = EpisodeDataset(val, config['model']['horizon'], config['model']['image_size'])
        loader = DataLoader(data, batch_size=config['training']['batch_size_per_rank'],
                            sampler=list(range(runtime.rank, len(data), runtime.world_size)))
        result = evaluate(policy, loader, config, runtime)
        if runtime.rank == 0:
            print(json.dumps(result, indent=2))
            if args.output:
                from utils.config import dump_json
                dump_json(args.output, result)
    finally:
        cleanup(runtime)


if __name__ == '__main__':
    main()
