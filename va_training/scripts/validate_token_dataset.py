"""Read all selected episodes and write the explicit VA V2 alignment audit."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.tokens import load_token_episodes, TokenDataset, validation_report
from datasets.episodes import split_episodes
from sonic.release import bundle_provenance
from utils.config import dump_json
from utils.v2_config import load_v2_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    config = load_v2_config(args.config)
    sonic = bundle_provenance(config['sonic'])
    episodes = load_token_episodes(config['data'])
    dataset = TokenDataset(episodes, config['model']['image_size'], config['model']['history_length'],
                           config['model']['token_chunk_size'], config['data']['target_offset'])
    train, val = split_episodes(episodes, config['data'])
    report = validation_report(episodes, dataset)
    report['split'] = {'train': [e.id for e in train], 'val': [e.id for e in val]}
    report['provenance'] = {'va_config': config, 'sonic': sonic}
    dump_json(args.output, report)
    print(f'Validated {len(episodes)} episodes, {len(dataset)} causal samples; source-level split {len(train)}/{len(val)}')


if __name__ == '__main__':
    main()
