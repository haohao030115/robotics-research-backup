"""Stage A/B VA V2 training, gated by successful frozen-decoder oracle replay."""
import argparse
from pathlib import Path
from utils.v2_config import load_v2_config
from trainers.token_training import train_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--oracle-report', required=True)
    p.add_argument('--overfit-report', help='Passing Stage A report, required for Stage B')
    p.add_argument('--stage', choices=['overfit', 'heldout'], required=True)
    p.add_argument('--episode-id', action='append'); p.add_argument('--device', choices=['cpu', 'musa'])
    p.add_argument('--max-steps', type=int); p.add_argument('--epochs', type=int)
    p.add_argument('--output'); args = p.parse_args()
    config = load_v2_config(args.config)
    if args.episode_id:
        config['data']['episode_ids'] = args.episode_id
    for key in ('device', 'max_steps', 'epochs'):
        if getattr(args, key) is not None:
            config['training'][key] = getattr(args, key)
    if args.output:
        config['training']['output_dir'] = str(Path(args.output).resolve())
    report = train_tokens(config, args.oracle_report, args.stage, args.overfit_report)
    gate = 'overfit_pass' if args.stage == 'overfit' else 'offline_evaluation_pass'
    if not report[gate]:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
