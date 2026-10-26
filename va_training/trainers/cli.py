import argparse
from utils.config import load_config
from .training import run_training


def main(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--device', choices=['cpu', 'musa'])
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--output')
    parser.add_argument('--stop-after-steps', type=int,
                        help='Graceful absolute global-step stop, preserving the configured scheduler plan')
    args = parser.parse_args()
    config = load_config(args.config)
    if config['model_type'] != kind:
        raise ValueError('model_type does not match training entry point')
    if args.device:
        config['training']['device'] = args.device
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError('max-steps must be positive')
        config['training']['max_steps'] = args.max_steps
    if args.output:
        from pathlib import Path
        config['training']['output_dir'] = str(Path(args.output).resolve())
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError('stop-after-steps must be positive')
    run_training(config, args.resume, args.stop_after_steps)
