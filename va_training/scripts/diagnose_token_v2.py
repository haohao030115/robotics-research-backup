"""Reproducible, isolated VA V2 physics diagnostics; no training or SONIC edits."""
import argparse
import copy
import json
from pathlib import Path
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from diagnostics.rollout import run
from datasets.tokens import load_token_episodes
from utils.runtime import initialize, cleanup
from utils.config import dump_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episode-id', required=True)
    p.add_argument('--oracle-report', required=True)
    p.add_argument('--overfit-report', required=True)
    p.add_argument('--offline-report', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--experiment', choices=['factorial', 'learned', 'sequence'], default='factorial')
    p.add_argument('--condition', choices=['A', 'B', 'C', 'D'])
    p.add_argument('--chunk-mode', action='store_true')
    p.add_argument('--expert-hand-observation', action='store_true', help='Privileged diagnostic: measured expert hand q/dq at capture tick')
    p.add_argument('--record-dds', action='store_true')
    p.add_argument('--sequence', help='NPZ file, tokens [T,64], for controlled corruptions')
    p.add_argument('--device', choices=['cpu', 'musa'], default='musa')
    args = p.parse_args()
    if args.experiment == 'factorial' and args.condition is None:
        output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
        reports = {}
        for name in ['A', 'B', 'C', 'D']:
            argv = sys.argv[1:]
            argv[argv.index('--output')+1] = str(output/name)
            result = subprocess.run([sys.executable, '-B', __file__, *argv, '--condition', name])
            if result.returncode: raise RuntimeError(f'Diagnostic condition {name} failed to execute')
            reports[name] = json.loads((output/name/'report.json').read_text())
            dump_json(output/'summary.json', reports)
            if name == 'A' and not reports[name]['closed_loop_pass']:
                raise RuntimeError('Oracle failed; stop before other conditions')
        return
    rt = initialize(args.device, 38)
    try:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        runner = run
        if args.chunk_mode:
            from diagnostics.chunk_rollout import run as runner
        config = copy.deepcopy(saved['config']); config['training']['device'] = args.device
        config['diagnostic'] = {'expert_hand_observation': args.expert_hand_observation, 'record_dds': args.record_dds}
        data = copy.deepcopy(config['data']); data['episode_ids'] = [args.episode_id]
        episode = load_token_episodes(data)[0]
        output = Path(args.output)
        gates = dict(oracle=args.oracle_report, overfit=args.overfit_report, offline=args.offline_report)
        conditions = [('A', 'gt', 'gt'), ('B', 'gt', 'pred'), ('C', 'pred', 'gt'), ('D', 'pred', 'pred')]
        if args.experiment == 'learned': conditions = [('D', 'pred', 'pred')]
        if args.experiment == 'sequence': conditions = [('sequence', 'sequence', 'gt')]
        if args.condition: conditions = [v for v in conditions if v[0] == args.condition]
        sequence = np.load(args.sequence)['tokens'] if args.sequence else None
        if args.experiment == 'sequence' and sequence is None: raise ValueError('--sequence is required')
        reports = {}
        for name, body, hand in conditions:
            reports[name] = runner(config, args.checkpoint, episode, output, gates, body, hand, sequence)
            dump_json(output/'summary.json', reports)
            if name == 'A' and not reports[name]['closed_loop_pass']:
                raise RuntimeError('Oracle control failed; stop and diagnose interface before other conditions')
    finally:
        cleanup(rt)


if __name__ == '__main__': main()
