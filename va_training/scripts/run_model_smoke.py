"""One default-size forward/backward plus sampling; no sustained training."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_policy
from utils.runtime import initialize, cleanup
from utils.config import dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='musa', choices=['musa', 'cpu'])
    parser.add_argument('--output', default=str(ROOT/'outputs/default_model_smoke.json'))
    args = parser.parse_args()
    runtime = initialize(args.device, 38)
    if runtime.distributed:
        raise ValueError('run_model_smoke is a single-device test')
    results = {}
    try:
        stats = {key: {'mean': [0.0]*dim, 'std': [1.0]*dim}
                 for key, dim in [('image', 3), ('state', 43), ('action', 43)]}
        for kind in ('baseline', 'flow'):
            config = json.loads((ROOT/f'configs/{kind}.json').read_text())
            model = build_policy(config, stats).to(runtime.device)
            size, horizon = config['model']['image_size'], config['model']['horizon']
            image = torch.rand(1, 3, size, size, device=runtime.device)
            q = torch.randn(1, 43, device=runtime.device)
            target = torch.randn(1, horizon, 43, device=runtime.device)
            if kind == 'flow':
                xt, t, velocity = model.scheduler.training_pair(target)
                prediction = model(image, q, xt, t)
                loss = (prediction-velocity).square().mean()
            else:
                prediction = model(image, q)
                loss = (prediction-target).abs().mean()
            assert prediction.shape == (1, horizon, 43) and torch.isfinite(loss).item()
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
                       for p in model.parameters())
            sample = model.predict(image, q[:, :29], q[:, 29:])
            assert sample.shape == (1, horizon, 43) and torch.isfinite(sample).all().item()
            results[kind] = {'config': config['model'], 'forward': 'PASS', 'backward': 'PASS',
                             'sampling_shape': list(sample.shape), 'sampling_finite': True,
                             'loss': loss.item(), 'parameters': sum(p.numel() for p in model.parameters())}
            print(json.dumps({kind: results[kind]}), flush=True)
            del model, prediction, loss, sample
        dump_json(args.output, results)
    finally:
        cleanup(runtime)


if __name__ == '__main__':
    main()
