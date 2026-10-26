"""Short real-project DDP check of both policies; random tensors only."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import torch
import torch.distributed as dist
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config
from utils.runtime import initialize, cleanup
from utils.distributed import wrap_ddp
from models import build_policy


def identity_stats():
    return {key: {'mean': [0.0]*dim, 'std': [1.0]*dim}
            for key, dim in [('image', 3), ('state', 43), ('action', 43)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError('use at least two steps to exercise repeated DDP forward')
    config = load_config(args.config)
    runtime = initialize(config['training']['device'], 38)
    stage = 'collective'
    try:
        value = torch.tensor(float(runtime.rank+1), device=runtime.device)
        dist.all_reduce(value)
        assert value.item() == runtime.world_size*(runtime.world_size+1)/2
        results = {'all_reduce': 'PASS', 'rank': runtime.rank, 'device': str(runtime.device)}
        for kind in ('baseline', 'flow'):
            stage = f'{kind} model init'
            config['model_type'] = kind
            policy = build_policy(config, identity_stats()).to(runtime.device)
            model = wrap_ddp(policy, runtime, config['training'].get('disable_musa_ddp_runtime_timing', True))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            size, horizon = config['model']['image_size'], config['model']['horizon']
            image = torch.rand(2, 3, size, size, device=runtime.device)
            q = torch.randn(2, 43, device=runtime.device)
            target = torch.randn(2, horizon, 43, device=runtime.device)
            before = next(policy.parameters()).detach().clone()
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                stage = f'{kind} forward'
                if kind == 'flow':
                    xt, t, velocity = policy.scheduler.training_pair(target)
                    prediction = model(image, q, xt, t)
                    loss = (prediction-velocity).square().mean()
                else:
                    prediction = model(image, q)
                    loss = (prediction-target).abs().mean()
                assert prediction.shape == target.shape and torch.isfinite(loss).item()
                stage = f'{kind} backward'
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
                           for p in policy.parameters())
                stage = f'{kind} optimizer'
                optimizer.step()
            assert (before-next(policy.parameters())).abs().max().item() > 0
            flat = torch.cat([p.detach().flatten() for p in policy.parameters()])
            average = flat.clone()
            dist.all_reduce(average)
            assert (flat-average/runtime.world_size).abs().max().item() < 1e-5
            results[kind] = {'forward': 'PASS', 'backward': 'PASS', 'optimizer': 'PASS',
                             'steps': args.steps, 'finite_loss': loss.item(),
                             'memory_bytes': torch.musa.memory_allocated()}
            del model, policy, optimizer, flat, average, prediction, loss, before
        dist.barrier()
        print(json.dumps(results), flush=True)
    except Exception:
        print(f'FAIL rank={runtime.rank} stage={stage}', flush=True)
        traceback.print_exc()
        raise
    finally:
        cleanup(runtime)


if __name__ == '__main__':
    main()
