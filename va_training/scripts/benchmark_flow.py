"""Synchronized wall-clock FP32 model-only latency; no training or controller."""
import argparse
import json
import statistics
import time
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_policy
from utils.checkpoint import load_policy
from utils.runtime import initialize, cleanup
from utils.config import dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Optional default-size trained Flow checkpoint')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', default=str(ROOT/'outputs/flow_latency.json'))
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 3:
        raise ValueError('warmup >=1 and repeats >=3 required')
    runtime = initialize('musa', 38)
    if runtime.distributed:
        raise ValueError('benchmark requires a single process/GPU')
    try:
        if args.checkpoint:
            model, checkpoint = load_policy(args.checkpoint, runtime.device)
            config = checkpoint['config']
        else:
            config = json.loads((ROOT/'configs/flow.json').read_text())
            stats = {k: {'mean': [0.0]*d, 'std': [1.0]*d}
                     for k, d in [('image', 3), ('state', 43), ('action', 43)]}
            model = build_policy(config, stats).to(runtime.device).eval()
        if config['model_type'] != 'flow' or config['model']['image_size'] != 128 or config['model']['horizon'] != 20:
            raise ValueError('benchmark requires Flow, image_size=128, H=20')
        image = torch.rand(1, 3, 128, 128, device=runtime.device)
        q = torch.randn(1, 43, device=runtime.device)
        xt = torch.randn(1, 20, 43, device=runtime.device)
        t = torch.full((1,), 0.5, device=runtime.device)

        def measure(operation):
            for _ in range(args.warmup):
                operation()
            torch.musa.synchronize(runtime.device)
            times = []
            for _ in range(args.repeats):
                torch.musa.synchronize(runtime.device)
                started = time.perf_counter()
                output = operation()
                torch.musa.synchronize(runtime.device)
                times.append((time.perf_counter()-started)*1000)
                assert torch.isfinite(output).all().item()
            return {'mean_ms': statistics.mean(times), 'p50_ms': float(np.percentile(times, 50)),
                    'p95_ms': float(np.percentile(times, 95)), 'min_ms': min(times), 'samples_ms': times}

        with torch.inference_mode():
            normalized_image = model.normalization.image(image)
            _, _, context = model.encode(image, q)
            vision = measure(lambda: model.observation.vision(normalized_image))
            velocity = measure(lambda: model.dit(xt, t, context))
            steps_report = {}
            for steps in (4, 8, 16):
                sampling = measure(lambda: model.scheduler.integrate(lambda x, time: model.dit(x, time, context),
                    (1, 20, 43), runtime.device, torch.float32, steps))
                policy = measure(lambda: model.predict(image, q[:, :29], q[:, 29:], steps=steps))
                steps_report[str(steps)] = {'total_sampling': sampling, 'end_to_end_policy': policy}
        report = {'device': torch.musa.get_device_name(runtime.local_rank), 'device_index': runtime.local_rank,
                  'torch': torch.__version__, 'dtype': 'float32', 'batch_size': 1, 'image_size': 128,
                  'model': config['model'], 'parameters': sum(p.numel() for p in model.parameters()),
                  'weights': args.checkpoint or 'random initialization; no dataset/accuracy claim',
                  'warmup': args.warmup, 'repeats': args.repeats,
                  'measurement': 'synchronized wall-clock; resident device tensors; no disk/image decoding/transfer',
                  'vision_encoder': vision, 'dit_velocity_eval': velocity, 'euler_steps': steps_report,
                  'boundaries': {'vision_encoder': 'CNN + spatial projection/position; input already normalized',
                    'dit_velocity_eval': 'one time embedding + 6 blocks + velocity projection; context cached',
                    'total_sampling': 'Gaussian draw + Euler loop; observation encoding and inverse normalization excluded',
                    'end_to_end_policy': 'model predict incl. state concat, normalization, observation encoding, Gaussian/Euler, inverse normalization'}}
        dump_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('model', 'boundaries')}, indent=2))
    finally:
        cleanup(runtime)


if __name__ == '__main__':
    main()
