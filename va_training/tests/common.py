import copy
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def config():
    value = json.loads((ROOT/'configs/baseline.json').read_text())
    value['model'].update(image_size=16, d_model=16, vision_channels=[4, 8],
                          state_hidden=16, fusion_hidden=32, depth=2, heads=4,
                          mlp_ratio=2, sampling_steps=3)
    return value


def stats():
    return {key: {'mean': [0.0]*dim, 'std': [1.0]*dim}
            for key, dim in [('image', 3), ('state', 43), ('action', 43)]}


def batch(cfg):
    model = cfg['model']
    return torch.rand(2, 3, model['image_size'], model['image_size']), torch.randn(2, 43), \
        torch.randn(2, model['horizon'], 43)
