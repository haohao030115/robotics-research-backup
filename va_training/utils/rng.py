"""Checkpoint RNG state without storing NumPy objects or requiring pickle code."""
import random
import numpy as np
import torch


def capture_rng(runtime):
    numpy = np.random.get_state()
    result = {'torch': torch.get_rng_state(), 'python': random.getstate(),
              'numpy': [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])]}
    if runtime.device.type == 'musa':
        result['musa'] = torch.musa.get_rng_state(runtime.local_rank).cpu()
    return result


def restore_rng(state, runtime):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))
    if runtime.device.type == 'musa':
        torch.musa.set_rng_state(state['musa'].cpu(), runtime.local_rank)
