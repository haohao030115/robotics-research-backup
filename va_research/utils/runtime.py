from dataclasses import dataclass
from datetime import timedelta
import os
import random
import numpy as np
import torch
import torch.distributed as dist


@dataclass
class Runtime:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int

    @property
    def distributed(self):
        return self.world_size > 1


def initialize(device_kind, seed):
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if device_kind == 'musa':
        import torch_musa  # Registers the installed MUSA/MCCL backend.
        if not torch.musa.is_available() or local >= torch.musa.device_count():
            raise RuntimeError('requested MUSA device is unavailable')
        torch.musa.set_device(local)
        device, backend = torch.device(f'musa:{local}'), 'mccl'
    elif device_kind == 'cpu':
        device, backend = torch.device('cpu'), 'gloo'
    else:
        raise ValueError('supported devices: musa, cpu')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_kind == 'musa':
        torch.musa.manual_seed_all(seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 1)))
    if world > 1:
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=120))
    print(f'rank={rank} device={device} world_size={world} backend={backend}', flush=True)
    return Runtime(device, rank, local, world)


def cleanup(runtime):
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()
