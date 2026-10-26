"""Local workaround for d0930 torch 2.9's absent MUSA reducer timer.

The native reducer, buckets, gradient all-reduce and initial synchronization remain
unchanged. Only optional runtime timing collection in the Python DDP logger is
skipped. No installed package is patched; validation requires multiple iterations.
"""
from torch.nn.parallel import DistributedDataParallel


class _WithoutRuntimeTiming:
    def __init__(self, logger):
        self._logger = logger

    def set_runtime_stats_and_log(self):
        return None

    def __getattr__(self, name):
        return getattr(self._logger, name)


def wrap_ddp(policy, runtime, disable_musa_runtime_timing=True):
    if not runtime.distributed:
        return policy
    if runtime.device.type == 'musa':
        model = DistributedDataParallel(policy, device_ids=[runtime.local_rank],
                                        output_device=runtime.local_rank)
        if disable_musa_runtime_timing:
            model.logger = _WithoutRuntimeTiming(model.logger)
            if runtime.rank == 0:
                print('MUSA DDP: optional runtime timing disabled (d0930 reducer timer workaround)', flush=True)
        return model
    return DistributedDataParallel(policy)
