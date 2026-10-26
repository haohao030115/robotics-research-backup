import time
import numpy as np
import torch
from .data import preprocess_uint8, current_proprio
from .model import synchronize


class CurrentPolicy:
    def __init__(self, model):
        self.model=model.eval(); self.device=next(model.parameters()).device

    @torch.inference_mode()
    def predict(self, rgb, state):
        timing={}; synchronize(self.device); start=time.perf_counter()
        image=(preprocess_uint8(rgb).float()/255)[None].to(self.device)
        proprio=torch.from_numpy(current_proprio(state))[None].to(self.device)
        synchronize(self.device); forward_start=time.perf_counter()
        timing['preprocessing']=(forward_start-start)*1000
        out=self.model(image,proprio,timing); synchronize(self.device)
        timing['total_va_forward']=(time.perf_counter()-forward_start)*1000
        arrays={k:v[0].cpu().numpy().copy() for k,v in out.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError('Non-finite prediction')
        return arrays,timing


def latency_report(rows):
    return {k:dict(count=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(np.max(v)))
            for k in sorted({k for r in rows for k in r}) if (v:=[r[k] for r in rows if k in r])}
