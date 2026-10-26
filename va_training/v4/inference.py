import time
import numpy as np
import torch
from PIL import Image
from .dataset import gravity


def current_state(runtime):
    d=runtime.data;q=d.qpos[runtime.root_qa+3:runtime.root_qa+7].copy()
    body=np.concatenate([d.qpos[runtime.body.qa],d.qvel[runtime.body.va],q,
                          d.qvel[runtime.root_va+3:runtime.root_va+6],gravity(q)])
    hand=np.concatenate([d.qpos[runtime.hand.qa],d.qvel[runtime.hand.va]])
    return body.copy(),hand.copy()


class Inference:
    def __init__(self, model):
        self.model=model.eval();self.device=next(model.parameters()).device

    @torch.inference_mode()
    def predict(self,rgb,body,hand):
        start=time.monotonic()
        image=np.asarray(Image.fromarray(rgb).resize((128,128),Image.Resampling.BILINEAR)).transpose(2,0,1)/255.
        tensors=[torch.tensor(x,dtype=torch.float32,device=self.device)[None] for x in (image,body,hand)]
        result={k:v[0].cpu().numpy() for k,v in self.model(*tensors).items()}
        return result,time.monotonic()-start
