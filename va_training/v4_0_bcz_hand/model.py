import torch
from torch import nn
from va_training.v4.model import V4Policy
class BCZPolicy(V4Policy):
    def __init__(self):
        super().__init__(); self.close_intent_head=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,rgb,body,hand):
        out=super().forward(rgb,body,hand)
        visual=self.visual(rgb).flatten(2).transpose(1,2)+self.position
        b=(body-self.body_mean)/self.body_std
        gc=self.grasp_cross(self.grasp_query(b)[:,None],visual)
        out['close_intent_logit']=self.close_intent_head(gc)[:,0]
        return out
