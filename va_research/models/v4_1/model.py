import torch
from torch import nn
from va_research.models.v4.model import V4Policy, mlp

class V41Policy(V4Policy):
    def __init__(self):
        super().__init__()
        # Readiness is predicted from grasp context only; hand q/dq are excluded.
        self.grasp_ready_head = nn.Linear(256, 1)

    def forward(self, rgb, body, hand):
        out = super().forward(rgb, body, hand)
        visual = self.visual(rgb).flatten(2).transpose(1, 2) + self.position
        b = (body - self.body_mean) / self.body_std
        gc = self.grasp_cross(self.grasp_query(b)[:, None], visual)
        out['grasp_ready_logit'] = self.grasp_ready_head(gc)
        return out
