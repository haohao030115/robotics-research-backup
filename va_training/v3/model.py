"""Spatial CNN and state-query cross-attention; no temporal state."""
import time
import torch
from torch import nn


def synchronize(device):
    if device.type == 'musa':
        torch.musa.synchronize(device)
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnorm = nn.LayerNorm(256)
        self.vnorm = nn.LayerNorm(256)
        self.attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.ff = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, query, visual):
        v = self.vnorm(visual)
        query = query + self.attention(self.qnorm(query), v, v, need_weights=False)[0]
        return query + self.ff(self.norm(query))


class V3Policy(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for cin, cout in zip([3, 32, 64, 128], [32, 64, 128, 256]):
            blocks += [nn.Conv2d(cin, cout, 3, 2, 1), nn.GroupNorm(8, cout), nn.GELU()]
        self.visual = nn.Sequential(*blocks)
        self.position = nn.Parameter(torch.empty(1, 64, 256))
        nn.init.normal_(self.position, std=.02)
        self.state = nn.Sequential(nn.Linear(96, 256), nn.GELU(), nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU())
        self.query = nn.Linear(256, 256)
        self.cross = CrossBlock()
        self.shared = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU())
        self.body = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 128))
        self.hand = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 28))
        self.register_buffer('state_mean', torch.zeros(96))
        self.register_buffer('state_std', torch.ones(96))

    def forward(self, image, proprio, timings=None):
        if image.ndim != 4 or image.shape[1:] != (3, 128, 128) or proprio.shape != (len(image), 96):
            raise ValueError('V3 requires current RGB [B,3,128,128], current proprio [B,96]')
        def measure(name, operation):
            if timings is None:
                return operation()
            synchronize(image.device)
            start = time.perf_counter()
            result = operation()
            synchronize(image.device)
            timings[name] = (time.perf_counter()-start)*1000
            return result
        visual = measure('visual_encoder', lambda: self.visual(image).flatten(2).transpose(1, 2)+self.position)
        query = measure('state_encoder', lambda: self.query(self.state((proprio-self.state_mean)/self.state_std))[:, None])
        shared = measure('cross_attention_fusion', lambda: self.shared(self.cross(query, visual)[:, 0]))
        body, hand = measure('body_hand_heads', lambda: (self.body(shared).reshape(-1, 2, 64), self.hand(shared).reshape(-1, 2, 14)))
        return {'sonic_token': body, 'hand_ref': hand}
