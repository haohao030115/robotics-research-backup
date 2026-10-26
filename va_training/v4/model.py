import torch
from torch import nn


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, c))


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnorm = nn.LayerNorm(256)
        self.vnorm = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.ff = mlp(256, 512, 256)

    def forward(self, q, visual):
        v = self.vnorm(visual)
        q = q + self.attn(self.qnorm(q), v, v, need_weights=False)[0]
        return (q + self.ff(self.norm(q)))[:, 0]


class MLPFusion(nn.Module):
    """A fixed-shape visual fusion alternative for the architecture ablation.

    It keeps the 256-D interface used by the heads while replacing learned
    query-to-token attention with a pooled visual feature and an MLP.
    """
    def __init__(self):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU())

    def forward(self, q, visual):
        return self.fuse(torch.cat([q[:, 0], visual.mean(1)], dim=-1))


class V4Policy(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in zip([3, 32, 64, 128], [32, 64, 128, 256]):
            layers += [nn.Conv2d(a, b, 3, 2, 1), nn.GroupNorm(8, b), nn.GELU()]
        self.visual = nn.Sequential(*layers)
        self.position = nn.Parameter(torch.randn(1, 64, 256) * .02)
        self.body_query = nn.Sequential(mlp(68, 256, 256), nn.LayerNorm(256), nn.GELU())
        self.grasp_query = nn.Sequential(mlp(68, 256, 256), nn.LayerNorm(256), nn.GELU())
        self.body_cross = CrossAttention()
        self.grasp_cross = CrossAttention()
        self.hand_state = nn.Sequential(mlp(28, 128, 128), nn.GELU())
        self.body_feature = nn.Sequential(mlp(256, 512, 512), nn.GELU())
        self.body_head = mlp(512, 512, 640)
        self.phase_feature = nn.Sequential(nn.Linear(256, 64), nn.GELU())
        self.phase_head = nn.Linear(64, 3)
        self.progress_head = nn.Linear(64, 1)
        self.hand_head = nn.Sequential(nn.Linear(448, 512), nn.GELU(), mlp(512, 512, 700))
        for name, width in [('body', 68), ('hand', 28)]:
            self.register_buffer(name + '_mean', torch.zeros(width))
            self.register_buffer(name + '_std', torch.ones(width))

    def forward(self, rgb, body, hand):
        if rgb.shape[1:] != (3, 128, 128) or body.shape != (len(rgb), 68) or hand.shape != (len(rgb), 28):
            raise ValueError('Expected current RGB [B,3,128,128], body [B,68], hand [B,28]')
        visual = self.visual(rgb).flatten(2).transpose(1, 2) + self.position
        b = (body - self.body_mean) / self.body_std
        h = (hand - self.hand_mean) / self.hand_std
        bc = self.body_cross(self.body_query(b)[:, None], visual)
        gc = self.grasp_cross(self.grasp_query(b)[:, None], visual)
        phase = self.phase_feature(gc)
        return dict(body_tokens=self.body_head(self.body_feature(bc)).reshape(-1, 10, 64),
                    hand_ref=self.hand_head(torch.cat([gc, phase, self.hand_state(h)], -1)).reshape(-1, 50, 14),
                    phase_logits=self.phase_head(phase), closure_progress=self.progress_head(phase).sigmoid()[:, 0])


class MLPFusionPolicy(V4Policy):
    """V4 with pooled visual features and MLP fusion instead of cross-attention."""
    model_variant = 'mlp_fusion'

    def __init__(self):
        super().__init__()
        self.body_fusion = MLPFusion()
        self.grasp_fusion = MLPFusion()

    def forward(self, rgb, body, hand):
        if rgb.shape[1:] != (3, 128, 128) or body.shape != (len(rgb), 68) or hand.shape != (len(rgb), 28):
            raise ValueError('Expected current RGB [B,3,128,128], body [B,68], hand [B,28]')
        visual = self.visual(rgb).flatten(2).transpose(1, 2) + self.position
        b = (body - self.body_mean) / self.body_std
        h = (hand - self.hand_mean) / self.hand_std
        bc = self.body_fusion(self.body_query(b)[:, None], visual)
        gc = self.grasp_fusion(self.grasp_query(b)[:, None], visual)
        phase = self.phase_feature(gc)
        return dict(body_tokens=self.body_head(self.body_feature(bc)).reshape(-1, 10, 64),
                    hand_ref=self.hand_head(torch.cat([gc, phase, self.hand_state(h)], -1)).reshape(-1, 50, 14),
                    phase_logits=self.phase_head(phase), closure_progress=self.progress_head(phase).sigmoid()[:, 0])


class ProprioOnlyPolicy(V4Policy):
    """V4 policy whose forward path ignores RGB and uses proprioceptive fusion."""
    model_variant = 'proprio_only'

    def __init__(self):
        super().__init__()
        self.body_fusion = nn.Sequential(mlp(256, 512, 256), nn.LayerNorm(256), nn.GELU())
        self.grasp_fusion = nn.Sequential(mlp(256, 512, 256), nn.LayerNorm(256), nn.GELU())

    def forward(self, rgb, body, hand):
        if rgb.shape[1:] != (3, 128, 128) or body.shape != (len(rgb), 68) or hand.shape != (len(rgb), 28):
            raise ValueError('Expected current RGB [B,3,128,128], body [B,68], hand [B,28]')
        b = (body - self.body_mean) / self.body_std
        h = (hand - self.hand_mean) / self.hand_std
        bc = self.body_fusion(self.body_query(b))
        gc = self.grasp_fusion(self.grasp_query(b))
        phase = self.phase_feature(gc)
        return dict(body_tokens=self.body_head(self.body_feature(bc)).reshape(-1, 10, 64),
                    hand_ref=self.hand_head(torch.cat([gc, phase, self.hand_state(h)], -1)).reshape(-1, 50, 14),
                    phase_logits=self.phase_head(phase), closure_progress=self.progress_head(phase).sigmoid()[:, 0])


def policy_for_variant(variant=None):
    """Construct a policy from checkpoint metadata without changing baseline behavior."""
    return {'mlp_fusion': MLPFusionPolicy, 'proprio_only': ProprioOnlyPolicy}.get(variant, V4Policy)()
