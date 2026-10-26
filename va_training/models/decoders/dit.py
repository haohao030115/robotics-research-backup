import math
import torch
from torch import nn


class Attention(nn.Module):
    """Explicit standard PyTorch attention; avoids fused backend assumptions."""
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError('d_model must be divisible by heads')
        self.heads, self.width = heads, dim//heads
        self.q, self.k, self.v, self.output = [nn.Linear(dim, dim) for _ in range(4)]

    def forward(self, query, context):
        batch, length, dim = query.shape
        def reshape(x):
            return x.reshape(batch, -1, self.heads, self.width).transpose(1, 2)
        q, k, v = reshape(self.q(query)), reshape(self.k(context)), reshape(self.v(context))
        probabilities = ((q @ k.transpose(-2, -1)) / math.sqrt(self.width)).softmax(-1)
        output = (probabilities @ v).transpose(1, 2).contiguous().reshape(batch, length, dim)
        return self.output(output)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, ratio):
        super().__init__()
        self.norm1, self.norm2, self.context_norm, self.norm3 = [nn.LayerNorm(dim) for _ in range(4)]
        self.self_attention, self.cross_attention = Attention(dim, heads), Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*ratio), nn.GELU(), nn.Linear(dim*ratio, dim))

    def forward(self, tokens, context):
        x = self.norm1(tokens)
        tokens = tokens + self.self_attention(x, x)
        tokens = tokens + self.cross_attention(self.norm2(tokens), self.context_norm(context))
        return tokens + self.mlp(self.norm3(tokens))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError('d_model must be even and >=4')
        frequencies = torch.exp(-math.log(10000) * torch.arange(dim//2)/(dim//2-1))
        self.register_buffer('frequencies', frequencies)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, t):
        angles = t[:, None]*1000*self.frequencies[None]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))


class TrajectoryDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config['d_model']
        self.horizon, self.action_dim = config['horizon'], config['action_dim']
        self.input = nn.Linear(self.action_dim, d)
        self.position = nn.Parameter(torch.zeros(1, self.horizon, d))
        nn.init.normal_(self.position, std=0.02)
        self.time = TimeEmbedding(d)
        self.blocks = nn.ModuleList([DiTBlock(d, config['heads'], config['mlp_ratio'])
                                     for _ in range(config['depth'])])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.action_dim))

    def forward(self, trajectory, t, context):
        if trajectory.shape[1:] != (self.horizon, self.action_dim) or t.shape != (len(trajectory),):
            raise ValueError('expected trajectory [B,H,43] and t [B]')
        tokens = self.input(trajectory) + self.position + self.time(t)[:, None]
        for block in self.blocks:
            tokens = block(tokens, context)
        return self.output(tokens)
