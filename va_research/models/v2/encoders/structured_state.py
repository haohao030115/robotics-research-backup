"""Named dynamics/base fields and masked short history, independent of SONIC's decoder vector."""
import torch
from torch import nn
from va_research.datasets.tokens import STATE_WIDTHS


class StructuredStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, dim = config['field_hidden'], config['d_model']
        self.fields = nn.ModuleDict({name: nn.Sequential(nn.Linear(n, width), nn.GELU())
                                     for name, n in STATE_WIDTHS.items()})
        self.frame = nn.Sequential(nn.Linear(len(STATE_WIDTHS)*width, dim), nn.LayerNorm(dim), nn.GELU())
        self.history = nn.GRUCell(dim, dim)
        self.history_length, self.dim = config['history_length'], dim

    def forward(self, state, mask):
        if set(state) != set(STATE_WIDTHS) or mask.ndim != 2 or mask.shape[1] != self.history_length:
            raise ValueError('Expected named state fields and history_mask [B,L]')
        values = []
        for name, width in STATE_WIDTHS.items():
            if state[name].shape != (*mask.shape, width):
                raise ValueError(f'{name}: expected [B,L,{width}]')
            values.append(self.fields[name](state[name]))
        frames = self.frame(torch.cat(values, -1))
        hidden = frames.new_zeros((len(frames), self.dim))
        for i in range(self.history_length):
            candidate = self.history(frames[:, i], hidden)
            hidden = torch.where(mask[:, i, None], candidate, hidden)
        return hidden
