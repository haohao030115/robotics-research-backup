from torch import nn


class StateEncoder(nn.Module):
    """Current actual body_q + hand_q only; dq/history are future extensions."""
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(43), nn.Linear(43, config['state_hidden']),
                                    nn.GELU(), nn.Linear(config['state_hidden'], config['d_model']))

    def forward(self, q):
        if q.ndim != 2 or q.shape[1] != 43:
            raise ValueError('current q must have shape [B,43]')
        return self.layers(q)[:, None]
