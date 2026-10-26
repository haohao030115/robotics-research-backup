from torch import nn


class FusionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2*config['d_model'], config['fusion_hidden']),
                                    nn.LayerNorm(config['fusion_hidden']), nn.GELU(),
                                    nn.Linear(config['fusion_hidden'], config['fusion_hidden']), nn.GELU())

    def forward(self, features):
        return self.layers(features)
