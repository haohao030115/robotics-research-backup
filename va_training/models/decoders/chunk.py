from torch import nn


class ActionChunkDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.horizon, self.action_dim = config['horizon'], config['action_dim']
        self.projection = nn.Linear(config['fusion_hidden'], self.horizon*self.action_dim)

    def forward(self, x):
        return self.projection(x).reshape(-1, self.horizon, self.action_dim)
