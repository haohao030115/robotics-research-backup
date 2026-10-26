import torch
from torch import nn


class VisionEncoder(nn.Module):
    """Spatial tokens shared by both policies; no pretrained downloads."""
    def __init__(self, config):
        super().__init__()
        channels = config['vision_channels']
        layers, previous = [], 3
        for channel in channels:
            layers += [nn.Conv2d(previous, channel, 3, stride=2, padding=1), nn.GELU()]
            previous = channel
        self.backbone = nn.Sequential(*layers)
        self.projection = nn.Conv2d(previous, config['d_model'], 1)
        side = config['image_size']
        for _ in channels:
            side = (side + 1)//2
        self.position = nn.Parameter(torch.zeros(1, side * side, config['d_model']))
        nn.init.normal_(self.position, std=0.02)
        self.image_size = config['image_size']

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f'image must be [B,3,{self.image_size},{self.image_size}]')
        features = self.projection(self.backbone(image))
        return features.flatten(2).transpose(1, 2) + self.position
