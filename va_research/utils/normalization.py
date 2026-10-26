import torch
from torch import nn


class Normalization(nn.Module):
    def __init__(self, stats):
        super().__init__()
        for key, dim in [('image', 3), ('state', 43), ('action', 43)]:
            for kind in ('mean', 'std'):
                value = torch.as_tensor(stats[key][kind], dtype=torch.float32)
                if value.shape != (dim,) or not torch.isfinite(value).all():
                    raise ValueError(f'invalid {key} {kind}')
                if kind == 'std' and not (value > 0).all():
                    raise ValueError(f'{key} std must be positive')
                self.register_buffer(f'{key}_{kind}', value)

    def image(self, x):
        return (x - self.image_mean[None, :, None, None]) / self.image_std[None, :, None, None]

    def state(self, x):
        return (x - self.state_mean) / self.state_std

    def action(self, x):
        return (x - self.action_mean) / self.action_std

    def inverse_action(self, x):
        return x * self.action_std + self.action_mean
