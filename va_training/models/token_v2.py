"""VA V2: shared V1 vision, structured state history, separate native-token/hand heads."""
import torch
from torch import nn
from .encoders.vision import VisionEncoder
from .encoders.structured_state import StructuredStateEncoder
from .fusion.baseline import FusionMLP
from datasets.tokens import STATE_WIDTHS


class TokenV2Policy(nn.Module):
    def __init__(self, config, stats):
        super().__init__()
        self.config = config
        self.vision = VisionEncoder(config)
        self.state = StructuredStateEncoder(config)
        self.fusion = FusionMLP(config)
        self.k = config['token_chunk_size']
        self.body_head = nn.Linear(config['fusion_hidden'], self.k*64)
        self.hand_head = nn.Linear(config['fusion_hidden'], self.k*14)
        for prefix, values, dim in [('image', stats['image'], 3), ('hand', stats['hand'], 14)]:
            self._stats(prefix, values, dim)
        for name, dim in STATE_WIDTHS.items():
            self._stats('state_'+name, stats['state'][name], dim)

    def _stats(self, prefix, values, dim):
        for kind in ('mean', 'std'):
            value = torch.as_tensor(values[kind], dtype=torch.float32)
            if value.shape != (dim,) or not torch.isfinite(value).all() or (kind == 'std' and not (value > 0).all()):
                raise ValueError(f'Invalid {prefix} {kind} statistics')
            self.register_buffer(prefix+'_'+kind, value)

    def forward(self, image, state, history_mask):
        visual = self.vision((image-self.image_mean[None, :, None, None])/self.image_std[None, :, None, None])
        normalized = {k: (v-getattr(self, 'state_'+k+'_mean'))/getattr(self, 'state_'+k+'_std') for k, v in state.items()}
        features = self.fusion(torch.cat([visual.mean(1), self.state(normalized, history_mask)], -1))
        token = self.body_head(features).reshape(-1, self.k, 64)
        hand = self.hand_head(features).reshape(-1, self.k, 14)*self.hand_std+self.hand_mean
        if self.k == 1:
            token, hand = token[:, 0], hand[:, 0]
        return {'sonic_token': token, 'hand_ref': hand}

    @torch.inference_mode()
    def predict(self, image, state, history_mask):
        previous = self.training
        self.eval()
        try:
            return self(image, state, history_mask)
        finally:
            self.train(previous)
