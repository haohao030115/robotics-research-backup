"""Model-only VA V2 interface, using the same image preprocessing as V1."""
import numpy as np
import torch
from datasets.episodes import image_tensor
from datasets.tokens import STATE_WIDTHS, projected_gravity
from utils.checkpoint import load_policy
from sonic.release import snap_token


class TokenModelPolicy:
    def __init__(self, policy):
        self.policy = policy.eval()

    @classmethod
    def from_checkpoint(cls, path, device):
        policy, checkpoint = load_policy(path, device)
        if checkpoint['config']['model_type'] != 'token_v2':
            raise ValueError('Expected a VA V2 checkpoint')
        return cls(policy), checkpoint

    def prepare(self, rgb, state, history_mask):
        device = self.policy.image_mean.device
        mask = torch.as_tensor(history_mask, dtype=torch.bool, device=device)
        single = mask.ndim == 1
        if single:
            mask = mask[None]
        if mask.ndim != 2 or mask.shape[1] != self.policy.config['history_length'] or not mask[:, -1].all():
            raise ValueError('Expected history_mask [L]/[B,L] with valid current state')
        source = dict(state)
        if 'projected_gravity' not in source:
            q = source['base_quat']
            if torch.is_tensor(q):
                q = q.detach().cpu().numpy()
            source['projected_gravity'] = projected_gravity(q)
        if set(source) != set(STATE_WIDTHS):
            raise ValueError('V2 requires all named state fields')
        values = {}
        for key, width in STATE_WIDTHS.items():
            value = torch.as_tensor(source[key], dtype=torch.float32, device=device)
            if single:
                value = value[None]
            if value.shape != (*mask.shape, width) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid {key} history')
            values[key] = value
        q = values['base_quat']
        if not torch.allclose(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=1e-5, rtol=0):
            raise ValueError('Non-unit base quaternion')
        if isinstance(rgb, np.ndarray):
            images = rgb[None] if rgb.ndim == 3 else rgb
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError('NumPy RGB must be HWC/BHWC')
            if images.dtype != np.uint8 and (not np.isfinite(images).all() or images.min() < 0 or images.max() > 1):
                raise ValueError('RGB must be uint8 or float [0,1]')
            image = torch.stack([image_tensor(v, self.policy.config['image_size']) for v in images]).to(device)
        else:
            image = torch.as_tensor(rgb, device=device)
            if image.ndim == 3:
                image = image[None]
            if image.ndim != 4 or image.shape[1:] != (3, self.policy.config['image_size'], self.policy.config['image_size']):
                raise ValueError('Tensor image must be pre-resized CHW/BCHW')
            image = image.float()/255 if image.dtype == torch.uint8 else image.float()
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('RGB must be finite [0,1]')
        if len(image) != len(mask):
            raise ValueError('Observation batch mismatch')
        return image, values, mask, single

    @torch.inference_mode()
    def predict(self, rgb, state, history_mask, project=False):
        image, values, mask, single = self.prepare(rgb, state, history_mask)
        result = self.policy.predict(image, values, mask)
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise RuntimeError('Non-finite VA V2 output')
        if project:
            result['sonic_token'] = snap_token(result['sonic_token'])
        return {k: v[0] if single else v for k, v in result.items()}
