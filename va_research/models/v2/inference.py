"""Model-only RGB/state interface. No controller or scene dependencies."""
import numpy as np
import torch
import torch.nn.functional as F
from va_research.datasets.episodes import image_tensor
from utils.checkpoint import load_policy


def split_action_chunk(action_chunk):
    if action_chunk.shape[-1] != 43:
        raise ValueError('expected 43 action coordinates')
    return action_chunk[..., :29], action_chunk[..., 29:]


class ModelOnlyPolicy:
    def __init__(self, policy):
        self.policy = policy.eval()

    @classmethod
    def from_checkpoint(cls, path, device):
        policy, checkpoint = load_policy(path, device)
        return cls(policy), checkpoint

    def prepare(self, head_rgb, body_q, hand_q):
        device = self.policy.normalization.state_mean.device
        body = torch.as_tensor(body_q, dtype=torch.float32, device=device)
        hand = torch.as_tensor(hand_q, dtype=torch.float32, device=device)
        single = body.ndim == 1
        if single:
            body, hand = body[None], hand[None]
        if body.ndim != 2 or body.shape[1] != 29 or hand.shape != (len(body), 14):
            raise ValueError('current body_q/hand_q must be [29]/[14] or [B,29]/[B,14]')
        size = self.policy.config['image_size']
        if isinstance(head_rgb, np.ndarray):
            images = head_rgb[None] if head_rgb.ndim == 3 else head_rgb
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError('NumPy RGB layout must be HWC or BHWC')
            if images.dtype != np.uint8 and (not np.issubdtype(images.dtype, np.floating)
                    or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1):
                raise ValueError('NumPy RGB must be uint8 or float [0,1]')
            image = torch.stack([image_tensor(frame, size) for frame in images]).to(device)
        elif torch.is_tensor(head_rgb):
            image = head_rgb[None] if head_rgb.ndim == 3 else head_rgb
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError('Tensor RGB layout must be CHW or BCHW')
            was_uint8 = image.dtype == torch.uint8
            image = image.to(device=device, dtype=torch.float32)
            if was_uint8:
                image = image/255
            image = F.interpolate(image, size=(size, size), mode='bilinear', align_corners=False) \
                if image.shape[-2:] != (size, size) else image
            if not torch.isfinite(image).all().item() or image.min().item() < 0 or image.max().item() > 1:
                raise ValueError('Tensor RGB must be finite uint8 or float [0,1]')
        else:
            raise TypeError('head_rgb must be a NumPy array or PyTorch tensor')
        if len(image) != len(body) or not torch.isfinite(body).all().item() or not torch.isfinite(hand).all().item():
            raise ValueError('observation batch mismatch or non-finite current state')
        return image, body, hand, single

    @torch.inference_mode()
    def predict(self, head_rgb, body_q, hand_q, instruction=None, steps=None, initial_noise=None):
        image, body, hand, single = self.prepare(head_rgb, body_q, hand_q)
        if hasattr(self.policy, 'sample_normalized'):
            output = self.policy.predict(image, body, hand, instruction, steps, initial_noise)
        else:
            if steps is not None or initial_noise is not None:
                raise ValueError('Euler steps/noise apply only to the Flow policy')
            output = self.policy.predict(image, body, hand, instruction)
        if output.shape != (len(body), self.policy.config['horizon'], 43) or not torch.isfinite(output).all().item():
            raise RuntimeError('policy returned invalid trajectory')
        return output[0] if single else output
