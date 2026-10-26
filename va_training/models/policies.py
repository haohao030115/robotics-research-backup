import torch
from torch import nn
from .encoders.observation import ObservationEncoder
from .fusion.baseline import FusionMLP
from .decoders.chunk import ActionChunkDecoder
from .decoders.dit import TrajectoryDiT
from .flow_scheduler import FlowScheduler
from utils.normalization import Normalization


class Policy(nn.Module):
    def __init__(self, config, stats):
        super().__init__()
        if config['action_dim'] != 43:
            raise ValueError('action_dim must be 29 body + 14 hand = 43')
        self.config = config
        self.normalization = Normalization(stats)
        self.observation = ObservationEncoder(config)

    def encode(self, image, q, instructions=None):
        return self.observation(self.normalization.image(image), self.normalization.state(q), instructions)

    def inputs(self, image, body_q, hand_q):
        device = self.normalization.state_mean.device
        image = torch.as_tensor(image, device=device, dtype=torch.float32)
        body_q = torch.as_tensor(body_q, device=device, dtype=torch.float32)
        hand_q = torch.as_tensor(hand_q, device=device, dtype=torch.float32)
        if body_q.ndim != 2 or body_q.shape[1] != 29 or hand_q.shape != (len(body_q), 14):
            raise ValueError('body_q [B,29], hand_q [B,14] required')
        return image, torch.cat((body_q, hand_q), dim=1)


class BaselinePolicy(Policy):
    def __init__(self, config, stats):
        super().__init__(config, stats)
        self.fusion, self.decoder = FusionMLP(config), ActionChunkDecoder(config)

    def forward(self, image, q, instructions=None):
        visual, state, context = self.encode(image, q, instructions)
        # The baseline conditions on the optional language token if enabled.
        if self.observation.language is not None:
            state = state + context[:, -1:]
        features = torch.cat((visual.mean(1), state[:, 0]), dim=-1)
        return self.decoder(self.fusion(features))

    @torch.no_grad()
    def predict(self, image, body_q, hand_q, instruction=None):
        previous = self.training
        self.eval()
        try:
            image, q = self.inputs(image, body_q, hand_q)
            return self.normalization.inverse_action(self(image, q, instruction))
        finally:
            self.train(previous)


class FlowPolicy(Policy):
    def __init__(self, config, stats):
        super().__init__(config, stats)
        self.dit, self.scheduler = TrajectoryDiT(config), FlowScheduler()

    def forward(self, image, q, xt, t, instructions=None):
        _, _, context = self.encode(image, q, instructions)
        return self.dit(xt, t, context)

    @torch.no_grad()
    def sample_normalized(self, image, q, instructions=None, steps=None, initial_noise=None):
        _, _, context = self.encode(image, q, instructions)
        return self.scheduler.integrate(lambda x, t: self.dit(x, t, context),
            (len(q), self.config['horizon'], 43), q.device, q.dtype,
            self.config['sampling_steps'] if steps is None else steps, initial_noise)

    @torch.no_grad()
    def predict(self, image, body_q, hand_q, instruction=None, steps=None, initial_noise=None):
        previous = self.training
        self.eval()
        try:
            image, q = self.inputs(image, body_q, hand_q)
            normalized = self.sample_normalized(image, q, instruction, steps, initial_noise)
            return self.normalization.inverse_action(normalized)
        finally:
            self.train(previous)


def build_policy(config, stats):
    if config['model_type'] == 'token_v2':
        from .token_v2 import TokenV2Policy
        return TokenV2Policy(config['model'], stats)
    if config['model_type'] == 'baseline':
        return BaselinePolicy(config['model'], stats)
    if config['model_type'] == 'flow':
        return FlowPolicy(config['model'], stats)
    raise ValueError('model_type must be baseline or flow')
