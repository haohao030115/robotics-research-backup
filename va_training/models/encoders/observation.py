import torch
from torch import nn
from .vision import VisionEncoder
from .state import StateEncoder
from .language import InstructionEncoder


class ObservationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision = VisionEncoder(config)
        self.state = StateEncoder(config)
        self.language = InstructionEncoder(config) if config['language_enabled'] else None

    def forward(self, image, q, instructions=None):
        visual = self.vision(image)
        state = self.state(q)
        context = [visual, state]
        if self.language is not None:
            if instructions is None:
                instructions = [''] * len(q)
            if isinstance(instructions, str):
                instructions = [instructions] * len(q)
            if len(instructions) != len(q):
                raise ValueError('instruction batch size mismatch')
            context.append(self.language(instructions, q.device))
        return visual, state, torch.cat(context, dim=1)
