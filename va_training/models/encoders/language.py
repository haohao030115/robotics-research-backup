import torch
from torch import nn


class InstructionEncoder(nn.Module):
    """Optional trainable UTF-8 byte embedding, suitable for short task labels.

    Not a pretrained semantic language model. Disabled for the initial single task.
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(257, config['d_model'], padding_idx=0)
        self.norm = nn.LayerNorm(config['d_model'])
        self.max_bytes = config['language_max_bytes']

    def forward(self, instructions, device):
        sequences = [[b+1 for b in text.encode('utf-8')[:self.max_bytes]] for text in instructions]
        ids = torch.zeros(len(sequences), self.max_bytes, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            if sequence:
                ids[i, :len(sequence)] = torch.tensor(sequence)
        ids = ids.to(device)
        mask = (ids != 0).to(self.embedding.weight.dtype)[..., None]
        pooled = (self.embedding(ids)*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.norm(pooled)[:, None]
