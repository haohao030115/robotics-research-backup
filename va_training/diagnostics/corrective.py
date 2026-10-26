"""Isolated bounded corrective experiment utilities; no deployment modifications."""
import numpy as np
import torch
import torch.nn.functional as F
from datasets.episodes import image_tensor
from datasets.tokens import STATE_WIDTHS
from losses.token import TokenMetrics
from sonic.release import snap_token
from trainers.token_training import move
from diagnostics.phase import phase_match, quantiles


def masked_corrective_loss(output, token, hand, expert_count, branch, config):
    """Unselected head receives nominal supervision only, never learner labels."""
    body_end = len(token) if branch in ('body', 'combined') else expert_count
    hand_end = len(hand) if branch in ('hand', 'combined') else expert_count
    body = F.huber_loss(output['sonic_token'][:body_end], token[:body_end], delta=config['huber_delta'])
    hands = F.mse_loss(output['hand_ref'][:hand_end], hand[:hand_end])
    return config['body_loss_weight'] * body + config['hand_loss_weight'] * hands


def corrective_samples(path, image_size):
    z = np.load(path, allow_pickle=False)
    return [{'image': image_tensor(z['rgb'][i], image_size),
             'state': {k: torch.tensor(z[k][i], dtype=torch.float32) for k in STATE_WIDTHS},
             'history_mask': torch.tensor(z['history_mask'][i], dtype=torch.bool),
             'target_sonic_token': torch.tensor(z['target_sonic_token'][i], dtype=torch.float32),
             'target_hand_ref': torch.tensor(z['target_hand_ref'][i], dtype=torch.float32)}
            for i in range(len(z['capture_anchor']))]


@torch.inference_mode()
def metrics(policy, loader, device):
    policy.eval(); result = TokenMetrics(); exact = 0; chunks = 0
    for batch in loader:
        image, state, mask, token, hand = move(batch, device)
        output = policy(image, state, mask); result.add(output, token, hand)
        equals = (snap_token(output['sonic_token']) == token).all(-1)
        exact += int(equals.sum()); chunks += int(equals.all(-1).sum()) if equals.ndim == 2 else int(equals.sum())
    r = result.report()
    r['whole_token_exact_accuracy'] = exact / r['samples'] if r['samples'] else None
    r['whole_chunk_exact_accuracy'] = chunks / len(loader.dataset) if len(loader.dataset) else None
    r['input_samples'] = len(loader.dataset)
    return r

