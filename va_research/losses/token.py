"""Token-space regression and separately weighted physical hand-reference loss."""
import torch
import torch.nn.functional as F
from va_research.sonic.release import snap_token


def token_loss(output, token, hand, config):
    if output['sonic_token'].shape != token.shape or output['hand_ref'].shape != hand.shape:
        raise ValueError('VA V2 prediction/target shapes differ')
    if token.shape[-1] != 64 or hand.shape[-1] != 14:
        raise ValueError('Expected token64 and hand14')
    if config['token_loss'] == 'huber':
        body = F.huber_loss(output['sonic_token'], token, delta=config['huber_delta'])
    elif config['token_loss'] == 'mse':
        body = F.mse_loss(output['sonic_token'], token)
    else:
        raise ValueError('token_loss must be huber or mse')
    hands = F.mse_loss(output['hand_ref'], hand)
    return config['body_loss_weight']*body+config['hand_loss_weight']*hands, {'body_loss': body, 'hand_loss': hands}


class TokenMetrics:
    def __init__(self):
        self.count = 0
        self.token_abs = torch.zeros(64, dtype=torch.float64)
        self.token_sq = torch.zeros(64, dtype=torch.float64)
        self.correct = torch.zeros(64, dtype=torch.float64)
        self.hand_abs = torch.zeros(14, dtype=torch.float64)
        self.hand_sq = torch.zeros(14, dtype=torch.float64)

    def add(self, output, token, hand):
        predicted = output['sonic_token'].detach().cpu().double().reshape(-1, 64)
        target = token.detach().cpu().double().reshape(-1, 64)
        hp = output['hand_ref'].detach().cpu().double().reshape(-1, 14)
        ht = hand.detach().cpu().double().reshape(-1, 14)
        self.count += len(target)
        error = predicted-target
        self.token_abs += error.abs().sum(0); self.token_sq += error.square().sum(0)
        self.correct += (snap_token(predicted) == target).sum(0)
        self.hand_abs += (hp-ht).abs().sum(0); self.hand_sq += (hp-ht).square().sum(0)

    def report(self):
        if not self.count:
            return {'samples': 0}
        return {'samples': self.count, 'token_mae': float(self.token_abs.mean()/self.count),
                'token_mse': float(self.token_sq.mean()/self.count),
                'fsq_level_accuracy': float(self.correct.mean()/self.count),
                'token_mae_per_dimension': (self.token_abs/self.count).tolist(),
                'token_mse_per_dimension': (self.token_sq/self.count).tolist(),
                'fsq_accuracy_per_dimension': (self.correct/self.count).tolist(),
                'hand_mae': float(self.hand_abs.mean()/self.count), 'hand_mse': float(self.hand_sq.mean()/self.count),
                'hand_mae_per_dimension': (self.hand_abs/self.count).tolist()}
