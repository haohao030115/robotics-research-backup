from torch.nn import functional as F


def objectives(pred, target, weights=(1., 1., .1, .1)):
    terms = dict(body=F.smooth_l1_loss(pred['body_tokens'], target['body_tokens']),
                 hand=F.mse_loss(pred['hand_ref'], target['hand_ref']),
                 phase=F.cross_entropy(pred['phase_logits'], target['phase']),
                 progress=F.mse_loss(pred['closure_progress'], target['progress']))
    return sum(w * v for w, v in zip(weights, terms.values())), terms
