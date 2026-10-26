import torch


def action_loss(prediction, target, kind='l1', body_weight=1.0, hand_weight=1.0):
    if prediction.shape != target.shape or target.shape[-1] != 43:
        raise ValueError('prediction/target must share [B,H,43] shape')
    error = prediction-target
    if kind == 'l1':
        error = error.abs()
    elif kind == 'mse':
        error = error.square()
    else:
        raise ValueError('kind must be l1 or mse')
    body, hand = error[..., :29].mean(), error[..., 29:].mean()
    total = (29*body_weight*body + 14*hand_weight*hand)/43
    return total, {'total': error.mean(), 'body': body, 'hand': hand}
