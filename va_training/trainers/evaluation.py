import torch
import torch.distributed as dist
from losses.action import action_loss


class Metrics:
    def __init__(self):
        self.totals = {}

    def add(self, values, batch):
        for key, value in values.items():
            value = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            old, count = self.totals.get(key, (0.0, 0))
            self.totals[key] = (old + value*batch, count + batch)

    def result(self, runtime, keys=None):
        keys = sorted(self.totals) if keys is None else keys
        packed = torch.tensor([self.totals.get(k, (0.0, 0)) for k in keys],
                              dtype=torch.float32, device=runtime.device).reshape(-1, 2)
        if runtime.distributed:
            dist.all_reduce(packed)
        return {key: float(row[0]/row[1]) for key, row in zip(keys, packed.cpu().tolist()) if row[1] > 0}


@torch.no_grad()
def evaluate(policy, loader, config, runtime):
    policy.eval()
    totals = Metrics()
    is_flow = config['model_type'] == 'flow'
    for i, batch in enumerate(loader):
        image, q, target = [batch[k].to(runtime.device) for k in ['image', 'q', 'target']]
        normalized = policy.normalization.action(target)
        if is_flow:
            xt, t, velocity = policy.scheduler.training_pair(normalized)
            predicted = policy(image, q, xt, t, batch['instruction'])
            loss, parts = action_loss(predicted, velocity, 'mse')
            totals.add({'loss': loss, 'flow_matching_loss': loss,
                        'body_velocity_mse': parts['body'], 'hand_velocity_mse': parts['hand']}, len(q))
            if i >= config['training']['validation_sampling_batches']:
                continue
            output = policy.sample_normalized(image, q, batch['instruction'])
        else:
            output = policy(image, q, batch['instruction'])
            loss, _ = action_loss(output, normalized)
            totals.add({'loss': loss}, len(q))
        physical = policy.normalization.inverse_action(output)
        _, parts = action_loss(physical, target)
        totals.add({'action_l1': parts['total'], 'body_l1': parts['body'], 'hand_l1': parts['hand']}, len(q))
    keys = ['loss', 'action_l1', 'body_l1', 'hand_l1']
    if is_flow:
        keys += ['flow_matching_loss', 'body_velocity_mse', 'hand_velocity_mse']
    return totals.result(runtime, keys)
