import torch


def compute_statistics(dataset, minimum_std=1e-3):
    """Train-only statistics, after image resizing and before normalization."""
    totals = {k: [None, None, 0] for k in ('image', 'state', 'action')}
    for sample in dataset:
        values = {'image': sample['image'].permute(1, 2, 0).reshape(-1, 3),
                  'state': sample['q'][None], 'action': sample['target']}
        for key, value in values.items():
            value = value.double()
            total, square, count = totals[key]
            total = value.sum(0) if total is None else total + value.sum(0)
            square = value.square().sum(0) if square is None else square + value.square().sum(0)
            totals[key] = [total, square, count + len(value)]
    result = {'train_episode_ids': [e.id for e in dataset.episodes],
              'minimum_std': minimum_std, 'image_scale': 'uint8/255 or float[0,1]'}
    for key, (total, square, count) in totals.items():
        if count == 0:
            raise ValueError('cannot compute statistics from empty training data')
        mean = total/count
        std = (square/count - mean.square()).clamp_min(0).sqrt().clamp_min(minimum_std)
        result[key] = {'mean': mean.float().tolist(), 'std': std.float().tolist(), 'count': count}
        if key in ('state', 'action'):
            result[key]['body'] = {'mean': mean[:29].tolist(), 'std': std[:29].tolist()}
            result[key]['hand'] = {'mean': mean[29:].tolist(), 'std': std[29:].tolist()}
    return result
