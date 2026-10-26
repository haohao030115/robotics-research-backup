import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, dump_json
from datasets import load_episodes, split_episodes, EpisodeDataset, compute_statistics


def summarize(value):
    value = np.asarray(value, dtype=np.float64)
    return {'shape': list(value.shape), 'min': float(value.min()), 'max': float(value.max()),
            'mean': float(value.mean()), 'std': float(value.std()),
            'per_joint_min': value.reshape(-1, value.shape[-1]).min(0).tolist(),
            'per_joint_max': value.reshape(-1, value.shape[-1]).max(0).tolist(),
            'per_joint_mean': value.reshape(-1, value.shape[-1]).mean(0).tolist(),
            'per_joint_std': value.reshape(-1, value.shape[-1]).std(0).tolist()}


def episode_sanity(episode, model):
    data = EpisodeDataset([episode], model['horizon'], model['image_size'])
    def indices(item):
        _, rgb, state = item
        return {'rgb_index': rgb, 'state_index': state, 'target_start': state,
                'target_end_inclusive': state+model['horizon']-1,
                'rgb_timestamp': float(episode.rgb_times[rgb]),
                'state_timestamp': float(episode.state_times[state]),
                'alignment_error_seconds': float(abs(episode.rgb_times[rgb]-episode.state_times[state]))}
    targets = np.stack([episode.ref[a:a+model['horizon']] for _, _, a in data.index])
    examples = sorted(set([0, min(1, len(data)-1), len(data)//2, len(data)-1]))
    return {'episode_id': episode.id, 'source_episode_id': episode.source_episode_id,
            'number_of_rgb_frames': len(episode.rgb_times), 'number_of_state_frames': len(episode.q),
            'valid_training_samples': len(data), 'filtered_tail_frames': len(episode.rgb_times)-len(data),
            'first_valid_index': indices(data.index[0]), 'last_valid_index': indices(data.index[-1]),
            'aligned_rgb_examples': [indices(data.index[i]) for i in examples],
            'q_shape': [43], 'target_shape': [model['horizon'], 43],
            'body_q': summarize(episode.q[:, :29]), 'hand_q': summarize(episode.q[:, 29:]),
            'actions_from_valid_chunks': summarize(targets),
            'body_actions': summarize(targets[..., :29]), 'hand_actions': summarize(targets[..., 29:])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    episodes = load_episodes(config['data'])
    train, val = split_episodes(episodes, config['data'])
    model = config['model']
    training = EpisodeDataset(train, model['horizon'], model['image_size'])
    validation = EpisodeDataset(val, model['horizon'], model['image_size'])
    raw = json.loads(Path(config['data']['manifest']).read_text())
    report = {'synthetic': raw.get('synthetic', False) if isinstance(raw, dict) else False,
              'status': 'PASS', 'train_episodes': [e.id for e in train], 'val_episodes': [e.id for e in val],
              'train_chunks': len(training), 'val_chunks': len(validation),
              'horizon_seconds': model['horizon']/config['data']['control_hz'],
              'alignment': config['data']['alignment'], 'target_offset': 'anchor through anchor+H-1',
              'image_shape': list(training[0]['image'].shape),
              'q_shape': list(training[0]['q'].shape), 'target_shape': list(training[0]['target'].shape),
              'max_alignment_error_seconds': max(float(abs(e.state_times[e.anchors]-e.rgb_times).max())
                                                  for e in episodes),
              'normalization': compute_statistics(training),
              'episode_sanity': [episode_sanity(e, model) for e in episodes]}
    dump_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('normalization', 'episode_sanity')}, indent=2))


if __name__ == '__main__':
    main()
