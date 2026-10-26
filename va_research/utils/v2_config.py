"""Separate VA V2 configuration contract; V1 configs/checkpoints are untouched."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_v2_config(path):
    config = copy.deepcopy(json.loads(Path(path).read_text()))
    if config.get('model_type') != 'token_v2':
        raise ValueError('Expected model_type=token_v2')
    def resolve(value):
        p = Path(value).expanduser()
        return str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    config['data']['manifests'] = [resolve(p) for p in config['data']['manifests']]
    for key in ('repository', 'encoder', 'decoder', 'observation_config', 'robot_xml', 'binary'):
        config['sonic'][key] = resolve(config['sonic'][key])
    config['training']['output_dir'] = resolve(config['training']['output_dir'])
    for key in ('history_length', 'token_chunk_size', 'image_size'):
        if config['model'][key] < 1:
            raise ValueError(f'Invalid {key}')
    if config['data']['target_offset'] < 1:
        raise ValueError('V2 post-step snapshots require causal target_offset>=1')
    if config['data']['control_hz'] != 50:
        raise ValueError('Frozen native release deployment executes at 50 Hz')
    ratio = config['data']['control_hz']/config['rollout']['visual_hz']
    if ratio < 1 or ratio != int(ratio):
        raise ValueError('visual_hz must divide control_hz')
    if config['rollout']['physics_hz'] % config['data']['control_hz']:
        raise ValueError('physics_hz must be a multiple of control_hz')
    t = config['training']
    if min(t['epochs'], t['max_steps'], t['batch_size_per_rank'], t['body_loss_weight'], t['hand_loss_weight']) <= 0:
        raise ValueError('Invalid training limits/weights')
    return config
