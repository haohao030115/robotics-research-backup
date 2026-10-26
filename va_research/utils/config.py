import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config['data']['manifest'] is None:
        raise ValueError('dataset manifest is required; real dataset has not been uploaded')
    config = copy.deepcopy(config)
    for section, field in [('data', 'manifest'), ('training', 'output_dir')]:
        value = Path(config[section][field])
        config[section][field] = str((ROOT/value).resolve()) if not value.is_absolute() else str(value)
    if config['model']['action_dim'] != 43 or config['model']['horizon'] < 1:
        raise ValueError('expected positive H and action_dim=43')
    training = config['training']
    for key in ['epochs', 'max_steps', 'batch_size_per_rank', 'log_every', 'checkpoint_every_epochs']:
        if training[key] < 1:
            raise ValueError(f'{key} must be positive')
    if training['body_loss_weight'] <= 0 or training['hand_loss_weight'] <= 0:
        raise ValueError('loss weights must be positive')
    return config


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)
