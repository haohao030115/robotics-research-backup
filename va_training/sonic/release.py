"""Validate and describe the frozen release bundle and reuse its v4 transport."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bundle_provenance(config):
    import onnx
    import yaml
    result = {'variant': 'release', 'frozen': True}
    expected = {'encoder': ([1, 1762], [1, 64]), 'decoder': ([1, 994], [1, 29])}
    for name, shapes in expected.items():
        path = Path(config[name]).resolve()
        model = onnx.load(str(path))
        actual = [[d.dim_value for d in value.type.tensor_type.shape.dim]
                  for value in (model.graph.input[0], model.graph.output[0])]
        if len(model.graph.input) != 1 or len(model.graph.output) != 1 or actual != list(shapes):
            raise ValueError(f'{name}: expected release shapes {shapes}, got {actual}')
        if any(v.type.tensor_type.elem_type != 1 for v in (model.graph.input[0], model.graph.output[0])):
            raise ValueError('SONIC release I/O must be float32')
        result[name] = {'path': str(path), 'sha256': digest(path), 'shapes': actual}
    path = Path(config['observation_config']).resolve()
    obs = yaml.safe_load(path.read_text())
    g1 = next(m for m in obs['encoder']['encoder_modes'] if m['mode_id'] == 0)
    required = ['encoder_mode_4', 'motion_joint_positions_10frame_step5',
                'motion_joint_velocities_10frame_step5', 'motion_anchor_orientation_10frame_step5']
    if g1['required_observations'] != required or obs['encoder']['dimension'] != 64:
        raise ValueError('Expected release full-base-orientation G1 configuration')
    result['observation_config'] = {'path': str(path), 'sha256': digest(path)}
    result['repository'] = str(Path(config['repository']).resolve())
    return result


def snap_token(token):
    """Release FSQ has 32 levels: -16/16 ... 15/16. Optional inference ablation."""
    if hasattr(token, 'clamp'):
        return (token * 16).round().clamp(-16, 15) / 16
    return np.clip(np.rint(np.asarray(token) * 16), -16, 15) / 16


class ExternalTokenAdapter:
    """Reuse SONIC's protocol-v4 packer; native deployment owns decoder history/PD."""
    def __init__(self, repository):
        path = Path(repository) / 'gear_sonic/utils/teleop/zmq/zmq_planner_sender.py'
        spec = importlib.util.spec_from_file_location('_sonic_v2_packer', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.pack = module.pack_pose_message

    def message(self, token, execution_index, project=False):
        token = np.asarray(token, dtype=np.float32)
        if token.shape != (64,) or not np.isfinite(token).all():
            raise ValueError('Expected finite token [64]')
        if project:
            token = snap_token(token)
        return self.pack({'token_state': token.reshape(1, 64),
                          'frame_index': np.array([execution_index], dtype=np.int64)},
                         topic='pose', version=4)


def require_pass(path, field, provenance=None):
    report = json.loads(Path(path).read_text())
    if report.get(field) is not True:
        raise ValueError(f'Gate {path}: {field} is not true; diagnose before proceeding')
    if provenance is not None and report.get('provenance', {}).get('sonic') != provenance:
        raise ValueError('Gate was produced with a different SONIC bundle')
    return report
