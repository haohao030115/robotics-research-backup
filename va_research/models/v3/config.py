import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    c = json.loads(Path(path).read_text())
    if c['model_type'] != 'token_v3' or c['timing'] != dict(control_hz=50, inference_hz=25, stride=2, prediction_horizon=2, execution_horizon=2, post_step_target_offset=1):
        raise ValueError('Invalid V3 timing contract')
    expected=dict(image_size=128,proprio_dim=96,embedding_dim=256,heads=4,cross_attention_blocks=1,shared_dim=512,token_chunk_size=2)
    if c['model']!=expected or c['rollout']['project_token'] is not False:
        raise ValueError('V3 uses the fixed spatial cross-attention baseline and raw regressed tokens')
    def resolve(p):
        return str((ROOT/p).resolve())
    c['data']['manifests'] = [resolve(p) for p in c['data']['manifests']]
    c['sonic'] = {k: resolve(v) for k, v in c['sonic'].items()}
    if c['data']['control_hz'] != 50 or c['rollout']['visual_hz'] != 25:
        raise ValueError('V3 requires 50-Hz control and 25-Hz observations')
    return c
