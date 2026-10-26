import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'dl-group2'
SONIC = ROOT / 'GR00T-WholeBodyControl'
ROBOT = SONIC / 'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml'
BUNDLE = SONIC / 'gear_sonic_deploy/policy/release'
CPP = SONIC / 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref'
HASHES = {'model_encoder.onnx': '013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3',
          'model_decoder.onnx': 'c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed',
          'observation_config.yaml': '466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def read(path):
    return json.loads(Path(path).read_text())

def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]

def jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def table(path, rows):
    if not rows:
        raise ValueError('Empty truth table: '+str(path))
    np.savez_compressed(path, **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})

def require(condition, message):
    if not condition:
        raise ValueError(message)

def index(table, key):
    values = table[key].tolist()
    require(len(values) == len(set(values)), 'Duplicate '+key)
    return {v: i for i, v in enumerate(values)}

def snapshot_hash(qpos, qvel, ctrl, sim_time):
    h = hashlib.sha256()
    for a in (qpos, qvel, ctrl, [sim_time]):
        h.update(np.asarray(a, dtype='<f8').tobytes())
    return h.hexdigest()
