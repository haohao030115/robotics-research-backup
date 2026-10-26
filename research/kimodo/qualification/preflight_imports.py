"""CPU/import-only preflight, no checkpoint loading or model generation."""
import json,sys
from pathlib import Path
import torch,mujoco
import prepare_grasp_task,export_kimodo_reference,replay_validate_kimodo
r=Path(__file__).resolve().parents[1]
c=json.loads((r/'config/frozen_config.json').read_text())
from kimodo.skeleton import G1Skeleton34
from kimodo.exports.mujoco import MujocoQposConverter
v=MujocoQposConverter(G1Skeleton34())
assert Path(v.xml_path).is_file()
for p in (r/'assets/text_encoders').rglob('adapter_config.json'):
 b=json.loads(p.read_text())['base_model_name_or_path'];assert Path(b).is_dir() and '/ardy/' not in b
import run_sonic_grasp
m,d=prepare_grasp_task.load_scene(supported=False,robot_path=prepare_grasp_task.ROBOT)
assert (m.nq,m.nv,m.nu,m.neq)==(57,55,43,0)
print('Preflight imports/assets resolved without model loading; MuJoCo',mujoco.__version__)
