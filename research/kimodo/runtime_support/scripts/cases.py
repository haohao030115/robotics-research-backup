"""JSON case selection and deterministic full-state reset."""
import json
from pathlib import Path
import re
import mujoco
import numpy as np
from project_paths import CONFIGS

DEFAULT_CONFIG = CONFIGS / 'grasp_cases.json'


def read_cases(path=DEFAULT_CONFIG):
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    cases=config['cases']
    if not cases or len({c['id'] for c in cases})!=len(cases):
        raise ValueError('Cases must have unique nonempty IDs')
    for case in cases:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',case['id']):raise ValueError('Unsafe case ID')
        if len(case['block_xy'])!=2 or not np.isfinite(case['block_xy']).all():
            raise ValueError('block_xy must contain two finite numbers')
        if not np.isfinite(case.get('block_yaw_deg',0)):raise ValueError('Invalid block yaw')
    for key in ('state_hz','image_hz','image_width','image_height','repeats'):
        if not isinstance(config[key],int) or config[key]<=0:raise ValueError(f'Invalid {key}')
    if not np.isfinite(config['duration_seconds']) or config['duration_seconds']<=0:
        raise ValueError('Invalid duration_seconds')
    return config


def case_index(config, name):
    ids=[c['id'] for c in config['cases']]
    if name in ids:return ids.index(name)
    if str(name).isdigit() and 1<=int(name)<=len(ids):return int(name)-1
    raise ValueError(f'Unknown case {name}; choose 1..{len(ids)} or {ids}')


def reset_case(model, data, case):
    """Only reset may reposition the cube; rollouts never teleport it."""
    mujoco.mj_resetData(model,data)
    mujoco.mj_forward(model,data)
    table=model.geom('task_table_top').id
    cube=model.geom('task_cube_geom').id
    xy=np.asarray(case['block_xy'],float)
    yaw=np.deg2rad(case.get('block_yaw_deg',0))
    # This environment has a horizontal box table. Check the rotated cube footprint.
    size=model.geom_size[cube]
    footprint=np.array([[abs(np.cos(yaw)),abs(np.sin(yaw))],
                        [abs(np.sin(yaw)),abs(np.cos(yaw))]])@size[:2]
    if np.any(abs(xy-data.geom_xpos[table,:2])+footprint>model.geom_size[table,:2]):
        raise ValueError(f"Case {case['id']} places cube beyond the tabletop")
    top=data.geom_xpos[table,2]+model.geom_size[table,2]
    addr=model.joint('task_cube_free').qposadr[0]
    data.qpos[addr:addr+3]=[*xy,top+size[2]]
    data.qpos[addr+3:addr+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    mujoco.mj_forward(model,data)
