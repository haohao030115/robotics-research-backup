#!/usr/bin/env python3
"""Add schema fields to existing episode metadata without rewriting any arrays."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scene import load_scene
from episode_schema import dataset_fields
from expert_trajectory import write_json


def standardize(episode):
    meta = json.loads((episode/'metadata.json').read_text())
    model, data = load_scene(root_height=meta['source_root_height_m'])
    data.qpos[:] = np.load(episode/'source/qpos.npy')[0]
    mujoco.mj_forward(model, data)
    fields = dataset_fields(model, data, meta['case'], episode.name,
                            meta.get('source_arguments', {}).get('random_seed'))
    files = [episode/'metadata.json', episode/'sonic_reference/metadata.json']
    files += list((episode/'candidates').glob('*/metadata.json'))
    files += list((episode/'validation').glob('*/metadata.json'))
    for path in files:
        if not path.exists():continue
        original = json.loads(path.read_text())
        original.update(fields)
        original.setdefault('initial_pose', {'name':'legacy_tablefront_preparation',
                                            'neutral_standard_compliant':False})
        write_json(path, original)
    return fields


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(standardize(args.episode),indent=2))
