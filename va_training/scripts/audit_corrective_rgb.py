"""Independently verify every prototype RGB against its learner physics snapshot."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sonic.release import digest
from datasets.tokens import projected_gravity
from utils.config import dump_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prototype',required=True);p.add_argument('--rollout',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    source=Path(args.rollout);proto=Path(args.prototype);r=json.loads((source/'report.json').read_text());config=r['provenance']['va_config']
    sys.path.insert(0,str(Path(config['sonic']['repository'])));sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'dl-group2/scripts'))
    import scene,mujoco
    epath=Path(r['provenance']['dataset']['token_label_source']).parents[3];meta=json.loads((epath/'metadata.json').read_text())
    model,data=scene.load_scene(supported=False,robot_path=Path(config['sonic']['robot_xml']),root_height=meta.get('source_root_height_m'))
    z=np.load(proto/'corrective_samples.npz');physics=np.load(source/'physics.npz');renderer=mujoco.Renderer(model,height=240,width=320)
    maxima=[]
    try:
        for i,anchor in enumerate(z['capture_anchor']):
            data.qpos[:]=physics['qpos'][anchor];data.qvel[:]=physics['qvel'][anchor]
            mujoco.mj_forward(model,data);renderer.update_scene(data,camera='head_rgb');image=renderer.render()
            maxima.append(int(abs(image.astype(int)-z['rgb'][i].astype(int)).max()))
            np.testing.assert_array_equal(z['history_mask'][i],np.arange(anchor-2,anchor+1)>=0)
            np.testing.assert_allclose(z['projected_gravity'][i],projected_gravity(z['base_quat'][i]),atol=1e-12)
    finally:renderer.close()
    if max(maxima)!=0:raise ValueError(f'RGB source reconstruction mismatch: {max(maxima)}')
    dump_json(args.output,{'rgb_rows':len(maxima),'all_rgb_exact':True,'max_pixel_abs':max(maxima),
        'history_masks_verified':True,'derived_gravity_verified':True,'source_rollout':str(source),
        'prototype_sha256':digest(proto/'corrective_samples.npz'),'physics_sha256':digest(source/'physics.npz')})
    print('RGB source audit passed',len(maxima),'exact frames',flush=True)


if __name__=='__main__':main()
