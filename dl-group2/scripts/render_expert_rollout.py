#!/usr/bin/env python3
"""Render saved actual states on their recorded clock without advancing physics."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from scene import load_scene
from episode_schema import CAMERA_ROLE

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--rollout',default='source');p.add_argument('--stride',type=int,default=5);args=p.parse_args()
    import cv2
    ep=args.episode;folder=ep/args.rollout;out=folder/'vision'
    if out.exists():p.error('vision exists')
    out.mkdir();meta=json.loads((ep/'metadata.json').read_text());m,d=load_scene(root_height=meta['source_root_height_m'])
    q=np.load(folder/'qpos.npy');v=np.load(folder/'qvel.npy');ts=np.load(folder/'timestamps.npy')
    frame_path=folder/'reference_frame.npy'
    frames=np.load(frame_path) if frame_path.exists() else np.arange(len(q))
    sim_path=folder/'sim_timestamps.npy'
    sim_ts=np.load(sim_path) if sim_path.exists() else ts
    renderers={c:mujoco.Renderer(m,height=240,width=320) for c in ['head_rgb','wrist_rgb','task_overview']}
    writers={c:cv2.VideoWriter(str(out/f'{c}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50/args.stride,(320,240)) for c in renderers}
    pixels={c:[] for c in renderers};ix=np.arange(0,len(q),args.stride)
    try:
        for f in ix:
            d.qpos[:]=q[f];d.qvel[:]=v[f];mujoco.mj_forward(m,d)
            for c,r in renderers.items():
                r.update_scene(d,camera=c);rgb=r.render().copy();pixels[c].append(rgb)
                writers[c].write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        np.savez_compressed(out/'images.npz',**{c:np.asarray(a) for c,a in pixels.items()},reference_frame=frames[ix],timestamps=ts[ix],sim_timestamps=sim_ts[ix])
        (out/'metadata.json').write_text(json.dumps({'rendered_from':'saved actual qpos/qvel; no physics stepping', 'image_shape':[len(ix),240,320,3], 'color_order':'RGB','timestamps':'simulation/reference clock', 'state_frame_indices':ix.tolist(), 'reference_frame_indices':frames[ix].tolist(), 'camera_role':CAMERA_ROLE},indent=2))
    finally:
        for r in renderers.values():r.close()
        for w in writers.values():w.release()
    print(out)
if __name__=='__main__':main()
