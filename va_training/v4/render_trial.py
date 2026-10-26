"""Render one native rollout's saved MuJoCo integration states to MP4.

The frames come only from the actual learned-policy rollout physics.npz.  No
expert state, replayed target, or synthetic trajectory is inserted.
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rollout',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    import imageio.v2 as imageio
    from canonical_va.native_scene import SnapshotReader
    run=Path(a.rollout); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with np.load(run/'physics.npz',allow_pickle=False) as z:
        states=z['integration_state']; ticks=z['frame']
    reader=SnapshotReader(render=True)
    from OpenGL import GL
    vendor=GL.glGetString(GL.GL_VENDOR).decode(); renderer=GL.glGetString(GL.GL_RENDERER).decode()
    proof=dict(gl_vendor=vendor, gl_renderer=renderer, hardware_verified=('Moore Threads' in vendor and 'MTT S4000' in renderer))
    if not proof['hardware_verified']:
        reader.close(); raise RuntimeError('Refusing video render without Moore Threads MTT S4000 renderer: '+json.dumps(proof))
    frames=[]
    try:
        with imageio.get_writer(out,fps=10,codec='libx264',quality=8,macro_block_size=1) as writer:
            for i in range(0,len(states),5):
                reader.restore(states[i]); frame=reader.render(); writer.append_data(frame); frames.append(int(ticks[i]))
    finally: reader.close()
    (out.parent/'video_renderer.json').write_text(json.dumps(proof,indent=2)+'\n')
    meta=dict(path=str(out),format='mp4',fps=10,frame_count=len(frames),source_rollout=str(run),source_ticks=frames,
              source='native learned-policy physics.npz integration_state restored into validated MuJoCo scene',
              expert_replay=False,synthetic_visualization=False,renderer_proof=str(out.parent/'video_renderer.json'),hardware_verified=True,
              complete_native_rollout=bool((run/'report.json').exists() and json.loads((run/'report.json').read_text()).get('rollout_complete',False)))
    (out.with_suffix('.metadata.json')).write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)


if __name__=='__main__': main()
