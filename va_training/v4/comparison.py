"""Read-only physical comparison and unpadded synchronized video."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .dataset import CanonicalV4, dump, read
from .evaluation import load


def compare(checkpoint, rollout, output):
    import torch
    torch.set_num_threads(1)
    _,saved=load(checkpoint);ds=CanonicalV4(saved['manifest']);out=Path(output);run=Path(rollout)
    report=read(run/'report.json')
    if not (run/'physics.npz').exists():
        dump(out/'failure_localization.json',dict(first_failure=report['failure'],video_available=False,reason='No learned physical frames'))
        return
    with np.load(run/'physics.npz') as z:learner={k:z[k] for k in z.files}
    with np.load(ds.root/'canonical_replay/post_states.npz') as z:expert={k:z[k] for k in z.files}
    ei={v:i for i,v in enumerate(expert['observation_id'])};li={int(v):i for i,v in enumerate(learner['frame'])}
    pairs=[(r['anchor_execution_index'],ei[r['observation_id']],li[r['anchor_execution_index']]) for r in ds.mapping
           if r['anchor_execution_index'] in li and r['observation_id'] in ei]
    thresholds=dict(body_q=.1,hand_q=.1,right_wrist_pose=.05,cube_pose=.02)
    divergence={};series={}
    for k,threshold in thresholds.items():
        values=[]
        for tick,e,l in pairs:
            delta=learner[k][l]-expert[k][e]
            error=float(np.linalg.norm(delta[:3])) if k.endswith('pose') else float(np.sqrt(np.mean(delta**2)))
            values.append(dict(tick=tick,time_seconds=tick/50,error=error))
        series[k]=values
        hits=[v for v in values if v['error']>threshold]
        divergence[k]=dict(threshold=threshold,metric='position L2 m' if k.endswith('pose') else 'joint RMS rad',first=hits[0] if hits else None)
    timelines=read(run/'control_timeline.json');tl={r['actual_tick']:r for r in timelines if 'hand_target_applied' in r}
    refids={v:i for i,v in enumerate(ds.refs['observation_id'])}
    herrors=[]
    for tick,e,l in pairs:
        if tick in tl:
            ref=ds.refs['hand_ref_q'][refids[expert['observation_id'][e]]]
            error=float(np.abs(np.array(tl[tick]['hand_target_applied'])-ref).mean())
            if error>.1:herrors.append(dict(tick=tick,time_seconds=tick/50,error=error))
    divergence['hand_reference']=dict(threshold=.1,metric='all-joint MAE rad',first=herrors[0] if herrors else None)
    orientation=[]
    contact_errors=[]
    def contact_signature(raw):
        return sorted({tuple(sorted((c['geom1'],c['geom2']))) for c in json.loads(str(raw)) if abs(c['force'][0])>1e-4})
    for tick,e,l in pairs:
        dot=abs(float(np.dot(learner['right_wrist_pose'][l,3:],expert['right_wrist_pose'][e,3:])))
        angle=2*np.arccos(np.clip(dot,0,1))
        if angle>.2:orientation.append(dict(tick=tick,time_seconds=tick/50,error=float(angle)))
        if contact_signature(learner['contacts'][l])!=contact_signature(expert['contacts'][e]):
            contact_errors.append(dict(tick=tick,time_seconds=tick/50))
    divergence['wrist_orientation']=dict(threshold=.2,metric='quaternion geodesic radians',first=orientation[0] if orientation else None)
    divergence['contacts']=dict(first=contact_errors[0] if contact_errors else None,definition='First differing set of positive-force geom contact pairs at canonical sample ticks')
    with np.load(run/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    phase_errors=[]
    by_anchor={r['anchor_execution_index']:i for i,r in enumerate(ds.mapping)}
    for anchor,phase in zip(pred['anchor'],pred['phase_logits'].argmax(-1)):
        # Runtime captures between dataset samples; do not relabel/interpolate reference phase.
        if int(anchor) in by_anchor:
            row=ds.mapping[by_anchor[int(anchor)]];target=ds.phase_all[refids[row['observation_id']]]
            if phase!=target:phase_errors.append(dict(tick=int(anchor),time_seconds=float(anchor/50),predicted=int(phase),expert=int(target)))
    divergence['phase']=dict(first=phase_errors[0] if phase_errors else None,note='Only exact canonical observation anchors are compared; no interpolated labels')
    dump(out/'failure_localization.json',dict(runtime_failure=report['failure'],sample_interval_seconds=.1,divergence=divergence,series=series,
        conclusion='Runtime failure prevents full physical reproduction assessment' if not report['strict_runtime_pass'] else 'See threshold crossings for first measured physical divergence'))
    if not pairs:return
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from canonical_va.native_scene import SnapshotReader
    import mujoco
    from PIL import Image,ImageDraw
    import imageio.v2 as imageio
    reader=SnapshotReader();renderer=mujoco.Renderer(reader.model,height=480,width=640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.15,0,.9];camera.distance=2.6;camera.azimuth=135;camera.elevation=-12
    initial_z=float(ds.runtime['runtime']['initial_qpos'][-5])
    physical=read(run/'physical_metrics.json')
    def contacts(data):
        names=[]
        cube=reader.model.geom('task_cube_geom').id
        for c in data.contact:
            other=c.geom2 if c.geom1==cube else c.geom1 if c.geom2==cube else None
            if other is not None:names.append(reader.model.body(reader.model.geom_bodyid[other]).name)
        return any('thumb' in n for n in names) and any('index' in n or 'middle' in n for n in names),'task_table' in names
    filename=out/'expert_vs_v4.mp4';hold=0.
    with imageio.get_writer(filename,fps=10,codec='libx264',quality=8,macro_block_size=1) as writer:
        for tick,e,l in pairs:
            panels=[]
            for title,state,pose in [('Expert',expert['integration_state'][e],expert['cube_pose'][e]),('V4',learner['integration_state'][l],learner['cube_pose'][l])]:
                reader.restore(state);renderer.update_scene(reader.data,camera=camera)
                panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel)
                opposing,table=contacts(reader.data);lift=float(pose[2]-initial_z)
                if title=='Expert':
                    hold=hold+.1 if opposing and not table and lift>=.1 else 0.
                    held=f'{hold:.1f}s sampled'
                else:held=f'{physical[l]["hold_seconds"]:.2f}s'
                lines=[f'{title} | t={tick/50:.2f}s | lift={lift:.3f}m',f'Opposing contacts={opposing} | table={table} | hold={held}']
                if title=='V4':
                    ix=np.flatnonzero(pred['start_tick']<=tick)
                    if len(ix):lines.append('Phase='+['OPEN','CLOSING','HOLD'][int(pred['phase_logits'][ix[-1]].argmax())])
                    if not report['rollout_complete']:lines.append(f'ABORTED RUN: {report["recorded_ticks"]} ticks; not strict success')
                draw.rectangle([0,0,640,18*len(lines)+8],fill='black')
                for j,line in enumerate(lines):draw.text((8,5+18*j),line,fill='white')
                panels.append(np.asarray(panel))
            writer.append_data(np.concatenate(panels,axis=1))
    renderer.close();reader.close()
    dump(out/'video_metadata.json',dict(path=str(filename),frames=len(pairs),fps=10,first_tick=pairs[0][0],last_tick=pairs[-1][0],
        learner_complete=report['rollout_complete'],padded=False,alignment='Exact canonical sample observation IDs select expert post-state; same absolute execution tick selects actual learner state',
        expert_hold_overlay='Sampled estimate; physical criterion unchanged; canonical authoritative success is stored in expert validation'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--rollout',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    compare(a.checkpoint,a.rollout,a.output)


if __name__=='__main__':main()
