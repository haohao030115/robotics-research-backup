"""Same metric definitions for original and 19-episode checkpoints."""
import json
from pathlib import Path
import numpy as np
from v4.dataset import read,dump,progress_labels


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def stats(x):
    x=np.asarray(x);return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))


def main():
    root=Path('/workspace/group2');out=root/'outputs/va_v4_kimodo19';source=root/'outputs/canonical_va/kimodo_episode_0001_native_final'
    refs=load(source/'canonical_replay/references.npz');meta=read(source/'source_reference/metadata.json')
    _,_,moving,initial,direction=progress_labels(refs['hand_ref_q'],meta['hand_joint_names'])
    post=load(source/'canonical_replay/post_states.npz');pi={x:i for i,x in enumerate(post['observation_id'])}
    mapping=[json.loads(x) for x in (source/'va_dataset/sample_mapping.jsonl').read_text().splitlines()]
    result={}
    for label,run in [('single_episode',root/'outputs/va_v4_overfit/kimodo/normal_gpu'),('kimodo19',out/'normal_gpu')]:
        report=read(run/'report.json');p=load(run/'predictions.npz');phys=load(run/'physics.npz');timeline=read(run/'control_timeline.json')
        fi={int(x):i for i,x in enumerate(phys['frame'])};early=[]
        for r in mapping:
            t=r['anchor_execution_index']
            if t>50 or t not in fi:continue
            e=pi[r['observation_id']];l=fi[t];dot=abs(float(phys['right_wrist_pose'][l,3:]@post['right_wrist_pose'][e,3:]))
            early.append(dict(tick=t,time_seconds=t/50,body_q_rms=float(np.sqrt(np.mean((phys['body_q'][l]-post['body_q'][e])**2))),
                wrist_position_error=float(np.linalg.norm(phys['right_wrist_pose'][l,:3]-post['right_wrist_pose'][e,:3])),
                wrist_orientation_error=float(2*np.arccos(np.clip(dot,0,1)))))
        targets=np.array([r['hand_target_applied'] for r in timeline if 'hand_target_applied' in r]);ticks=np.array([r['actual_tick'] for r in timeline if 'hand_target_applied' in r])
        closure=np.clip(((targets-initial)*direction).sum(-1)/(direction@direction),0,1);found=np.flatnonzero(closure>.05)
        measured=np.clip(((phys['hand_q']-initial)*direction).sum(-1)/(direction@direction),0,1);mf=np.flatnonzero(measured>.05)
        assert np.all(np.diff(p['start_tick'])==5)
        delta=np.abs(p['hand_ref'][:-1,5]-p['hand_ref'][1:,0]);over=np.abs(p['hand_ref'][:-1,5:]-p['hand_ref'][1:,:45])
        phases=p['phase_logits'].argmax(-1)
        result[label]=dict(physical=report,early_rows=early,early_summary={k:stats([r[k] for r in early]) for k in ('body_q_rms','wrist_position_error','wrist_orientation_error')},
            hand_closure_onset_tick=int(ticks[found[0]]) if len(found) else None,hand_closure_onset_seconds=float(ticks[found[0]]/50) if len(found) else None,
            measured_closure_onset_tick=int(phys['frame'][mf[0]]) if len(mf) else None,peak_hand_progress=float(closure.max()),
            hand_boundary_all14=stats(delta.mean(-1)),hand_boundary_moving_right=stats(delta[:,moving].mean(-1)),
            hand_overlap_moving_right=stats(over[...,moving].mean(-1)),hand_internal_moving_right=stats(np.abs(np.diff(p['hand_ref'],axis=1)[...,moving]).mean(-1)),
            phase_flips=int(np.count_nonzero(np.diff(phases))),phase_counts={str(k):int((phases==k).sum()) for k in range(3)})
    dump(out/'comparison.json',result)
    import sys
    sys.path.insert(0,str(root/'outputs/va_v2_diagnosis/plot_dependencies'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for label,values in result.items():
        rows=values['early_rows']
        for ax,k in zip(axes,('body_q_rms','wrist_position_error','wrist_orientation_error')):
            ax.plot([r['time_seconds'] for r in rows],[r[k] for r in rows],label=label);ax.set_ylabel(k);ax.legend()
    axes[-1].set_xlabel('Control time (s)');fig.tight_layout();fig.savefig(out/'first_second_comparison.png');plt.close(fig)
    print(json.dumps({k:{x:v[x] for x in ('early_summary','hand_closure_onset_seconds','hand_boundary_moving_right','phase_flips')} for k,v in result.items()},indent=2))


if __name__=='__main__':main()
