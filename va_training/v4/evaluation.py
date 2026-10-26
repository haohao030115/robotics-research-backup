import argparse
from pathlib import Path
import numpy as np
import torch
from .model import policy_for_variant
from .dataset import CanonicalV4, dump
from .training import predict, metrics, passes


def load(checkpoint, device='cpu'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=policy_for_variant(saved.get('model_variant')).to(device)
    model.load_state_dict(saved['model']);model.eval()
    return model,saved


def evaluate(checkpoint, output):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'outputs/va_v2_diagnosis/plot_dependencies'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    model,saved=load(checkpoint);ds=CanonicalV4(saved['manifest']);d=ds.data
    if ds.provenance != saved['provenance']:raise ValueError('Checkpoint dataset changed')
    p=predict(model,d);m=metrics(p,d,ds.moving)
    pn={k:v.numpy() for k,v in p.items()};dn={k:v.numpy() for k,v in d.items()}
    h=np.abs(pn['hand_ref']-dn['hand_ref']);b=np.abs(pn['body_tokens']-dn['body_tokens'])
    def closure(ref):return np.clip(((ref-ds.initial)*ds.direction).sum(-1)/(ds.direction@ds.direction),0,1)
    hp,he=closure(pn['hand_ref']),closure(dn['hand_ref'])
    m['body_mae_by_slot']=b.mean((0,2)).tolist();m['hand_mae_by_slot']=h.mean((0,2)).tolist()
    masks=dict(open=(dn['progress']<=.05)&(he.max(1)<=.05),
        transition_preclosure=(dn['progress']<=.05)&(he.max(1)>.05), closing=(dn['phase']==1),hold=(dn['phase']==2))
    m['phase_specific_hand_mae']={k:dict(samples=int(v.sum()),mae=float(h[v].mean()) if v.any() else None) for k,v in masks.items()}
    ticks=np.array([ds.mapping[i]['future_execution_indices'][0] for i in ds.keep]);times=ticks/50
    def onset(x):
        found=np.flatnonzero(x>.05)
        return float(times[found[0]]) if len(found) else None
    po,eo=onset(hp[:,0]),onset(he[:,0])
    m.update(predicted_closure_onset_seconds=po,expert_closure_onset_seconds=eo,
        closure_onset_error_seconds=po-eo if po is not None and eo is not None else None,
        onset_definition='First slot-zero sample with reference-projected closure > .05, 10 Hz sample resolution',
        final_predicted_closure_amplitude=hp[-1,0].item(),expert_final_closure_amplitude=he[-1,0].item(),
        peak_predicted_closure_amplitude=float(hp[:,0].max()),expert_peak_closure_amplitude=float(he[:,0].max()),
        final_predicted_hand_q=pn['hand_ref'][-1,0].tolist(),expert_final_hand_q=dn['hand_ref'][-1,0].tolist(),
        offline_gate=passes(m,saved['config']['offline_gate']),epoch=saved['epoch'],single_source=True)
    out=Path(output);dump(out/'offline_metrics.json',m)
    np.savez_compressed(out/'offline_predictions.npz',**pn, sample_indices=ds.keep, target_ticks=ticks,
        target_body=dn['body_tokens'],target_hand=dn['hand_ref'],target_phase=dn['phase'],target_progress=dn['progress'])
    fig,axes=plt.subplots(7,2,figsize=(14,16),sharex=True)
    for j,ax in enumerate(axes.flat):
        ax.plot(times,dn['hand_ref'][:,0,j],label='Expert');ax.plot(times,pn['hand_ref'][:,0,j],label='V4',alpha=.8)
        ax.set_title(ds.hand_names[j],fontsize=9);ax.set_ylabel('rad')
    axes[0,0].legend();fig.supxlabel('Canonical target control time (s), slot 0');fig.tight_layout();fig.savefig(out/'hand_all14.png');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    axes[0].plot(times,he[:,0],label='Expert hand-reference closure');axes[0].plot(times,hp[:,0],label='V4 hand-trajectory closure')
    current_times=np.array([ds.mapping[i]['anchor_execution_index'] for i in ds.keep])/50
    axes[1].plot(current_times,dn['progress'],label='Expert derived progress');axes[1].plot(current_times,pn['closure_progress'],label='V4 auxiliary progress')
    axes[2].step(current_times,dn['phase'],label='Expert derived phase');axes[2].step(current_times,pn['phase_logits'].argmax(-1),label='V4 phase')
    axes[2].set_yticks([0,1,2],['OPEN','CLOSING','HOLD'])
    for ax in axes:ax.legend()
    fig.supxlabel('Canonical control time (s)');fig.tight_layout();fig.savefig(out/'closure_phase_progress.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].plot(m['body_mae_by_slot']);axes[0].set(title='Body token MAE',xlabel='Horizon slot')
    axes[1].plot(m['hand_mae_by_slot']);axes[1].set(title='Hand reference MAE (rad)',xlabel='Horizon slot')
    fig.tight_layout();fig.savefig(out/'horizon_errors.png');plt.close(fig)
    print(m,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    evaluate(a.checkpoint,a.output)


if __name__=='__main__':main()
