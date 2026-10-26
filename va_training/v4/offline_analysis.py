"""Compute comparable offline and phase/closure diagnostics from flat v0 tensors."""
import json
from pathlib import Path
import numpy as np
import torch
from .model import policy_for_variant

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/va_v4_ablation'; DATA=ROOT/'expert_datasets/train_dataset_v0'
CONFIGS=['AUX-0_baseline','AUX-1_no_phase','AUX-2_no_progress','AUX-3_no_phase_progress','ARCH-1_mlp_fusion','ARCH-2_proprio_only']

@torch.no_grad()
def pred(model,d):
 model.eval(); out={}
 for i in range(0,len(d['rgb']),256):
  p=model(d['rgb'][i:i+256],d['body'][i:i+256],d['hand'][i:i+256])
  for k,v in p.items():out.setdefault(k,[]).append(v.cpu())
 return {k:torch.cat(v).numpy() for k,v in out.items()}

def main():
 with np.load(DATA/'samples.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 rows=[json.loads(x) for x in (DATA/'index.jsonl').read_text().splitlines() if x]; split=json.loads((DATA/'split.json').read_text()); va_ids=set(split['val_episodes']); ix=np.array([i for i,r in enumerate(rows) if r['episode_id'] in va_ids])
 d={k:torch.tensor(a[k][ix],dtype=torch.float32 if k not in ('phase',) else torch.long) for k in ('rgb','body','hand')}
 target={k:a[k][ix] for k in ('body_tokens','hand_ref','phase','progress')}
 # Derive the same right-hand closure direction used by v4.dataset.progress_labels.
 all_ref=a['hand_ref'][:,0]; initial=all_ref[0]; right=np.array([i>=7 for i in range(14)]); peak=np.argmax(np.square((all_ref-initial)*right).sum(-1)); direction=all_ref[peak]-initial; moving=right & (np.abs(direction)>.05); direction=direction*moving; denom=float(direction@direction)
 target_closure=np.clip(((target['hand_ref'][:,0]-initial)*direction).sum(-1)/denom,0,1)
 results={}
 for cfg in CONFIGS:
  ck=OUT/'AUX-0_baseline_checkpoint.pt' if cfg=='AUX-0_baseline' else OUT/cfg/'checkpoint.pt'; saved=torch.load(ck,map_location='cpu',weights_only=False); model=policy_for_variant(saved.get('model_variant')).to('cpu'); model.load_state_dict(saved['model']); p=pred(model,d)
  b=p['body_tokens']-target['body_tokens']; h=p['hand_ref']-target['hand_ref']; ph=p['phase_logits'].argmax(-1); prog=p['closure_progress']; masks={'open':target['phase']==0,'closing':target['phase']==1,'hold':target['phase']==2}
  phase_mae={name:dict(samples=int(mask.sum()),hand_mae=float(np.abs(h[mask]).mean()) if mask.any() else None) for name,mask in masks.items()}
  predicted_closure=np.clip(((p['hand_ref'][:,0]-initial)*direction).sum(-1)/denom,0,1)
  onset_errors=[]
  for eid in va_ids:
   ids=np.array([j for j,i in enumerate(ix) if rows[i]['episode_id']==eid]);
   if len(ids):
    et=np.flatnonzero(target_closure[ids]>.05); ep=np.flatnonzero(predicted_closure[ids]>.05)
    if len(et) and len(ep): onset_errors.append(float(abs(rows[ix[ids[ep[0]]]]['mapping']['anchor_execution_index']-rows[ix[ids[et[0]]]]['mapping']['anchor_execution_index'])/50))
  result=dict(config=cfg,split='held_out_val_episodes',samples=int(len(ix)),body_token_mae=float(np.abs(b).mean()),hand_reference_mae=float(np.abs(h).mean()),
    moving_right_hand_mae=float(np.abs(h[:,:,moving]).mean()),phase_accuracy=float((ph==target['phase']).mean()),progress_mae=float(np.abs(prog-target['progress']).mean()),
    phase_specific_hand_mae=phase_mae,expert_progress_mean=float(target['progress'].mean()),predicted_progress_mean=float(prog.mean()),expert_progress_peak=float(target['progress'].max()),predicted_progress_peak=float(prog.max()),
    expert_closure_mean=float(target_closure.mean()),predicted_closure_mean=float(predicted_closure.mean()),expert_closure_peak=float(target_closure.max()),predicted_closure_peak=float(predicted_closure.max()),
    closure_onset_error_mean_s=float(np.mean(onset_errors)) if onset_errors else None,closure_onset_episode_count=len(onset_errors),
    auxiliary_loss_connection='phase_feature feeds hand_head, phase_head, and progress_head; zeroing phase/progress direct losses does not sever shared hand path')
  (OUT/cfg/'offline_analysis.json' if cfg!='AUX-0_baseline' else OUT/'AUX-0_offline_analysis.json').write_text(json.dumps(result,indent=2)+'\n'); results[cfg]=result
 (OUT/'offline_comparison.json').write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2))

if __name__=='__main__':main()
