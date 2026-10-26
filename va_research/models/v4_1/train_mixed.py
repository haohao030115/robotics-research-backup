import json,time
from pathlib import Path
import numpy as np, torch
from .model import V41Policy
from .losses import objectives

def run(dataset, prior, output, epochs=50):
    d=Path(dataset); out=Path(output); out.mkdir(parents=True,exist_ok=False)
    with np.load(d/'samples.npz') as z: a={k:z[k] for k in z.files}
    split=json.loads((d/'split.json').read_text())
    rows=[json.loads(x) for x in (d/'index.jsonl').read_text().splitlines()]
    tr=np.array([i for i,r in enumerate(rows) if r['episode_id'] in set(split['train_episodes'])]); va=np.array([i for i,r in enumerate(rows) if r['episode_id'] in set(split['val_episodes'])])
    dev='musa:0'; torch.set_num_threads(2); torch.manual_seed(4); torch.musa.manual_seed_all(4)
    train={k:torch.tensor(a[k][tr],device=dev) for k in ('rgb','body','hand','body_tokens','hand_ref','phase','progress')}; val={k:torch.tensor(a[k][va],device=dev) for k in train}
    m=V41Policy().to(dev); c=torch.load(prior,map_location='cpu',weights_only=False); base={k:v for k,v in c['model'].items() if k in m.state_dict() and m.state_dict()[k].shape==v.shape}; m.load_state_dict(base,strict=False)
    with torch.no_grad():
        m.body_mean.copy_(train['body'].mean(0));m.body_std.copy_(train['body'].std(0).clamp_min(.05));m.hand_mean.copy_(train['hand'].mean(0));m.hand_std.copy_(train['hand'].std(0).clamp_min(.05))
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0); weights={'body':1.,'hand':1.,'phase':.1,'progress':.1,'trigger':.1,'velocity':.1,'reopen':.02}; start=time.time(); history=[]
    for ep in range(1,epochs+1):
      m.train(); ix=torch.randperm(len(tr),device=dev)
      for b in ix.split(32):
        opt.zero_grad(set_to_none=True); p=m(train['rgb'][b],train['body'][b],train['hand'][b]); loss,_=objectives(p,{k:v[b] for k,v in train.items()},weights); loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10);opt.step()
      if ep==1 or ep%10==0 or ep==epochs:
        m.eval();
        with torch.no_grad():
          p=m(val['rgb'],val['body'],val['hand']); vl,terms=objectives(p,val,weights)
          mae=float((p['hand_ref']-val['hand_ref']).abs().mean()); phase=float((p['phase_logits'].argmax(-1)==val['phase']).float().mean()); trig=float(torch.sigmoid(p['grasp_ready_logit'][:,0]).mean())
        row={'epoch':ep,'val_loss':float(vl),'val_hand_mae':mae,'val_phase_accuracy':phase,'val_trigger_rate':trig,'terms':{k:float(v) for k,v in terms.items()}};history.append(row);print(json.dumps(row),flush=True)
    m.eval(); torch.save({'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},'config':weights,'epoch':epochs,'prior_checkpoint':str(prior),'dataset':str(d),'split':split,'v41':True},out/'checkpoint.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2)); (out/'training_status.json').write_text(json.dumps({'finished':True,'epochs':epochs,'parameters':sum(p.numel() for p in m.parameters()),'checkpoint':str(out/'checkpoint.pt'),'train_samples':len(tr),'val_samples':len(va)},indent=2)); return out/'checkpoint.pt'
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--prior',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=50);a=p.parse_args();run(a.dataset,a.prior,a.output,a.epochs)
