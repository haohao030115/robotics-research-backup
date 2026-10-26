import argparse,json,time
from pathlib import Path
import numpy as np,torch
from .model import BCZPolicy
from .losses import objectives

def run(dataset,prior,out,mode='B',epochs=20):
 d=Path(dataset);o=Path(out);o.mkdir(parents=True,exist_ok=False)
 with np.load(d/'samples.npz') as z:a={k:z[k] for k in z.files}
 split=json.loads((d/'split.json').read_text()); rows=[json.loads(x) for x in (d/'index.jsonl').read_text().splitlines()]; trids=set(split['train_episodes']);vaids=set(split['val_episodes'])
 tr=np.array([i for i,r in enumerate(rows) if r['episode_id'] in trids]);va=np.array([i for i,r in enumerate(rows) if r['episode_id'] in vaids])
 # initiation means the upcoming 10 Hz opportunity is within two 50 Hz ticks of first future closure progress; hard negatives are near approach but before that window.
 progress=a['progress']; close=(progress>0.05); labels=np.zeros(len(progress),np.float32)
 for ep in set(r['episode_id'] for r in rows):
  ids=np.array([i for i,r in enumerate(rows) if r['episode_id']==ep]); j=np.flatnonzero(close[ids]);
  if len(j): labels[ids[max(0,j[0]-2):j[0]+1]]=1
 labels_t=torch.tensor(labels)
 dev='musa:0';torch.set_num_threads(2);torch.manual_seed(4);torch.musa.manual_seed_all(4)
 T={k:torch.tensor(a[k][tr],device=dev) for k in ('rgb','body','hand','body_tokens','hand_ref','phase','progress')};V={k:torch.tensor(a[k][va],device=dev) for k in T};T['close_intent']=labels_t[tr].to(dev);V['close_intent']=labels_t[va].to(dev)
 m=BCZPolicy().to(dev);c=torch.load(prior,map_location='cpu',weights_only=False);base={k:v for k,v in c['model'].items() if k in m.state_dict() and m.state_dict()[k].shape==v.shape};m.load_state_dict(base,strict=False)
 # Freeze shared visual/body/grasp context and body/phase heads; tune learned hand head and new trigger only.
 for n,p in m.named_parameters():p.requires_grad=n.startswith('hand_head.') or n.startswith('close_intent_head.')
 with torch.no_grad():
  m.body_mean.copy_(T['body'].mean(0));m.body_std.copy_(T['body'].std(0).clamp_min(.05));m.hand_mean.copy_(T['hand'].mean(0));m.hand_std.copy_(T['hand'].std(0).clamp_min(.05))
 initial=T['hand_ref'][:,0].mean(0);direction=(T['hand_ref'][:,-1]-T['hand_ref'][:,0]).mean(0);mask=(direction.abs()>.05).float();
 if mode=='B': weights={'body':1.,'hand':3.,'phase':.1,'progress':.1,'trigger':.2,'advance':0.}
 else: weights={'body':1.,'hand':3.,'phase':.1,'progress':.1,'trigger':.2,'advance':.5}
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4,weight_decay=0);hist=[]
 for ep in range(1,epochs+1):
  m.train();ix=torch.randperm(len(tr),device=dev)
  for b in ix.split(32):
   opt.zero_grad(set_to_none=True);p=m(T['rgb'][b],T['body'][b],T['hand'][b]);l,_,_,_=objectives(p,{k:v[b] for k,v in T.items()},weights,initial,direction,mask);l.backward();torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.requires_grad],10);opt.step()
  if ep==1 or ep%10==0 or ep==epochs:
   m.eval();
   with torch.no_grad():p=m(V['rgb'],V['body'],V['hand']);l,terms,pp,pg=objectives(p,V,weights,initial,direction,mask);pr=(torch.sigmoid(p['close_intent_logit'])>.5); y=V['close_intent'];tp=int(((pr==1)&(y==1)).sum());fp=int(((pr==1)&(y==0)).sum());fn=int(((pr==0)&(y==1)).sum());
   row={'epoch':ep,'loss':float(l),'trigger_precision':tp/max(tp+fp,1),'trigger_recall':tp/max(tp+fn,1),'trigger_pos':int(y.sum()),'advance':float(terms['advance']),'hand_mae':float((p['hand_ref']-V['hand_ref']).abs().mean())};hist.append(row);print(json.dumps(row),flush=True)
 state={k:v.detach().cpu() for k,v in m.state_dict().items()};torch.save({'model':state,'epoch':epochs,'mode':mode,'config':weights,'split':split,'dataset':str(d),'prior_checkpoint':str(prior),'v40_bcz_hand':True,'manifest':c['manifest'],'provenance':c['provenance'],'offline_gate':True},o/'checkpoint.pt');(o/'history.json').write_text(json.dumps(hist,indent=2));(o/'training_status.json').write_text(json.dumps({'mode':mode,'epochs':epochs,'train_samples':len(tr),'val_samples':len(va),'parameters':sum(p.numel() for p in m.parameters())},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--prior',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['B','C'],required=True);p.add_argument('--epochs',type=int,default=20);a=p.parse_args();run(a.dataset,a.prior,a.output,a.mode,a.epochs)
