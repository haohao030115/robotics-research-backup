"""Single-device deterministic regression with episode-level validation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets.tokens import load_token_episodes
from datasets.episodes import split_episodes
from sonic.release import bundle_provenance, digest
from utils.config import dump_json
from utils.runtime import initialize, cleanup
from .config import load_config
from .data import V3Dataset, FIELDS, current_proprio, alignment_report
from .model import V3Policy


def losses(output, body, hand, config):
    if config['body_loss'] == 'huber':
        b = F.huber_loss(output['sonic_token'], body, delta=config['huber_delta'])
    elif config['body_loss'] == 'mse':
        b = F.mse_loss(output['sonic_token'], body)
    else:
        raise ValueError('Body loss must be huber or mse')
    h = F.mse_loss(output['hand_ref'], hand)
    return b, h


def move(batch, device):
    return [batch[k].to(device) for k in ('image', 'proprio', 'target_sonic_token', 'target_hand_ref')]


@torch.inference_mode()
def evaluate(model, loader, device, config):
    model.eval(); n=0; body_sum=torch.zeros(2,64,device=device); hand_sum=torch.zeros(2,14,device=device)
    body_sq=torch.zeros_like(body_sum); hand_sq=torch.zeros_like(hand_sum); loss_sum=np.zeros(2)
    for batch in loader:
        image, state, body, hand=move(batch,device)
        output=model(image,state); b,h=losses(output,body,hand,config)
        be=output['sonic_token']-body; he=output['hand_ref']-hand
        body_sum+=be.abs().sum(0); hand_sum+=he.abs().sum(0)
        body_sq+=be.square().sum(0); hand_sq+=he.square().sum(0)
        loss_sum+=np.array([b.item(),h.item()])*len(image); n+=len(image)
    if not n: raise ValueError('Empty evaluation split')
    b=(body_sum/n).cpu().numpy(); h=(hand_sum/n).cpu().numpy()
    return dict(samples=n,body_loss=float(loss_sum[0]/n),hand_loss=float(loss_sum[1]/n),
        token_mae=float(b.mean()),token_mae_per_slot=b.mean(1).tolist(),token_mae_per_dimension=b.mean(0).tolist(),
        token_mae_per_slot_dimension=b.tolist(),hand_mae=float(h.mean()),hand_mae_per_slot=h.mean(1).tolist(),
        hand_mae_per_dimension=h.mean(0).tolist(),token_mse_per_slot=(body_sq/n).mean(1).cpu().tolist(),
        hand_mse_per_slot=(hand_sq/n).mean(1).cpu().tolist())


def load_checkpoint(path, device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['format']!='va_v3_current_observation_2tick': raise ValueError('Not a V3 checkpoint')
    model=V3Policy().to(device); model.load_state_dict(saved['model']); model.eval()
    return model,saved


def train_main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int); p.add_argument('--device',choices=['musa','cpu'])
    args=p.parse_args(); c=load_config(args.config); t=c['training']
    if args.steps is not None: t['max_steps']=args.steps
    if args.device: t['device']=args.device
    if t['max_steps']<1: raise ValueError('Positive steps required')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    rt=initialize(t['device'],t['seed'])
    if rt.distributed: raise ValueError('V3 trainer supports one device')
    try:
        episodes=load_token_episodes(c['data']); train,val=split_episodes(episodes,c['data'])
        if not val: raise ValueError('Episode-held-out validation is required')
        split={'train':[e.id for e in train],'val':[e.id for e in val]}
        dump_json(out/'split.json',split); dump_json(out/'config.json',c)
        dump_json(out/'alignment.json',alignment_report(episodes))
        provenance=dict(sonic=bundle_provenance(c['sonic']),body_joint_order=episodes[0].body_order,
            hand_joint_order=episodes[0].hand_order,episodes=[e.provenance for e in episodes])
        dump_json(out/'provenance.json',provenance)
        ds=V3Dataset(train); vd=V3Dataset(val)
        dl=DataLoader(ds,batch_size=t['batch_size'],shuffle=True,num_workers=0)
        vl=DataLoader(vd,batch_size=t['batch_size'],num_workers=0)
        model=V3Policy().to(rt.device)
        state=np.stack([current_proprio({k:ds.episodes[i].state[k][r] for k in FIELDS}) for i,j,r in ds.index])
        model.state_mean.copy_(torch.from_numpy(state.mean(0)).to(rt.device))
        model.state_std.copy_(torch.from_numpy(state.std(0).clip(.001)).to(rt.device))
        optim=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
        dump_json(out/'initial_val.json',evaluate(model,vl,rt.device,t))
        iterator=iter(dl); started=time.monotonic(); best=float('inf')
        with (out/'training.jsonl').open('w') as log:
            for step in range(1,t['max_steps']+1):
                try: batch=next(iterator)
                except StopIteration: iterator=iter(dl); batch=next(iterator)
                model.train(); image,state,body,hand=move(batch,rt.device)
                optim.zero_grad(set_to_none=True); output=model(image,state); b,h=losses(output,body,hand,t)
                (b+h).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optim.step()
                if step==1 or step%t['log_every']==0:
                    be=(output['sonic_token']-body).detach().abs(); he=(output['hand_ref']-hand).detach().abs()
                    row=dict(step=step,body_loss=b.item(),hand_loss=h.item(),token_mae=be.mean().item(),
                        token_mae_per_slot=be.mean((0,2)).cpu().tolist(),token_mae_per_dimension=be.mean((0,1)).cpu().tolist(),
                        hand_mae=he.mean().item(),hand_mae_per_slot=he.mean((0,2)).cpu().tolist(),elapsed_seconds=time.monotonic()-started)
                    log.write(json.dumps(row)+'\n'); log.flush()
                    print({k:row[k] for k in ('step','body_loss','hand_loss','token_mae','hand_mae','elapsed_seconds')},flush=True)
                if step%t['eval_every']==0 or step==t['max_steps']:
                    metrics=evaluate(model,vl,rt.device,t); score=metrics['body_loss']+metrics['hand_loss']
                    saved=dict(format='va_v3_current_observation_2tick',model=model.state_dict(),optimizer=optim.state_dict(),
                        config=c,split=split,provenance=provenance,step=step,metrics=metrics)
                    torch.save(saved,out/'latest.pt')
                    if score<best:
                        best=score; torch.save(saved,out/'best.pt'); dump_json(out/'best_val.json',metrics)
                    dump_json(out/'latest_val.json',metrics)
                    print(dict(step=step,validation=metrics['token_mae'],hand_mae=metrics['hand_mae']),flush=True)
        dump_json(out/'report.json',dict(steps=t['max_steps'],elapsed_seconds=time.monotonic()-started,
            parameters=sum(p.numel() for p in model.parameters()),split=split,best_checkpoint_sha256=digest(out/'best.pt'),
            latest_validation=metrics,trained=True,closed_loop_success_not_implied=True))
    finally: cleanup(rt)


def eval_main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='musa',choices=['musa','cpu']);args=p.parse_args()
    rt=initialize(args.device,38)
    try:
        model,saved=load_checkpoint(args.checkpoint,rt.device)
        episodes=load_token_episodes(saved['config']['data']); ids=set(saved['split']['val'])
        val=[e for e in episodes if e.id in ids]
        if {e.id for e in val}!=ids: raise ValueError('Checkpoint validation episodes unavailable')
        result=evaluate(model,DataLoader(V3Dataset(val),batch_size=32),rt.device,saved['config']['training'])
        result.update(checkpoint_sha256=digest(args.checkpoint),episode_ids=sorted(ids))
        dump_json(args.output,result);print(result,flush=True)
    finally:cleanup(rt)
