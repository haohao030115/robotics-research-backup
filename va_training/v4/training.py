import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .model import V4Policy
from .dataset import CanonicalV4, dump
from .losses import objectives


def metrics(p, d, moving):
    b=p['body_tokens']-d['body_tokens'];h=p['hand_ref']-d['hand_ref']
    projected=(p['body_tokens']*16).round().clamp(-16,15)/16
    correct=projected==d['body_tokens']
    return dict(token_mae=float(b.abs().mean()), token_mse=float(b.square().mean()),
        fsq_scalar_accuracy=float(correct.float().mean()), whole_token_exact_accuracy=float(correct.all(-1).float().mean()),
        hand_mae=float(h.abs().mean()), moving_right_hand_mae=float(h[:,:,moving].abs().mean()),
        phase_accuracy=float((p['phase_logits'].argmax(-1)==d['phase']).float().mean()),
        progress_mae=float((p['closure_progress']-d['progress']).abs().mean()))


def passes(m, gate):
    return all(m[k]>=v if 'accuracy' in k else m[k]<=v for k,v in gate.items())


@torch.no_grad()
def predict(model, data, batch=32):
    result={}
    model.eval()
    for start in range(0,len(data['rgb']),batch):
        p=model(*[data[k][start:start+batch] for k in ('rgb','body','hand')])
        for k,v in p.items():result.setdefault(k,[]).append(v)
    return {k:torch.cat(v) for k,v in result.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(Path(__file__).parents[1]/'configs/token_v4.json'))
    parser.add_argument('--device',default='musa:0');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists():raise FileExistsError('Refusing to overwrite checkpoint')
    torch.set_num_threads(2)
    if args.device.startswith('musa'):import torch_musa
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    ds=CanonicalV4(args.manifest);dump(out/'schema.json',ds.report());dump(out/'config.json',cfg)
    d={k:v.to(args.device) for k,v in ds.data.items()};moving=torch.tensor(ds.moving,device=args.device)
    model=V4Policy().to(args.device)
    with torch.no_grad():
        for k in ('body','hand'):
            getattr(model,k+'_mean').copy_(d[k].mean(0));getattr(model,k+'_std').copy_(d[k].std(0).clamp_min(.05))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0)
    start=time.monotonic();best=float('inf');gate=False
    with (out/'training.jsonl').open('w') as log:
        for epoch in range(1,cfg['epochs']+1):
            model.train();totals={k:0. for k in ('body','hand','phase','progress')}
            order=torch.randperm(len(d['rgb']),device=args.device)
            for ids in order.split(cfg['batch_size']):
                target={k:v[ids] for k,v in d.items()}
                opt.zero_grad(set_to_none=True)
                p=model(target['rgb'],target['body'],target['hand']);loss,terms=objectives(p,target,cfg['loss_weights'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                for k,v in terms.items():totals[k]+=float(v.detach())*len(ids)/len(order)
            if epoch%50==0 or epoch==1:
                p=predict(model,d);m=metrics(p,d,moving);gate=passes(m,cfg['offline_gate'])
                row=dict(epoch=epoch,seconds=time.monotonic()-start,losses=totals,metrics=m,offline_gate=gate)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                score=m['token_mae']+m['hand_mae']+.1*m['progress_mae']
                if score<best or gate:
                    best=score
                    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,
                        provenance=ds.provenance,manifest=str(ds.manifest),epoch=epoch,metrics=m,offline_gate=gate),out/'checkpoint.pt')
                if gate:break
            if epoch in (1500,3000):
                for group in opt.param_groups:group['lr']*=.3
    dump(out/'training_status.json',dict(finished=True,epochs=epoch,elapsed_seconds=time.monotonic()-start,offline_gate=gate))


if __name__=='__main__':main()
