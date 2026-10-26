"""Random-init original V4 on merged data; episode-disjoint validation."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .dataset import CanonicalV4, read, dump, sha, gravity
from .model import V4Policy
from .losses import objectives
from .training import metrics, passes, predict


def main(dataset=None, output=None, config=None):
    import torch_musa
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[2];out=Path(output) if output else root/'outputs/va_v4_kimodo19';out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists():raise FileExistsError('Refusing checkpoint overwrite')
    cfg=read(Path(config) if config else Path(__file__).parents[1]/'configs/token_v4.json')
    if dataset is not None:
        from grasp_pipeline.training import load_baseline_inputs
        manifest, merged_path, ids, datasets, source_provenance, train_ids, val_ids = load_baseline_inputs(dataset)
        def collect(which):return {k:torch.cat([ds.data[k] for ds in datasets if ds.episode_id in which]).to('musa:0') for k in datasets[0].data}
        train,val=collect(train_ids),collect(val_ids);moving=torch.tensor(datasets[0].moving,device='musa:0')
    else:
        cfg=read(Path(__file__).parents[1]/'configs/token_v4.json');manifest=root/'outputs/canonical_va/kimodo_19_merged/training_manifest.json'
        merged_manifest=read(manifest);merged_path=Path(merged_manifest['output'])
        with np.load(merged_path,allow_pickle=False) as z:merged={k:z[k] for k in z.files}
        ids=merged['sample_id'].tolist()
        if len(set(ids))!=len(ids):raise ValueError('Duplicate merged sample IDs')
        index={v:i for i,v in enumerate(ids)}
        source_paths=[Path(p) for p in merged_manifest['sources']]
        datasets=[];source_provenance={}
        for source in source_paths:
            ds=CanonicalV4(source.parents[1]/'training_manifest.json')
            with np.load(source,allow_pickle=False) as z:
                rows=np.array([index[x] for x in z['sample_id']])
                for k in z.files:
                    if not np.array_equal(merged[k][rows],z[k]):raise ValueError('Merged/source field differs: '+k)
            # Inputs explicitly read from merged arrays; only RGB/horizon extension use source.
            rows=rows[ds.keep];q=merged['state_history_root_quat'][rows,-1]
            ds.data['body']=torch.tensor(np.concatenate([merged['state_history_body_q'][rows,-1],merged['state_history_body_dq'][rows,-1],q,
                merged['state_history_root_velocity'][rows,-1,3:],gravity(q)],axis=-1),dtype=torch.float32)
            ds.data['hand']=torch.tensor(np.concatenate([merged['state_history_hand_q'][rows,-1],merged['state_history_hand_dq'][rows,-1]],axis=-1),dtype=torch.float32)
            ds.episode_id=ds.mapping[0]['episode_id'];datasets.append(ds);source_provenance[ds.episode_id]=ds.provenance
        episode_ids=sorted(ds.episode_id for ds in datasets)
        shuffled=episode_ids.copy();random.Random(cfg['seed']).shuffle(shuffled);val_ids=set(shuffled[:4]);train_ids=set(shuffled[4:])
        if len(datasets)!=19 or train_ids&val_ids:raise ValueError('Invalid split')
        for ds in datasets:
            if not np.array_equal(ds.moving,datasets[0].moving):raise ValueError('Moving hand joint mask differs')
        def collect(which):return {k:torch.cat([ds.data[k] for ds in datasets if ds.episode_id in which]).to('musa:0') for k in datasets[0].data}
        train,val=collect(train_ids),collect(val_ids);moving=torch.tensor(datasets[0].moving,device='musa:0')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=V4Policy().to('musa:0')
    with torch.no_grad():
        for k in ('body','hand'):
            getattr(model,k+'_mean').copy_(train[k].mean(0));getattr(model,k+'_std').copy_(train[k].std(0).clamp_min(.05))
        sample=model(train['rgb'][:2],train['body'][:2],train['hand'][:2])
    split=dict(seed=cfg['seed'],train_episodes=sorted(train_ids),val_episodes=sorted(val_ids),train_samples=len(train['rgb']),val_samples=len(val['rgb']),
        raw_samples=len(ids),usable_samples=sum(len(ds.keep) for ds in datasets),normalization='train only; same mean/std(clamp_min=.05) formula',
        per_episode={ds.episode_id:len(ds.keep) for ds in datasets})
    preflight=dict(split=split,musa_available=torch.musa.is_available(),device=torch.musa.get_device_name(0),
        input_shapes={k:list(train[k][:2].shape) for k in ('rgb','body','hand')},output_shapes={k:list(v.shape) for k,v in sample.items()},
        parameters=sum(p.numel() for p in model.parameters()),random_init=True,prior_checkpoint_loaded=False,
        frozen_implementation={n:sha(Path(__file__).parent/n) for n in ('model.py','losses.py','inference.py','runtime.py','rollout.py')})
    dump(out/'preflight.json',preflight);dump(out/'split.json',split);dump(out/'config.json',cfg)
    print('PREFLIGHT '+json.dumps(preflight),flush=True)
    training_provenance=dict(manifest=str(manifest),manifest_sha256=sha(manifest),merged_sha256=sha(merged_path),sources=source_provenance)
    dump(out/'training_provenance.json',training_provenance)
    eval_ds=next((ds for ds in datasets if ds.episode_id=='kimodo_episode_0001'),datasets[0])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=0)
    start=time.monotonic();best=float('inf');gate=False
    with (out/'training.jsonl').open('w') as log:
        for epoch in range(1,cfg['epochs']+1):
            model.train();totals={k:0. for k in ('body','hand','phase','progress')}
            order=torch.randperm(len(train['rgb']),device='musa:0')
            for ix in order.split(cfg['batch_size']):
                batch={k:v[ix] for k,v in train.items()};opt.zero_grad(set_to_none=True)
                p=model(batch['rgb'],batch['body'],batch['hand']);loss,terms=objectives(p,batch,cfg['loss_weights'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
                for k,v in terms.items():totals[k]+=float(v.detach())*len(ix)/len(order)
            if epoch%50==0 or epoch==1:
                tp=predict(model,train);vp=predict(model,val);tm=metrics(tp,train,moving);vm=metrics(vp,val,moving)
                with torch.no_grad():
                    tl,tt=objectives(tp,train,cfg['loss_weights']);vl,vt=objectives(vp,val,cfg['loss_weights'])
                gate=passes(tm,cfg['offline_gate'])
                row=dict(epoch=epoch,seconds=time.monotonic()-start,optimization_losses=totals,train_loss=float(tl),val_loss=float(vl),
                    train_objectives={k:float(v) for k,v in tt.items()},val_objectives={k:float(v) for k,v in vt.items()},train_metrics=tm,val_metrics=vm,train_gate=gate)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                # Same score and training-overfit stop condition as original; validation is reported independently.
                score=tm['token_mae']+tm['hand_mae']+.1*tm['progress_mae']
                if score<best or gate:
                    best=score
                    saved=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,epoch=epoch,
                        metrics=tm,val_metrics=vm,offline_gate=gate,training_provenance=training_provenance,split=split,
                        manifest=str(eval_ds.manifest),provenance=eval_ds.provenance,
                        manifest_purpose='Evaluation episode metadata for unchanged runtime; training uses training_provenance and split',random_init=True,prior_checkpoint_loaded=False)
                    torch.save(saved,out/'checkpoint.pt');dump(out/'checkpoint_metrics.json',row)
                if gate:break
            if epoch in (1500,3000):
                for group in opt.param_groups:group['lr']*=.3
    dump(out/'training_status.json',dict(finished=True,epochs=epoch,elapsed_seconds=time.monotonic()-start,train_gate=gate,
        checkpoint=str(out/'checkpoint.pt'),parameters=preflight['parameters']))


if __name__=='__main__':main()
