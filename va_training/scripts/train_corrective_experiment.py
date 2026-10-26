"""Bounded branch-controlled K5 fine tuning with exact mixed batch composition."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from datasets.tokens import load_token_episodes
from datasets.episodes import split_episodes
from trainers.token_training import dataset_for,move
from diagnostics.corrective import corrective_samples,metrics,masked_corrective_loss
from utils.checkpoint import load_policy,save_checkpoint
from utils.config import dump_json
from utils.runtime import initialize,cleanup
from sonic.release import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--corrective',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--lr',type=float,default=0.0001);p.add_argument('--device',default='musa')
    p.add_argument('--branch',choices=['none','body','hand','combined'],required=True);p.add_argument('--fraction',type=float,required=True)
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--from-initialization',action='store_true');args=p.parse_args()
    if args.steps<1 or args.lr<=0 or args.fraction not in [0,.25,.5] or ((args.branch=='none')!=(args.fraction==0)):
        raise ValueError('Require positive bounded steps/lr and a matched branch/fraction')
    rt=initialize(args.device,38)
    try:
        if args.from_initialization:
            from models import build_policy
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            policy=build_policy(saved['config'],saved['normalization_stats']).to(rt.device)
        else:policy,saved=load_policy(args.checkpoint,rt.device)
        config=copy.deepcopy(saved['config']);config['training']['device']=args.device
        out=Path(args.output);out.mkdir(parents=True,exist_ok=False);config['training']['output_dir']=str(out)
        episodes=load_token_episodes(config['data']);train,val=split_episodes(episodes,config['data'])
        corr=corrective_samples(args.corrective,config['model']['image_size']);z=np.load(args.corrective)
        if not any(e.source_episode_id==json.loads((Path(args.corrective).parent/'audit.json').read_text())['source_episode_id'] for e in train):
            raise ValueError('Corrective source must belong to original nominal training split')
        nominal=dataset_for(train,config);validation=dataset_for(val,config)
        # Remove metadata so both sources collate identically; normalization stays checkpoint-frozen.
        keys=['image','state','history_mask','target_sonic_token','target_hand_ref']
        expert=[]
        for i in range(len(nominal)):
            sample=nominal[i];expert.append({k:sample[k] for k in keys})
        cs_train=[s for i,s in enumerate(corr) if z['train_mask'][i]];cs_val=[s for i,s in enumerate(corr) if z['val_mask'][i]]
        batchsize=config['training']['batch_size_per_rank'];nc=int(batchsize*args.fraction);ne=batchsize-nc
        loaders={'nominal_validation':DataLoader(validation,batch_size=batchsize),
            'corrective_validation':DataLoader(cs_val,batch_size=batchsize),'corrective_train':DataLoader(cs_train,batch_size=batchsize)}
        initial={name:metrics(policy,loader,rt.device) for name,loader in loaders.items()}
        if args.from_initialization:
            initial['nominal_train']=metrics(policy,DataLoader(expert,batch_size=batchsize),rt.device)
            original_initial=json.loads((Path(args.checkpoint).parents[1]/'initial_metrics.json').read_text())
            for key in ['token_mae','hand_mae','fsq_level_accuracy']:
                if abs(initial['nominal_train'][key]-original_initial[key])>1e-7:
                    raise ValueError(f'Scratch initialization did not reproduce original {key}')
        dump_json(out/'initial_metrics.json',initial)
        optimizer=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=config['training']['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps)
        rng=np.random.default_rng(38);started=time.monotonic();steps=0
        with (out/'training.jsonl').open('w') as log:
            if not args.evaluate_only:
                for step in range(args.steps):
                    nominal_ids=rng.integers(len(expert),size=ne);corrective_ids=rng.integers(len(cs_train),size=nc)
                    batch=default_collate([expert[i] for i in nominal_ids]+[cs_train[i] for i in corrective_ids])
                    image,state,mask,token,hand=move(batch,rt.device);policy.train();optimizer.zero_grad(set_to_none=True)
                    output=policy(image,state,mask);loss=masked_corrective_loss(output,token,hand,ne,args.branch,config['training'])
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite corrective loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),config['training']['grad_clip']);optimizer.step();scheduler.step();steps=step+1
                    if steps==1 or steps%50==0:
                        record={'step':steps,'loss':float(loss.detach()),'nominal_count':ne,'corrective_count':nc,'branch':args.branch}
                        log.write(json.dumps(record)+'\n');log.flush();print(out.name,record,flush=True)
        checkpoint=out/'checkpoints/latest.pt'
        provenance=copy.deepcopy(saved['provenance']);provenance['corrective_experiment']={'round':'corrective_round1','branch':args.branch,'fraction':args.fraction,
            'initial_checkpoint':args.checkpoint,'initial_checkpoint_sha256':digest(args.checkpoint),'corrective':args.corrective,'corrective_sha256':digest(args.corrective),
            'normalization':'original checkpoint buffers frozen; expert dataset unchanged','steps':steps,'lr':args.lr,'seed':38,'expert_per_batch':ne,'corrective_per_batch':nc}
        provenance['corrective_experiment']['from_original_initialization']=args.from_initialization
        save_checkpoint(checkpoint,policy,optimizer,scheduler,0,steps,config,saved['normalization_stats'])
        stored=torch.load(checkpoint,map_location='cpu',weights_only=True);stored['provenance']=provenance;torch.save(stored,checkpoint)
        final={name:metrics(policy,loader,rt.device) for name,loader in loaders.items()}
        m=final['nominal_validation'];t=config['training']
        gate=(m['token_mae']<=t['heldout_token_mae_max'] and m['fsq_level_accuracy']>=t['heldout_fsq_accuracy_min'] and m['hand_mae']<=t['heldout_hand_mae_max'])
        report={'offline_evaluation_pass':bool(gate),'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'provenance':provenance,
            'split':{'train':[e.id for e in train],'val':[e.id for e in val]},'initial':initial,'metrics':final,'steps':steps,'seconds':time.monotonic()-started,
            'branch':args.branch,'fraction':args.fraction,'evaluation_only':args.evaluate_only,'from_initialization':args.from_initialization,
            'nominal_data_unchanged':True,'corrective_validation_is_independent_rollout':False}
        dump_json(out/'report.json',report);print(out.name,'complete',gate,{k:(v['token_mae'],v['hand_mae']) for k,v in final.items()},flush=True)
    finally:cleanup(rt)


if __name__=='__main__':main()
