"""Deterministic oracle-sequence perturbations, scheduling controls and substitutions."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.config import dump_json
from sonic.release import digest,snap_token


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','oracle-report','overfit-report','offline-report','episode-id','predictions','residuals','output']: p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,default=20261005)
    p.add_argument('--prepare-only',action='store_true'); p.add_argument('--only',nargs='+')
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    nominal=np.load(args.predictions); gt=nominal['gt_tokens']; predicted=nominal['tokens']; residuals=np.load(args.residuals)['residual']
    sequences={'fsq_0':gt.copy()}
    for rate in [.01,.02,.05,.1,.2,.5]:
        rng=np.random.default_rng(args.seed); change=rng.random(gt.shape)<rate
        step=rng.choice([-1.,1.],size=gt.shape)/16
        sequences['fsq_'+str(int(rate*100))]=np.clip(gt+change*step,-1,15/16).astype(np.float32)
    rng=np.random.default_rng(args.seed)
    sequences['empirical']=gt+residuals[rng.integers(0,len(residuals),len(gt))]
    for rate in [.1,.25,.5,.75,1.]:
        rng=np.random.default_rng(args.seed); frames=rng.choice(len(gt),int(len(gt)*rate),replace=False)
        a=gt.copy(); a[frames]=predicted[frames]; sequences['substitution_'+str(int(rate*100))]=a
    # Pure scheduling interventions disentangle model errors from latency/holding.
    sequences['gt_delay2']=gt[np.maximum(np.arange(len(gt))-2,0)]
    sequences['gt_hold5']=gt[(np.arange(len(gt))//5)*5]
    sequences['gt_delay2_hold5']=gt[(np.maximum(np.arange(len(gt))-2,0)//5)*5]
    manifest={}
    for name,tokens in sequences.items():
        path=out/(name+'.npz'); np.savez_compressed(path,tokens=tokens)
        wrong=(snap_token(tokens)!=gt).sum(-1)
        manifest[name]={'sequence':str(path.resolve()),'sha256':digest(path),'token_mae':float(abs(tokens-gt).mean()),
                        'whole_token_fsq_accuracy':float((wrong==0).mean()),'wrong_coordinates_mean':float(wrong.mean()),
                        'wrong_coordinate_histogram':{str(k):int((wrong==k).sum()) for k in np.unique(wrong)}}
    dump_json(out/'manifest.json',{'seed':args.seed,'oracle_episode':args.episode_id,'conditions':manifest,
        'prediction_method':'Dense nominal expert state at 50 Hz, causal latest expert RGB at 10 Hz, GT startup token 0',
        'residual_method':'Sample whole 64D held-out residual vectors independently per tick; retain within-token correlations'})
    if args.prepare_only: return
    results={}
    for name in sequences:
        if args.only and name not in args.only: continue
        target=out/name
        if (target/'report.json').exists(): results[name]=json.loads((target/'report.json').read_text()); continue
        command=[sys.executable,'-B',str(Path(__file__).with_name('diagnose_token_v2.py')),
            '--checkpoint',args.checkpoint,'--episode-id',args.episode_id,'--oracle-report',args.oracle_report,
            '--overfit-report',args.overfit_report,'--offline-report',args.offline_report,
            '--experiment','sequence','--device','cpu','--sequence',str(out/(name+'.npz')),'--output',str(target)]
        subprocess.run(command,check=True)
        results[name]=json.loads((target/'report.json').read_text()); dump_json(out/'summary.json',results)
        if name=='fsq_0' and not results[name]['closed_loop_pass']: raise RuntimeError('Oracle failed: stop corruption suite')
    dump_json(out/'summary.json',results)


if __name__=='__main__': main()
