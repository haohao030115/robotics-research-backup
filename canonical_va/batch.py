"""Serial native episode runner. Each episode gets its own existing DDS network isolation."""
import argparse
import subprocess
import sys
from pathlib import Path
from .common import dump,read,digest
from .production import accepted_episode

REQUIRED=('reference.npz','metadata.json','provenance.json','reference_validation.json')


def run_batch(references_root,output_root,mode='production',gpu=0,qa_rate=0.,render_profile='auto',allow_software=False,position_bin_size=None,retention='compact',runner=subprocess.run):
    output_root.mkdir(parents=True,exist_ok=True);results=[]
    references=sorted(p for p in references_root.iterdir() if p.is_dir())
    for ref in references:
        out=output_root/ref.name;row=dict(reference=str(ref.resolve()),output=str(out.resolve()),name=ref.name)
        try:
            missing=[x for x in REQUIRED if not (ref/x).is_file()]
            if missing:raise ValueError('Missing reference files: '+', '.join(missing))
            if out.exists():
                e,_=accepted_episode(out)
                prov=read(out/'provenance.json')
                expected={name:digest(ref/name) for name in REQUIRED}
                if prov.get('reference_source_hashes')!=expected:raise ValueError('Existing accepted episode has different source hashes')
                if e['episode_id']!=read(ref/'metadata.json')['episode_id']:raise ValueError('Existing output episode identity mismatch')
                row.update(status='SKIP',reason='Verified accepted output with identical source; no overwrite',episode_id=e['episode_id'])
            else:
                cmd=['unshare','--net','bash','-c','ip link set lo up && exec "$@"','canonical-va-batch',sys.executable,'-B','-m','canonical_va.pipeline',
                    '--reference',str(ref.resolve()),'--output',str(out.resolve()),'--mode',mode,'--backend','native','--gpu',str(gpu),
                    '--qa-rate',str(qa_rate),'--qa-registry',str((output_root/'qa_registry.json').resolve()),'--render-profile',render_profile]
                cmd += ['--retention',retention]
                if allow_software:cmd+=['--allow-software-rendering']
                if position_bin_size is not None:cmd+=['--position-bin-size',str(position_bin_size)]
                # Logs are outside compact accepted directories and never overwrite a previous attempt.
                with (output_root/(ref.name+'.batch.log')).open('x') as log:
                    result=runner(cmd,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode!=0:
                    summary=read(out/'validation_summary.json') if (out/'validation_summary.json').exists() else {}
                    raise ValueError(summary.get('error') or f'Child exited {result.returncode}; see {ref.name}.batch.log')
                e,_=accepted_episode(out);row.update(status='PASS',episode_id=e['episode_id'],sample_count=e['sample_count'])
        except Exception as exc:row.update(status='FAIL',reason=f'{type(exc).__name__}: {exc}')
        results.append(row);summary={k:sum(x['status']==k for x in results) for k in ('PASS','FAIL','SKIP')}
        dump(output_root/'batch_summary.json',dict(summary,episodes=results,enumerated=len(references),mode=mode,serial=True))
        print(row,flush=True)
    summary={k:sum(x['status']==k for x in results) for k in ('PASS','FAIL','SKIP')}
    dump(output_root/'batch_summary.json',dict(summary,episodes=results,enumerated=len(references),mode=mode,serial=True))
    print('\n'.join(f'{key}: {value}' for key,value in summary.items()),flush=True)
    return 2 if summary['FAIL'] else 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--references-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--backend',choices=['native'],default='native');p.add_argument('--gpu',type=int,default=0);p.add_argument('--mode',choices=['audit','production'],default='production')
    p.add_argument('--qa-rate',type=float,default=0.);p.add_argument('--render-profile',choices=['auto','current','mthreads'],default='auto');p.add_argument('--allow-software-rendering',action='store_true');p.add_argument('--position-bin-size',type=float);p.add_argument('--retention',choices=['compact','training'],default='compact')
    args=p.parse_args();return run_batch(args.references_root,args.output_root,args.mode,args.gpu,args.qa_rate,args.render_profile,args.allow_software_rendering,args.position_bin_size,args.retention)

if __name__=='__main__':raise SystemExit(main())
