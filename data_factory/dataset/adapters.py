"""Process adapter for existing compiler. No simulation or V4 indexing logic."""
import os
from pathlib import Path
import signal
import subprocess
import sys
from .contracts import *

REPO=Path(__file__).resolve().parents[1]

def reference_identity(root):
    root=Path(root);require(root.is_dir(),'Reference root missing: '+str(root))
    paths=[p/n for p in sorted(root.iterdir()) if p.is_dir() for n in ('reference.npz','metadata.json','provenance.json','reference_validation.json') if (p/n).is_file()]
    hashes=files_record(paths)
    return dict(artifact_id='references:'+fingerprint(hashes),path=str(root.resolve()),hashes=hashes)

def canonical_identity(root):
    from .dataset import candidates
    paths=[p/'training_manifest.json' for p in candidates(root)]
    if (Path(root)/'training_manifest.json').exists():paths.append(Path(root)/'training_manifest.json')
    if (Path(root)/'batch_summary.json').exists():paths.append(Path(root)/'batch_summary.json')
    hashes=files_record(paths);return dict(artifact_id='canonical-index:'+fingerprint(hashes),path=str(Path(root).resolve()),hashes=hashes)

def compiler_command(ref,out,config):
    command=[sys.executable,'-B','-m','data_factory.runtime.batch','--references-root',str(ref),'--output-root',str(out),
             '--backend','native','--gpu',str(config.get('gpu',0)),'--mode',config.get('mode','audit'),'--render-profile',config.get('render_profile','auto'),'--retention',config.get('retention','compact')]
    if config.get('allow_software_rendering'):command.append('--allow-software-rendering')
    return command

def terminate_tree(process):
    """Include native descendants that create their own sessions; never signal unrelated PIDs."""
    identities={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            identities[int(path.parent.name)]=(int(fields[1]),fields[19])
        except (OSError,ValueError,IndexError):continue
    descendants={process.pid};changed=True
    while changed:
        added={pid for pid,(parent,_) in identities.items() if parent in descendants}-descendants
        descendants.update(added);changed=bool(added)
    def send(sig):
        for pid in descendants:
            try:
                fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
                if pid in identities and fields[19]==identities[pid][1]:os.kill(pid,sig)
            except (OSError,IndexError):pass
    send(signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    send(signal.SIGKILL)
    process.wait()


def run_compiler(ref,out,config,log,child_started,lock_fd):
    cmd=compiler_command(ref,out,config)
    env=os.environ.copy();env['PYTHONPATH']=str(REPO)+os.pathsep+env.get('PYTHONPATH','')
    # No shell orchestration. Existing batch owns per-episode network namespace semantics.
    p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
    child_started(p.pid)
    try:rc=p.wait()
    except BaseException:
        terminate_tree(p)
        raise
    summary=Path(out)/'batch_summary.json'
    result=dict(exit_code=rc,path=str(out),command=cmd,evidence=[str(summary)] if summary.is_file() else [])
    if summary.exists():
        batch=read(summary);result['batch_summary']=batch
        if batch.get('FAIL'):
            result.update(error_kind='partial_batch' if batch.get('PASS',0)+batch.get('SKIP',0)>0 else 'execution',error='Compiler batch has failures; original episode reasons retained in batch_summary')
    return result

def validate_compile(result):
    from data_factory.runtime.production import accepted_episode
    path=Path(result['path']);batch=read(path/'batch_summary.json')
    require(batch.get('enumerated',0)>0,'Compiler enumerated no episodes')
    require(len(batch.get('episodes',[]))==batch['enumerated'],'Incomplete compiler batch summary')
    for status in ('PASS','FAIL','SKIP'):
        require(batch.get(status)==sum(r['status']==status for r in batch['episodes']),'Invalid compiler totals')
    accepted=[]
    for row in batch['episodes']:
        if row['status'] in ('PASS','SKIP'):
            folder=Path(row['output']);require(folder.resolve().is_relative_to(path.resolve()),'Compiler output outside attempt')
            entry,sample=accepted_episode(folder);require(entry['episode_id']==row['episode_id'],'Compiler episode identity mismatch')
            accepted.append(dict(episode_id=entry['episode_id'],path=str(folder),manifest_sha256=sha(folder/'training_manifest.json'),samples_sha256=sha(sample)))
    if 'accepted' in result:require(result['accepted']==accepted,'Previously validated compiler outputs changed')
    result['accepted']=accepted
    outcomes=[]
    for row in batch['episodes']:
        summary=Path(row['output'])/'validation_summary.json'
        if summary.is_file():
            v=read(summary);complete=v.get('native_execution_completed',v.get('physical',{}).get('replay_complete',False))
            outcome='accepted' if v.get('expert_valid') else 'task_unsuccessful' if complete and v.get('physical_success') is False else 'validation_failed' if complete else 'execution_failed'
            outcomes.append(dict(episode_id=row.get('episode_id',row.get('name')),outcome=outcome,native_execution_completed=complete,physical_success=v.get('physical_success'),expert_valid=v.get('expert_valid'),error=v.get('error')))
    result['episode_outcomes']=outcomes
    evidence=set(result.get('evidence',[]))
    for item in accepted:
        folder=Path(item['path']);evidence.update(str(folder/n) for n in ('training_manifest.json','validation_summary.json','validation/dataset_validation.json') if (folder/n).is_file())
        _,sample=accepted_episode(folder);evidence.add(str(sample))
    result['evidence']=sorted(evidence)
    if result.get('exit_code',0)==0:require(batch['FAIL']==0 and bool(accepted),'Zero-exit compiler output is incomplete or partial')
    return result
