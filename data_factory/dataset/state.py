"""Atomic experiment state with immutable attempt records and a process-held lock."""
import contextlib
import fcntl
import os
from pathlib import Path
import traceback
from .contracts import *

class Experiment:
    def __init__(self, path):self.path=Path(path);self.manifest=self.path/'experiment_manifest.json';self.lock=None
    @classmethod
    def create(cls,path,experiment_id,config):
        obj=cls(path);obj.path.mkdir(parents=True,exist_ok=False)
        from . import __version__
        code=files_record(Path(__file__).parent.glob('*.py'))
        write(obj.manifest,dict(pipeline_version=__version__,implementation_hashes=code,schema_version=VERSION,artifact_type='experiment',experiment_id=experiment_id,created_at=now(),
            config=config,milestone='CREATED',stages={s:dict(state='NOT_STARTED',attempts=[]) for s in STAGES},training_authorized=False))
        return obj
    def load(self):return validate_experiment(read(self.manifest))
    def save(self,m):validate_experiment(m);write(self.manifest,m)
    def __enter__(self):
        self.lock=(self.path/'experiment.lock').open('a')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close();raise PipelineError('Experiment is already running','busy',3)
        return self
    def __exit__(self,*args):
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close();self.lock=None
    def recover(self):
        m=self.load()
        for name,stage in m['stages'].items():
            if stage['state']!='RUNNING':continue
            path=Path(stage['attempts'][-1]);record=read(path)
            # Child inherits our lock until it exits. Reaching here means no process owns that lock.
            record.update(state='FAILED',ended_at=now(),exit_code=130,error=dict(kind='interrupted',message='Previous attempt lost its process lock; retained, never promoted without validation'))
            write(path,record);stage['state']='FAILED'
        self.save(m)
    def set_milestone(self,value):
        m=self.load();m['milestone']=value;self.save(m)
    def external_canonical(self,path):
        m=self.load();m['stages']['compile'].update(state='SKIPPED',external_artifact=str(Path(path).resolve()),reason='Explicit existing canonical input; not a new compilation');self.save(m)
    def block(self,stage,message):
        m=self.load();m['stages'][stage].update(state='BLOCKED',reason=message);self.save(m)
    def execute(self,stage,inputs,config,command,action,validator):
        require(self.lock is not None,'Experiment execution requires lock','execution')
        m=self.load();s=m['stages'][stage]
        if s['state']=='SUCCEEDED':
            old=read(s['attempts'][-1])
            try:
                require(old['inputs']==inputs and old['config']==config,'Completed stage inputs/config changed; create a new experiment')
                verify_files(old['output_hashes']);validator(old['result'])
            except Exception as exc:
                s.update(state='BLOCKED',reason='Completed artifact cannot be reused: '+str(exc))
                m['milestone']='ARTIFACT_BLOCKED';self.save(m)
                raise PipelineError(s['reason']) from exc
            return old
        directory=self.path/'stages'/stage/f'attempt_{len(s["attempts"])+1:04d}'
        directory.mkdir(parents=True,exist_ok=False);record_path=directory/'attempt.json'
        record=dict(schema_version=VERSION,stage=stage,state='RUNNING',attempt=len(s['attempts'])+1,started_at=now(),ended_at=None,
                    inputs=inputs,config=config,command=command,exit_code=None,output_hashes={},result=None,error=None,log=str(directory/'execution.log'),owner_pid=os.getpid())
        write(record_path,record);s['state']='RUNNING';s['attempts'].append(str(record_path.resolve()));self.save(m)
        def child_started(pid):record['child_pid']=pid;write(record_path,record)
        with (directory/'execution.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                result=action(directory,log,child_started);record['result']=result
                record['exit_code']=result.get('exit_code',0)
                record['output_hashes']=files_record(result.get('evidence',[]))
                try:validator(result)
                except PipelineError:raise
                except Exception as exc:raise PipelineError('Output artifact validation failed: '+str(exc)) from exc
                record['output_hashes']=files_record(result.get('evidence',[]))
                if record['exit_code']!=0:
                    raise PipelineError(result.get('error','Process exited '+str(record['exit_code'])),result.get('error_kind','execution'))
                record['state']='SUCCEEDED'
            except BaseException as exc:
                record['state']='FAILED';record['exit_code']=record['exit_code'] or (130 if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 2)
                record['error']=dict(kind=getattr(exc,'kind','interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'execution'),message=str(exc))
                traceback.print_exc()
            finally:
                record['ended_at']=now();write(record_path,record)
                m=self.load();m['stages'][stage]['state']=record['state'];self.save(m)
        return record
