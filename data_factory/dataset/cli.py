"""Phase B commands. There is deliberately no trainer/evaluator command."""
import argparse
import json
from pathlib import Path
import sys
import uuid
from .contracts import *
from .state import Experiment


def parser():
    p=argparse.ArgumentParser(prog='python -m data_factory.dataset.cli',description='Prepare canonical → existing V4 tensors; always stop before training.')
    sub=p.add_subparsers(dest='command',required=True)
    def common(q):
        q.add_argument('--config',type=Path,help='Explicit JSON configuration; CLI values override it')
        q.add_argument('--workspace',type=Path);q.add_argument('--experiment');q.add_argument('--dry-run',action='store_true')
    def compiler(q):
        q.add_argument('--references-root',type=Path);q.add_argument('--gpu',type=int)
        q.add_argument('--render-profile',choices=['auto','current','mthreads']);q.add_argument('--allow-software-rendering',action='store_true',default=None);q.add_argument('--retention',choices=['compact','training'])
    def dataset(q):
        q.add_argument('--canonical-root',type=Path);q.add_argument('--model',choices=['v4'])
        group=q.add_mutually_exclusive_group();group.add_argument('--split',type=Path);group.add_argument('--all-train',action='store_true',default=None);group.add_argument('--val-count',type=int)
        q.add_argument('--seed',type=int);q.add_argument('--incompatible-policy',choices=['error','exclude'])
        q.add_argument('--partial-policy',choices=['error','accepted'])
    q=sub.add_parser('compile');common(q);compiler(q);q.add_argument('--output-root',type=Path);q.add_argument('--mode',choices=['audit','production'])
    d=sub.add_parser('dataset').add_subparsers(dest='dataset_command',required=True)
    q=d.add_parser('build');common(q);dataset(q);q.add_argument('--output-root',type=Path)
    q=d.add_parser('audit');q.add_argument('--dataset',type=Path,required=True);q.add_argument('--output',type=Path);q.add_argument('--dry-run',action='store_true')
    q=sub.add_parser('train');common(q);q.add_argument('--dataset',type=Path,required=True);q.add_argument('--output',type=Path)
    e=sub.add_parser('eval').add_subparsers(dest='eval_command',required=True)
    q=e.add_parser('offline');q.add_argument('--dry-run',action='store_true');q.add_argument('--checkpoint',type=Path,required=True);q.add_argument('--dataset',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=e.add_parser('closed');q.add_argument('--dry-run',action='store_true');q.add_argument('--checkpoint',type=Path,required=True);q.add_argument('--scenarios',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--seed',type=int,default=4)
    q=sub.add_parser('run');common(q);compiler(q);dataset(q)
    q.add_argument('--canonical-output-root',type=Path);q.add_argument('--dataset-output-root',type=Path);q.add_argument('--stop-before',choices=['train'],default='train')
    q=sub.add_parser('resume');q.add_argument('--workspace',type=Path,default=Path('grasp_workspace'));q.add_argument('--experiment',required=True);q.add_argument('--dry-run',action='store_true')
    q.add_argument('--through',choices=['compile','dataset'],default='dataset')
    q=sub.add_parser('status');q.add_argument('--workspace',type=Path,default=Path('grasp_workspace'));q.add_argument('--experiment',required=True)
    return p


def configuration(args):
    raw=vars(args).copy();file=raw.pop('config',None);cfg=read(file) if file else {}
    allowed={'workspace','experiment','references_root','canonical_root','output_root','canonical_output_root','dataset_output_root','gpu','mode','render_profile','allow_software_rendering','retention','model','split','all_train','val_count','seed','incompatible_policy','partial_policy','dataset','output','epochs','batch_size','learning_rate','loss_weights','body_horizon','hand_horizon','execution_horizon','control_hz','replan_hz','offline_gate','seed','stop_before'}
    require(isinstance(cfg,dict) and not set(cfg)-allowed,'Unknown configuration fields: '+str(set(cfg)-allowed),'configuration')
    base=file.resolve().parent if file else Path.cwd()
    for key in ('workspace','references_root','canonical_root','output_root','canonical_output_root','dataset_output_root','split'):
        if key in cfg:cfg[key]=str((base/Path(cfg[key])).resolve())
    cfg.update({k:str(v.resolve()) if isinstance(v,Path) else v for k,v in raw.items() if v is not None})
    cfg.setdefault('workspace',str(Path('grasp_workspace').resolve()));cfg.setdefault('experiment','exp_'+uuid.uuid4().hex[:12])
    require(cfg['experiment'] not in ('.','..') and Path(cfg['experiment']).name==cfg['experiment'],'Experiment must be a simple ID','configuration')
    cfg.setdefault('model','v4');cfg.setdefault('gpu',0);cfg.setdefault('mode','audit');cfg.setdefault('render_profile','auto');cfg.setdefault('retention','compact')
    cfg.setdefault('incompatible_policy','error');cfg.setdefault('partial_policy','error')
    require(cfg['model']=='v4' and cfg['mode'] in ('audit','production') and cfg['partial_policy'] in ('error','accepted') and cfg['incompatible_policy'] in ('error','exclude'),'Invalid configuration choice','configuration')
    if cfg['command'] in ('run','dataset'):
        choices=[bool(cfg.get('split')),bool(cfg.get('all_train')),cfg.get('val_count') is not None]
        require(sum(choices)==1,'Choose exactly one explicit split: --split, --all-train, or --val-count with --seed','configuration')
        if cfg.get('split'):
            sp=read(cfg['split']);cfg['split_config']=dict(mode='explicit',train_episodes=sp['train_episodes'],val_episodes=sp['val_episodes'])
            cfg['split_file_sha256']=sha(cfg['split'])
        elif cfg.get('all_train'):cfg['split_config']={'mode':'all_train'}
        else:
            require(type(cfg.get('seed'))==int and type(cfg['val_count'])==int and cfg['val_count']>0,'--val-count requires positive count and explicit integer --seed','configuration')
            cfg['split_config']=dict(mode='count',seed=cfg['seed'],val_count=cfg['val_count'])
    if cfg['command']=='train':require(cfg.get('dataset') and cfg.get('output'),'train requires --dataset and --output','configuration')
    if cfg['command']=='compile':require(cfg.get('references_root') and cfg.get('output_root'),'compile requires --references-root and --output-root','configuration')
    if cfg['command']=='dataset':require(cfg.get('canonical_root') and cfg.get('output_root'),'dataset build requires --canonical-root and --output-root','configuration')
    if cfg['command']=='run':
        require(bool(cfg.get('references_root')) != bool(cfg.get('canonical_root')),'run requires exactly one of --references-root or --canonical-root','configuration')
        require(cfg['mode']=='audit','V4 run uses existing full audit retention; production compact is insufficient','configuration')
    if file:
        cfg['config']=str(file.resolve());cfg['configuration_source']=dict(path=str(file.resolve()),sha256=sha(file))
    return cfg


def status(path):
    m=Experiment(path).load();result=dict(experiment=m['experiment_id'],manifest=str(Path(path)/'experiment_manifest.json'),status=m['milestone'],training_authorized=False,stages={})
    for name,s in m['stages'].items():
        row=dict(state=s['state'],attempts=len(s['attempts']))
        if s['attempts']:
            record=read(s['attempts'][-1]);row.update(error=record['error'],exit_code=record['exit_code'],output=(record.get('result') or {}).get('path'),log=record['log'])
        if s.get('reason'):row['reason']=s['reason']
        result['stages'][name]=row
    return result


def drive(exp,through='dataset'):
    from .adapters import reference_identity,canonical_identity,compiler_command,run_compiler,validate_compile
    from .dataset import build,validate
    exp.recover();cfg=exp.load()['config'];canonical=cfg.get('canonical_root');partial=False
    if canonical:
        if exp.load()['stages']['compile']['state']=='NOT_STARTED':exp.external_canonical(canonical)
    else:
        exp.set_milestone('COMPILING')
        input_error=None
        try:inputs=reference_identity(cfg['references_root'])
        except Exception as exc:input_error=exc;inputs=dict(path=cfg['references_root'],hashes={})
        old=exp.load()['stages']['compile']
        # Explicit accepted-subset policy reuses a previously failed but validated batch; no hidden rerun.
        prior=read(old['attempts'][-1]) if old['attempts'] else None
        if prior and prior['state']=='FAILED' and (prior.get('error') or {}).get('kind')=='partial_batch' and cfg['partial_policy']=='accepted':
            verify_files(prior['output_hashes']);validate_compile(prior['result']);record=prior
        else:
            attempt=len(old['attempts'])+1
            root=Path(cfg.get('output_root') if cfg['command']=='compile' else cfg.get('canonical_output_root') or Path(cfg['workspace'])/'artifacts/canonical')
            out=root/f'{cfg["experiment"]}_attempt_{attempt:04d}'
            def action(directory,log,child):
                if input_error:raise input_error
                require(not out.exists(),'Compiler output exists; retry must use a fresh attempt','invalid_artifact')
                result=run_compiler(cfg['references_root'],out,cfg,log,child,exp.lock.fileno());verify_files(inputs['hashes']);return result
            record=exp.execute('compile',inputs,cfg,compiler_command(cfg['references_root'],out,cfg),action,validate_compile)
        if record['state']!='SUCCEEDED':
            partial=(record.get('error') or {}).get('kind')=='partial_batch' and cfg['partial_policy']=='accepted' and bool((record.get('result') or {}).get('accepted'))
            if not partial:
                exp.block('dataset_build','Compiler failed; dependent stage not run');exp.set_milestone('COMPILATION_FAILED');return 2
        canonical=record['result']['path']
    exp.set_milestone('CANONICAL_READY' if not partial else 'CANONICAL_PARTIAL_ACCEPTED')
    if through=='compile' or cfg['command']=='compile':return 2 if partial else 0
    exp.set_milestone('BUILDING_DATASET');input_error=None
    try:identity=canonical_identity(canonical)
    except Exception as exc:input_error=exc;identity=dict(path=str(canonical),hashes={})
    old=exp.load()['stages']['dataset_build'];attempt=len(old['attempts'])+1
    root=Path(cfg.get('output_root') if cfg['command']=='dataset' else cfg.get('dataset_output_root') or Path(cfg['workspace'])/'artifacts/datasets')
    out=root/f'dataset_{cfg["experiment"]}_{attempt:04d}'
    builder_cfg=dict(model='v4',split=cfg['split_config'],incompatible_policy=cfg['incompatible_policy'],partial_policy=cfg['partial_policy'])
    def action(directory,log,child):
        if input_error:raise input_error
        print('Reusing v4.dataset.CanonicalV4; training is not implemented in Phase B.',flush=True)
        manifest=build(canonical,out,builder_cfg);verify_files(identity['hashes'])
        return dict(path=str(out.resolve()),artifact_id=manifest['dataset_id'],exit_code=0,evidence=[str(out/n) for n in ('COMPLETED.json','manifest.json','samples.npz','split.json','index.jsonl')])
    result=exp.execute('dataset_build',identity,builder_cfg,dict(entrypoint='data_factory.dataset.build -> va_research.models.v4.dataset.CanonicalV4',canonical_root=str(canonical),output=str(out)),action,lambda r:validate(r['path']))
    if result['state']!='SUCCEEDED':exp.set_milestone('DATASET_FAILED');return 2
    exp.set_milestone('READY_FOR_TRAINING')
    # Partial compiler failure stays visible in stage state AND overall exit status.
    return 2 if partial else 0


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='dataset' and getattr(args,'dataset_command',None)=='audit':
            from .audit import build
            out=args.output or args.dataset.parent/(args.dataset.name+'_audit')
            if args.dry_run: print(json.dumps(dict(dry_run=True,dataset=str(args.dataset),output=str(out),training=False),indent=2));return 0
            build(args.dataset,out);print(json.dumps(dict(status='SUCCEEDED',report=str(out/'report.html'),summary=str(out/'summary.json')),indent=2));return 0
        if args.command=='eval':
            if args.eval_command=='offline':
                if args.dry_run: print(json.dumps(dict(dry_run=True,training=False),indent=2));return 0
                from .evaluation import offline;met=offline(args.checkpoint,args.dataset,args.output);print(json.dumps(dict(status='SUCCEEDED',metrics=str(args.output/'metrics.json'),report=str(args.output/'report.html')),indent=2));return 0
            if args.dry_run: print(json.dumps(dict(dry_run=True,training=False),indent=2));return 0
            from execution.sonic_mujoco.rendering import bootstrap
            bootstrap('auto')
            from va_research.models.v4.rollout import run
            # Native evaluator itself remains authoritative; scenario file is provenance only.
            # NativeTokenSonic changes cwd to the frozen deploy repository;
            # pass absolute paths so its bootstrap and log paths remain valid.
            out=Path(args.output).resolve()
            checkpoint=Path(args.checkpoint).resolve()
            require(not out.exists(),'Evaluation output already exists; use a fresh directory','invalid_artifact')
            scenario=read(args.scenarios)
            require(isinstance(scenario,dict) and scenario,'Scenario config must be a non-empty JSON object','configuration')
            provenance=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),seed=args.seed,scenario_sha256=sha(args.scenarios),substitution=False)
            try:
                report=run(checkpoint,out)
                write(out/'scenario.json',scenario);write(out/'evaluation_provenance.json',provenance)
                runtime_succeeded=bool(report.get('failure') is None and report.get('rollout_complete') and report.get('strict_runtime_pass'))
                write(out/'metrics.json',dict(runtime_succeeded=runtime_succeeded,grasp_success=report.get('physical_grasp_success'),
                    max_lift_m=report.get('max_lift_m'),hold_seconds=report.get('max_continuous_hold_seconds'),
                    recorded_ticks=report.get('recorded_ticks'),expected_ticks=report.get('expected_ticks'),
                    failure=report.get('failure'),scheduler=report.get('scheduler'),checkpoint=str(checkpoint),scenario=str(args.scenarios.resolve())))
                import html
                (out/'report.html').write_text('<html><meta charset="utf-8"><title>Closed-loop evaluation</title><h1>Closed-loop evaluation</h1><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></html>')
                write(out/'execution_status.json',dict(state='SUCCEEDED' if runtime_succeeded else 'FAILED',runtime_succeeded=runtime_succeeded,grasp_success=report.get('physical_grasp_success'),report=str(out/'report.json')))
                return 0 if report.get('strict_runtime_pass') else 2
            except BaseException as exc:
                out.mkdir(parents=True,exist_ok=True)
                write(out/'scenario.json',scenario);write(out/'evaluation_provenance.json',provenance)
                write(out/'execution_status.json',dict(state='FAILED',runtime_started=(out/'native_launch.json').exists(),runtime_succeeded=False,grasp_success=None,error_type=type(exc).__name__,error=str(exc),report=str(out/'report.json') if (out/'report.json').exists() else None))
                raise
        if args.command=='train':
            cfg=configuration(args);out=Path(cfg['output'])
            if args.dry_run: print(json.dumps(dict(dry_run=True,dataset=cfg['dataset'],output=str(out),training=True),indent=2));return 0
            from .training import run
            from .reports import training_report
            result=run(cfg['dataset'],out,Path(cfg['config']) if cfg.get('config') else None);training_report(out);print(json.dumps(dict(status='SUCCEEDED',**result,report=str(out/'report.html')),indent=2));return 0
        if args.command in ('status','resume'):
            require(Path(args.experiment).name==args.experiment and args.experiment not in ('.','..'),'Invalid experiment ID','configuration')
            path=args.workspace.resolve()/'experiments'/args.experiment
            if args.command=='status':print(json.dumps(status(path),indent=2));return 0
            if args.dry_run:
                m=Experiment(path).load();print(json.dumps(dict(dry_run=True,experiment=m['experiment_id'],through=args.through,config=m['config'],training=False),indent=2));return 0
            with Experiment(path) as exp:code=drive(exp,args.through)
            print(json.dumps(status(path),indent=2));return code
        cfg=configuration(args);path=Path(cfg['workspace'])/'experiments'/cfg['experiment']
        if args.dry_run:
            print(json.dumps(dict(dry_run=True,experiment_manifest=str(path/'experiment_manifest.json'),config=cfg,
                stages=['compile'] if cfg['command']=='compile' else ['compile','dataset_build'],stop_before='train',training=False),indent=2));return 0
        exp=Experiment.create(path,cfg['experiment'],cfg)
        with exp:code=drive(exp)
        print(json.dumps(status(path),indent=2));return code
    except (PipelineError,FileNotFoundError,FileExistsError,KeyError,ValueError) as exc:
        print(json.dumps(dict(error=str(exc),kind=getattr(exc,'kind','invalid_artifact'))),file=sys.stderr)
        return getattr(exc,'code',2)

if __name__=='__main__':raise SystemExit(main())
