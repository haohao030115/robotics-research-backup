"""Single fail-closed production transaction; unsuccessful truth is retained for audit."""
import argparse
import shutil
from pathlib import Path
from .common import dump, read, digest, ROOT
from .schema import validate

FLAGS=('reference_schema_valid','camera_valid','physical_success','vision_alignment_valid',
       'encoder_reconstruction_valid','decoder_reconstruction_valid','command_trace_valid',
       'state_transition_valid','future_target_alignment_valid','schema_valid','dataset_valid','expert_valid')


def report(out,flags,source=None,replay=None,temporal=None,error=None):
    source=source or {}; replay=replay or {}; temporal=temporal or {}
    sections=[('Source reference summary',{'episode_id':source.get('episode_id'),'sha256':source.get('source_reference_sha256')}),
        ('Producer type',source.get('producer_type')),('Schema validation',flags['reference_schema_valid']),
        ('Canonical runtime config',read(out/'canonical_replay/metadata.json').get('runtime') if (out/'canonical_replay/metadata.json').exists() else 'Not started'),
        ('Actual camera config',read(out/'canonical_replay/metadata.json')['runtime']['camera'] if (out/'canonical_replay/metadata.json').exists() else 'Not loaded'),
        ('Replay timeline',{'ticks':replay.get('control_ticks'),'physics_steps':replay.get('physics_steps'),'file':'canonical_replay/timeline.jsonl'}),
        ('Observation semantic','PRE_ACTION physical snapshot before inference. State history ends at this observation.'),
        ('Encoder semantic','Frozen release mode 0; saved exact input and current post-FSQ encoded_tokens 64D output.'),
        ('Decoder semantic','Frozen release; previous_action is separate from same-decision current_raw_29d_action.'),
        ('Command-to-physics mapping',temporal.get('command_trace')),
        ('RGB-state alignment',temporal.get('vision_alignment')),
        ('Encoder reconstruction',temporal.get('encoder_reconstruction')),
        ('Decoder reconstruction',temporal.get('decoder_reconstruction')),
        ('No-off-by-one verification',temporal.get('no_off_by_one')),
        ('VA future target mapping',temporal.get('future_target_alignment')),
        ('Physical success metrics',replay),('Dataset validity',flags),('PASS/FAIL summary',{'pass':flags['expert_valid'],'error':error})]
    import json
    text='# Unified canonical VA episode report\n\n'
    for i,(name,value) in enumerate(sections,1):
        text+=f'## {i}. {name}\n\n```json\n{json.dumps(value,indent=2,ensure_ascii=False)}\n```\n\n'
    text+='Execution uses the original native controller with logging-only instrumentation, DDS transport and original TabletopDDS.step, startup WBC and standby flow. The synchronous reimplementation is not called. Timing is asynchronous; real publication, receipt, consumption and hand cursor are retained. No batch was run.\n\n'
    text+='Execution conclusion: '+('NATIVE_SONIC_REUSED_AND_VALIDATED' if flags['expert_valid'] else 'EXECUTION_NOT_EQUIVALENT')+'\n\n'
    text+=('READY_FOR_UNIFIED_DATASET_PRODUCTION' if flags['expert_valid'] else 'NOT_READY_FOR_UNIFIED_DATASET_PRODUCTION')+'\n'
    (out/'REPORT.md').write_text(text)


def run(reference,out,gpu=0,recorded_episode=None,mode='audit',allow_software_rendering=False,
        qa_rate=0.,qa_registry=None,position_bin_size=None,retention='compact'):
    import time
    from .common import require
    from .versioning import check_policy
    from .rendering import probe,renderer_allowed
    from .production import basic_validate,pack_samples,compact_success,training_success,metadata_fields,qa_decision,mark_qa
    require(position_bin_size is None or (__import__('math').isfinite(position_bin_size) and position_bin_size>0),'position_bin_size must be positive')
    require(0<=qa_rate<=1,'QA sample rate must be in [0,1]')
    require(retention in ('compact','training'),'Unknown retention')
    if out.exists(): raise FileExistsError('Output must be fresh; accepted episodes are never overwritten: '+str(out))
    started=time.perf_counter();out.mkdir(parents=True);(out/'validation').mkdir()
    flags=dict.fromkeys(FLAGS,False);meta={};physical={};temporal={};error=None;version={};render={};sample_schema={}
    timings={k:0. for k in ('startup','native_rollout','rgb_rendering','sample_compilation','validation','disk_writing','total')}
    registry=Path(qa_registry) if qa_registry else out.parent/'qa_registry.json'
    full_qa=mode=='audit';qa_reason='audit mode';qa_key=None;basic={};reference_hashes={}
    try:
        require(mode in ('audit','production'),'Unknown mode')
        require(not (mode=='production' and recorded_episode is not None),'Production requires new native execution; --recorded-episode is audit-only')
        t=time.perf_counter();a,meta,schema=validate(reference)
        reference_hashes={n:digest(reference/n) for n in ('reference.npz','metadata.json','provenance.json','reference_validation.json')}
        shutil.copytree(reference,out/'source_reference',ignore=lambda path,names:[n for n in names if n not in reference_hashes])
        (out/'source_reference').chmod(0o755)
        a,meta,schema=validate(out/'source_reference');flags['reference_schema_valid']=True
        dump(out/'validation/reference_validation.json',schema);meta['source_reference_sha256']=digest(out/'source_reference/reference.npz')
        version=check_policy();timings['validation']+=time.perf_counter()-t
        t=time.perf_counter();render=probe();render['warnings']=renderer_allowed(render,mode,allow_software_rendering)
        dump(out/'validation/renderer.json',render);timings['validation']+=time.perf_counter()-t
        if mode=='production':full_qa,qa_reason,qa_key=qa_decision(version,render,meta['episode_id'],meta['source_reference_sha256'],registry,qa_rate)
        if recorded_episode is None:
            from .replay import replay
            physical=replay(a,meta,out/'canonical_replay',gpu=gpu)
        else:
            recorded_episode=Path(recorded_episode).resolve()
            for name,h in reference_hashes.items():require(digest(recorded_episode/'source_reference'/name)==h,'Recorded source differs: '+name)
            recorded=recorded_episode/'canonical_replay';require(read(recorded/'metadata.json')['runtime']['backend']=='native_dds','Recorded input is not native replay')
            shutil.copytree(recorded,out/'canonical_replay');physical=read(out/'canonical_replay/replay_validation.json')
            dump(out/'validation/recorded_replay_reuse.json',dict(source_episode=str(recorded_episode),new_physics_steps=0,
                recording_hashes={str(p.relative_to(recorded)):digest(p) for p in recorded.rglob('*') if p.is_file()}))
            from .native_truth import materialize
            physical=materialize(out/'canonical_replay',a,meta,physical)
        for key,value in physical.get('stage_timings',{}).items():timings[key]=timings.get(key,0.)+value
        flags['camera_valid']=physical['camera_valid'];flags['physical_success']=physical['physical_success']
        t=time.perf_counter();basic=basic_validate(out/'canonical_replay',a,meta,physical);timings['validation']+=time.perf_counter()-t
        if not basic['physical_success']:full_qa=True;qa_reason='failure forensic retention'
        if full_qa:
            from .forensic import verify
            t=time.perf_counter();temporal=verify(out/'canonical_replay',a,meta);timings['validation']+=time.perf_counter()-t
            for key in ('vision_alignment','encoder_reconstruction','decoder_reconstruction','command_trace','state_transition','future_target_alignment','schema'):
                flags[key+'_valid']=temporal[key]['valid']
            flags['schema_valid'] &= temporal['no_off_by_one']['valid'] and temporal['physical_reconstruction']['valid'] and physical['replay_complete']
            flags['dataset_valid']=all(flags[k] for k in ('camera_valid','vision_alignment_valid','encoder_reconstruction_valid','decoder_reconstruction_valid','command_trace_valid','state_transition_valid','future_target_alignment_valid','schema_valid'))
        else:
            # Unperformed reconstruction is NULL, never falsely reported as passed.
            for k in ('encoder_reconstruction_valid','decoder_reconstruction_valid','state_transition_valid'):flags[k]=None
            flags.update(schema_valid=True,vision_alignment_valid=True,command_trace_valid=True,future_target_alignment_valid=True,dataset_valid=True)
        flags['expert_valid']=bool(basic['physical_success'] and flags['dataset_valid'])
        if not flags['expert_valid']:raise ValueError('Episode failed physical or dataset gate; complete evidence retained')
        from .compiler import compile_samples
        t=time.perf_counter();mappings=compile_samples(out/'canonical_replay',out/'va_dataset',flags);validate_export(out)
        timings['sample_compilation']=time.perf_counter()-t
        t=time.perf_counter();sample_schema=pack_samples(out/'va_dataset');timings['disk_writing']+=time.perf_counter()-t
        entry=dict(metadata_fields(meta,len(mappings),True,position_bin_size),path='va_dataset',**version)
        dump(out/'training_manifest.json',dict(schema_version=2,sample_contract='canonical-va-samples-v2',episodes=[entry]))
        # Check code and reference again before publishing the accepted status.
        ending=check_policy(emit=False);require(ending['source_hashes']==version['source_hashes'],'Sources changed during episode')
        require(all(digest(reference/n)==h for n,h in reference_hashes.items()),'Reference changed during production')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';flags['dataset_valid']=False;flags['expert_valid']=False
        if (out/'canonical_replay/replay_validation.json').exists():physical=read(out/'canonical_replay/replay_validation.json')
        if mode=='production' and not temporal and (out/'canonical_replay/observations.npz').exists():
            full_qa=True;qa_reason='failure forensic retention'
            try:
                from .forensic import verify
                t=time.perf_counter();temporal=verify(out/'canonical_replay',a,meta);timings['validation']+=time.perf_counter()-t
            except Exception as audit_error:temporal={'error':str(audit_error)}
        if (out/'training_manifest.json').exists():
            # A failure after export cannot leave an accepted manifest.
            rejected=read(out/'training_manifest.json')
            for e in rejected['episodes']:e['expert_valid']=False
            dump(out/'training_manifest.json',rejected)
    finally:
        timings['total']=time.perf_counter()-started
        dump(out/'validation/replay_validation.json',physical or {'replay_complete':False,'error':error})
        dump(out/'validation/temporal_validation.json',temporal or {'full_reconstruction_performed':False})
        dump(out/'validation/dataset_validation.json',dict(flags,error=error))
        if not (out/'validation/reference_validation.json').exists():dump(out/'validation/reference_validation.json',dict(reference_schema_valid=flags['reference_schema_valid'],error=error))
        essential=dict(version,reference_source_hashes=reference_hashes,source_reference=str(reference),mode=mode,retention=retention,
            metadata=metadata_fields(meta,sample_schema.get('sample_count',0),flags['physical_success'],position_bin_size) if meta.get('case') and meta.get('episode_id') and meta.get('producer_type') else {},
            renderer=render,native_execution=read(out/'canonical_replay/provenance.json') if (out/'canonical_replay/provenance.json').exists() else None)
        dump(out/'provenance.json',essential)
        summary=dict(flags,mode=mode,retention=retention,error=error,native_execution_completed=bool(physical.get('replay_complete')),
            new_native_execution=recorded_episode is None and bool(physical.get('replay_complete')),basic_checks=basic,
            sample_compilation_valid=bool(sample_schema) and flags['expert_valid'],sample_schema=sample_schema,
            qa=dict(full_reconstruction_performed=bool(temporal) and 'error' not in temporal,required=full_qa,reason=qa_reason,key=qa_key),
            forensic=temporal,renderer=render,physical=physical,timings_seconds=timings,**version)
        dump(out/'validation_summary.json',summary)
        report(out,flags,meta,physical,temporal,error)
    if flags['expert_valid']:
        if mode=='production':
            t=time.perf_counter();(training_success if retention=='training' else compact_success)(out);timings['disk_writing']+=time.perf_counter()-t;timings['total']=time.perf_counter()-started
            summary['timings_seconds']=timings;dump(out/'validation_summary.json',summary)
            if full_qa:mark_qa(registry,qa_key,out)
    print('READY_FOR_UNIFIED_DATASET_PRODUCTION' if flags['expert_valid'] else 'NOT_READY_FOR_UNIFIED_DATASET_PRODUCTION',flush=True)
    if error:print(error,flush=True)
    return 0 if flags['expert_valid'] else 2


def validate_export(out):
    import numpy as np
    from .common import load_npz, lines, index, require
    from .compiler import plan_samples, STATE_FIELDS
    truth=out/'canonical_replay'; dataset=out/'va_dataset'
    expected=plan_samples(truth); actual=lines(dataset/'sample_mapping.jsonl')
    require(actual==expected,'Exported sample mappings changed')
    z=load_npz(dataset/'samples.npz'); o=load_npz(truth/'observations.npz'); r=load_npz(truth/'references.npz'); e=load_npz(truth/'sonic_encoder.npz')
    oi=index(o,'observation_id'); hi=index(r,'hand_ref_id'); zi=index(e,'token_id')
    require(z['sample_id'].tolist()==[s['sample_id'] for s in expected],'Sample identity mismatch')
    for j,s in enumerate(actual):
        require(digest(dataset/s['rgb_path'])==digest(truth/s['rgb_path']),'Exported RGB changed')
        require(np.array_equal(z['history_valid_mask'][j],s['history_valid_mask']),'Padding mask mismatch')
        for field in STATE_FIELDS:
            require(np.array_equal(z['state_history_'+field][j],[o[field][oi[x]] for x in s['state_history_observation_ids']]),'Exported history mismatch')
        require(np.array_equal(z['body_target'][j],[e['current_post_fsq_64d_token'][zi[x]] for x in s['body_token_ids']]),'Exported body target mismatch')
        require(np.array_equal(z['hand_target'][j],[r['hand_ref_q'][hi[x]] for x in s['hand_ref_ids']]),'Exported hand target mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path); parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--backend',choices=['native'],default='native',help='Only original native SONIC is supported; no synchronous fallback')
    parser.add_argument('--gpu',type=int,default=0,help='Physical MUSA device for native controller')
    parser.add_argument('--recorded-episode',type=Path,help='Revalidate and compile immutable native trace into a fresh output; never runs physics')
    parser.add_argument('--mode',choices=['audit','production'],default='audit')
    parser.add_argument('--allow-software-rendering',action='store_true')
    parser.add_argument('--render-profile',choices=['auto','current','mthreads'],default='auto')
    parser.add_argument('--qa-rate',type=float,default=0.)
    parser.add_argument('--qa-registry',type=Path)
    parser.add_argument('--position-bin-size',type=float)
    parser.add_argument('--retention',choices=['compact','training'],default='compact')
    args=parser.parse_args()
    from .rendering import bootstrap
    bootstrap(args.render_profile)
    return run(args.reference.resolve(),args.output.resolve(),gpu=args.gpu,recorded_episode=args.recorded_episode,
        mode=args.mode,allow_software_rendering=args.allow_software_rendering,qa_rate=args.qa_rate,qa_registry=args.qa_registry,position_bin_size=args.position_bin_size,retention=args.retention)


if __name__=='__main__':raise SystemExit(main())
