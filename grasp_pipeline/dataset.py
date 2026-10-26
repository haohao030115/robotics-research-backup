"""Serialize existing CanonicalV4 tensors without implementing horizons or labels."""
import json
import random
import sys
from pathlib import Path
from .contracts import require, read, write, sha, fingerprint, files_record, verify_files, seal, verify_seal, VERSION, PipelineError
from . import __version__

ROOT=Path(__file__).resolve().parents[1]
FULL=('source_reference/metadata.json','canonical_replay/metadata.json','canonical_replay/sonic_encoder.npz',
      'canonical_replay/references.npz','canonical_replay/observations.npz')

def baseline_class():
    # Resolve this repository's existing package, not a user's working directory.
    sys.path.insert(0,str(ROOT/'va_training'))
    from v4.dataset import CanonicalV4
    return CanonicalV4

def candidates(root):
    root=Path(root).resolve();require(root.is_dir(),'Canonical root missing: '+str(root))
    if (root/'training_manifest.json').is_file():
        m=read(root/'training_manifest.json')
        if 'sources' not in m:return [root]
        paths=[Path(p) if Path(p).is_absolute() else root/p for p in m['sources']]
        require(len(paths)==m['num_episodes'],'Merged source count differs')
        require(len(set(paths))==len(paths),'Repeated merged source path')
        require(all(p.name=='samples.npz' and p.is_file() for p in paths),'Missing merged canonical source samples')
        return [p.parents[1] for p in paths]
    # Episode directories are direct batch children; follow explicit episode symlinks too.
    return sorted(p for p in root.iterdir() if p.is_dir() and (p/'training_manifest.json').is_file())

def source_info(folder):
    from canonical_va.production import accepted_episode
    entry,samples=accepted_episode(folder)
    data=samples.parent
    missing=[n for n in FULL if not (folder/n).is_file()]
    missing += [str(p.relative_to(folder)) for p in (data/'metadata.json',data/'sample_mapping.jsonl') if not p.is_file()]
    require(not missing,'INSUFFICIENT_CANONICAL_RETENTION: '+str(folder)+'; missing '+', '.join(missing)+'; use existing audit/full-retention output. No future-label fabrication.')
    source=read(folder/'source_reference/metadata.json')
    normalized=dict(source)
    reference=folder/'source_reference'
    if all((reference/n).is_file() for n in ('reference.npz','metadata.json','provenance.json','reference_validation.json')):
        from canonical_va.schema import validate as validate_reference
        _,normalized,_=validate_reference(reference)
    arm=entry.get('task_arm',normalized.get('task_arm','right'))
    require(arm=='right','UNSUPPORTED_V4_TASK_ARM: '+str(arm)+'; baseline phase/progress labels are right-hand-specific','incompatible_v4')
    rows=[json.loads(x) for x in (data/'sample_mapping.jsonl').read_text().splitlines() if x]
    paths=[folder/'training_manifest.json',samples,data/'metadata.json',data/'sample_mapping.jsonl',*[folder/n for n in FULL]]
    for row in rows:
        p=data/row['rgb_path'];require(p.resolve().is_relative_to(data.resolve()),'Unsafe RGB source path');paths.append(p)
    for n in ('provenance.json','validation_summary.json','validation/dataset_validation.json','source_reference/provenance.json','source_reference/reference_validation.json','source_reference/reference.npz'):
        if (folder/n).is_file():paths.append(folder/n)
    hashes=files_record(paths)
    return dict(entry=entry,source=source,canonical_metadata=normalized,hashes=hashes,canonical_id=entry['episode_id'],
                artifact_id='canonical:'+entry['episode_id']+':'+fingerprint(hashes)[:16],path=str(folder.resolve()))

def make_split(ids, cfg):
    ids=list(ids);require(len(set(ids))==len(ids),'Canonical episode IDs must be unique')
    mode=cfg.get('mode');seed=cfg.get('seed')
    if mode=='explicit':
        train=cfg['train_episodes'];val=cfg['val_episodes']
    elif mode=='all_train':train=ids.copy();val=[]
    elif mode=='count':
        n=cfg['val_count'];require(type(seed)==int and type(n)==int and 0<n<len(ids),'Explicit split requires integer seed and 0 < val_count < episodes')
        ordered=sorted(ids);random.Random(seed).shuffle(ordered);val=ordered[:n];train=ordered[n:]
    else:raise PipelineError('Specify --split, --all-train, or --val-count plus --seed','configuration')
    require(isinstance(train,list) and isinstance(val,list) and bool(train),'A nonempty training split is required')
    require(len(set(train))==len(train) and len(set(val))==len(val),'Repeated split IDs')
    require(not set(train)&set(val) and set(train)|set(val)==set(ids),'Split must cover included episodes exactly and be episode-disjoint')
    return dict(schema_version=VERSION,mode=mode,seed=seed,train_episodes=sorted(train),val_episodes=sorted(val),
                held_out_validation=bool(val),normalization='Not applied here; baseline trainer uses train-only mean/std clamp_min(.05).')

def build(canonical_root, output, config):
    import numpy as np
    output=Path(output);require(not output.exists(),'Dataset output already exists; immutable: '+str(output))
    output.mkdir(parents=True)
    require(config.get('model','v4')=='v4','Only existing V4 is supported')
    batch=Path(canonical_root)/'batch_summary.json'
    batch_report=read(batch) if batch.exists() else None
    require(not batch_report or not batch_report.get('FAIL') or config.get('partial_policy')=='accepted',
            'PARTIAL_BATCH: --partial-policy accepted is required to consume accepted subset','partial_batch')
    sources=[];excluded=[]
    merged_manifest=Path(canonical_root)/'training_manifest.json'
    merged_info=read(merged_manifest) if merged_manifest.exists() else {}
    merged_hashes={}
    if 'sources' in merged_info:
        merged_path=Path(merged_info['output'])
        merged_hashes=files_record([merged_manifest,merged_path])
        with np.load(merged_path,allow_pickle=False) as z: merged={k:z[k] for k in z.files}
        ids=merged['sample_id'].tolist()
        require(len(set(ids))==len(ids),'Merged sample IDs repeat')
        positions={v:i for i,v in enumerate(ids)}
        seen=[]
        for folder in candidates(canonical_root):
            with np.load(folder/'va_dataset/samples.npz',allow_pickle=False) as z:
                require(set(z.files)==set(merged),'Merged/source keys differ')
                rows=[positions[x] for x in z['sample_id']];seen.extend(rows)
                require(all(np.array_equal(merged[k][rows],z[k]) for k in z.files),'Merged/source tensor mismatch')
        require(sorted(seen)==list(range(len(ids))),'Merged source rows do not cover samples exactly')
    for folder in candidates(canonical_root):
        m=read(folder/'training_manifest.json')
        # Existing compiler is the authoritative gate; a failed episode is never requalified here.
        if len(m.get('episodes',[]))==1 and m['episodes'][0].get('expert_valid') is not True:
            excluded.append(dict(path=str(folder),reason='compiler expert_valid is not true'));continue
        try:sources.append(source_info(folder))
        except PipelineError as exc:
            if exc.kind=='incompatible_v4' and config.get('incompatible_policy')=='exclude':
                excluded.append(dict(path=str(folder),reason=str(exc)));continue
            raise
    require(bool(sources),'No accepted compatible full canonical episodes')
    ids=[s['canonical_id'] for s in sources];split=make_split(ids,config['split'])
    implementation=files_record([ROOT/'va_training/v4/dataset.py',Path(__file__),ROOT/'grasp_pipeline/contracts.py',ROOT/'canonical_va/schema.py'])
    chunks=[];entries=[];index=[];count=0;CanonicalV4=baseline_class()
    for src in sources:
        ds=CanonicalV4(Path(src['path'])/'training_manifest.json')
        arrays={k:v.detach().cpu().numpy() for k,v in ds.data.items()}
        n=len(ds.keep);require(n>0,'No complete V4 horizons')
        for i,(keep,start) in enumerate(zip(ds.keep,ds.starts)):
            mapping=ds.mapping[int(keep)]
            require(mapping['episode_id']==src['canonical_id'],'Canonical mapping episode identity mismatch')
            index.append(dict(row=count+i,episode_id=src['canonical_id'],canonical_artifact_id=src['artifact_id'],
                sample_id=mapping['sample_id'],canonical_sample_index=int(keep),hand_start_row=int(start),mapping=mapping))
        metadata=dict(src['source']);metadata.update(src['canonical_metadata']);metadata.update(src['entry'])
        metadata.setdefault('producer_type','unknown');metadata.setdefault('task_arm','right')
        metadata.setdefault('task_arm_source','metadata' if 'task_arm' in src['source'] or 'task_arm' in src['entry'] else 'existing canonical legacy default right')
        provenance=read(Path(src['path'])/'provenance.json') if (Path(src['path'])/'provenance.json').exists() else {}
        entries.append(dict(episode_id=src['canonical_id'],canonical_artifact_id=src['artifact_id'],source_manifest=str(Path(src['path'])/'training_manifest.json'),
            sample_start=count,sample_end=count+n,sample_count=n,metadata=metadata,source_hashes=src['hashes'],
            renderer=provenance.get('renderer'),canonical_versions={k:provenance.get(k,src['entry'].get(k)) for k in ('compiler_version','canonical_runtime_version','canonical_runtime_hash')},
            baseline_report=ds.report(),moving_hand_mask=ds.moving.tolist(),closure_initial=ds.initial.tolist(),closure_direction=ds.direction.tolist()))
        chunks.append(arrays);count+=n;verify_files(src['hashes'])
    keys=set(chunks[0]);require(all(set(x)==keys for x in chunks),'Baseline tensor keys differ')
    arrays={k:np.concatenate([x[k] for x in chunks],axis=0) for k in chunks[0]}
    np.savez_compressed(output/'samples.npz',**arrays)
    # Exact serialization check of every tensor; no tolerance or new normalization.
    with np.load(output/'samples.npz',allow_pickle=False) as z:
        require(set(z.files)==keys and all(z[k].dtype==v.dtype and np.array_equal(z[k],v) for k,v in arrays.items()),'V4 serialization equality failed')
    with (output/'index.jsonl').open('x') as f:
        for row in index:f.write(json.dumps(row,sort_keys=True)+'\n')
    split['train_samples']=sum(e['sample_count'] for e in entries if e['episode_id'] in split['train_episodes'])
    split['val_samples']=count-split['train_samples'];write(output/'split.json',split)
    for src in sources:verify_files(src['hashes'])
    verify_files(implementation);verify_files(merged_hashes)
    manifest=dict(schema_version=VERSION,artifact_type='v4_dataset',dataset_id=output.name,builder_version=__version__,model='v4',
        sample_count=count,tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},episodes=entries,
        builder_config=config,builder_source_hashes=implementation,excluded=excluded,batch_summary=batch_report,
        merged_source_hashes=merged_hashes,canonical_root=str(Path(canonical_root).resolve()),normalization_applied=False,baseline_tensor_serialization_exact=True,
        files={n:sha(output/n) for n in ('samples.npz','split.json','index.jsonl')})
    write(output/'manifest.json',manifest);seal(output,['manifest.json','samples.npz','split.json','index.jsonl'])
    validate(output)
    return manifest

def validate(folder):
    import numpy as np
    folder=Path(folder);verify_seal(folder);m=read(folder/'manifest.json')
    require(m.get('schema_version')==VERSION and m.get('artifact_type')=='v4_dataset' and m.get('model')=='v4','Invalid dataset manifest')
    for n,h in m['files'].items():require(sha(folder/n)==h,'Dataset file differs: '+n)
    shape={'rgb':(3,128,128),'body':(68,),'hand':(28,),'body_tokens':(10,64),'hand_ref':(50,14),'phase':(),'progress':()}
    with np.load(folder/'samples.npz',allow_pickle=False) as z:
        require(set(z.files)==set(shape),'Unexpected V4 tensor keys')
        for k,trailing in shape.items():
            a=z[k];require(a.shape==(m['sample_count'],*trailing) and str(a.dtype)==('int64' if k=='phase' else 'float32') and np.isfinite(a).all(),'Invalid tensor: '+k)
    index=[json.loads(x) for x in (folder/'index.jsonl').read_text().splitlines()]
    require(len(index)==m['sample_count'] and [r['row'] for r in index]==list(range(len(index))),'Invalid sample index ordering')
    cursor=0
    for e in m['episodes']:
        require(e['sample_start']==cursor and e['sample_end']==cursor+e['sample_count'],'Invalid episode ranges')
        require(all(r['episode_id']==e['episode_id'] for r in index[cursor:e['sample_end']]),'Episode sample identity mismatch');cursor=e['sample_end']
    require(cursor==m['sample_count'] and cursor>0,'Invalid sample count')
    split=read(folder/'split.json');make_split([e['episode_id'] for e in m['episodes']],dict(mode='explicit',train_episodes=split['train_episodes'],val_episodes=split['val_episodes']))
    return m
