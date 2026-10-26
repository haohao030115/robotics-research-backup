"""Index accepted episodes, report duplicates, and merge compatible sample-wise arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .common import dump,read,load_npz,lines,require,digest
from .production import accepted_episode


def load_episode(folder):
    entry,path=accepted_episode(folder);arrays=load_npz(path)
    metadata={}
    if (folder/'source_reference/metadata.json').exists():metadata=read(folder/'source_reference/metadata.json')
    provenance=read(folder/'provenance.json') if (folder/'provenance.json').exists() else {}
    if (folder/'canonical_replay/metadata.json').exists():
        runtime_meta=read(folder/'canonical_replay/metadata.json');metadata={**runtime_meta,**metadata}
        metadata.setdefault('case',runtime_meta.get('runtime',{}).get('case',{}))
    episode_id=entry['episode_id'];n=len(arrays['sample_id'])
    require(n>0 and n==entry.get('sample_count',entry.get('samples',n)),'Manifest sample count mismatch')
    require(all(v.ndim>=1 and len(v)==n and v.dtype.kind!='O' for v in arrays.values()),'Non sample-wise/object arrays')
    for k,v in arrays.items():
        if v.dtype.kind in 'fiu':require(np.isfinite(v).all(),'Nonfinite array '+k)
    if 'mapping_json' not in arrays:
        mapping=lines(path.parent/'sample_mapping.jsonl');require(len(mapping)==n,'Legacy mapping missing samples')
        arrays['mapping_json']=np.asarray([json.dumps(s,sort_keys=True,separators=(',',':')) for s in mapping])
    mapping=[json.loads(s) for s in arrays['mapping_json']]
    require(len(set(arrays['sample_id'].tolist()))==n,'Duplicate sample IDs')
    require(all(s['episode_id']==episode_id and s['sample_id']==str(sid) for s,sid in zip(mapping,arrays['sample_id'])),'Sample/episode ID mismatch')
    if 'rgb' not in arrays:
        images=[]
        for s in mapping:
            image=path.parent/s['rgb_path'];require(image.resolve().is_relative_to(path.parent.resolve()),'Unsafe RGB path')
            images.append(np.asarray(Image.open(image).convert('RGB')))
        arrays['rgb']=np.stack(images)
    require(arrays['body_target'].shape==(n,5,64) and arrays['hand_target'].shape==(n,5,14),'VA target contract mismatch')
    require(arrays['rgb'].shape==(n,240,320,3) and arrays['rgb'].dtype==np.uint8,'RGB contract mismatch')
    case=entry.get('case',metadata.get('case',{}));xy=entry.get('block_xy',case.get('block_xy',metadata.get('cube_initial_pos',[None,None])[:2]))
    require(len(xy)==2 and all(isinstance(v,(float,int)) and np.isfinite(v) for v in xy),'Missing block XY')
    producer=entry.get('producer_type',metadata.get('producer_type'))
    if producer is None:producer={'kimodo_expert_reference':'kimodo','ardy_expert_reference':'ardy','rule_expert_reference':'rule_based'}.get(metadata.get('schema_name'))
    require(producer in ('kimodo','ardy','rule_based'),'Unknown producer metadata')
    arm=entry.get('task_arm',metadata.get('task_arm','right'));require(arm in ('left','right'),'Invalid task_arm')
    info=dict(entry,episode_id=episode_id,source_path=str(folder.resolve()),producer_type=producer,block_xy=xy,task_arm=arm,
        position_bin=entry.get('position_bin'),sample_count=n,
        compiler_version=entry.get('compiler_version',provenance.get('compiler_version','legacy-unversioned')),
        canonical_runtime_version=entry.get('canonical_runtime_version',provenance.get('canonical_runtime_version','legacy-unversioned')),
        canonical_runtime_hash=entry.get('canonical_runtime_hash',provenance.get('canonical_runtime_hash')),
        source_samples_sha256=digest(path))
    return arrays,info


def aggregate(roots,output,duplicates='first'):
    require(not output.exists(),'Aggregation output must be fresh')
    output.mkdir(parents=True);seen={};included=[];chunks=[];rejected=[];dups=[];dtype_warnings=[];spec=None
    for root in roots:
        candidates=sorted(set(p.parent for p in root.rglob('training_manifest.json')))
        for folder in candidates:
            if folder==output or output in folder.parents:continue
            try:
                entry,_=accepted_episode(folder);eid=entry['episode_id']
                if eid in seen:
                    item=dict(episode_id=eid,kept=seen[eid],duplicate=str(folder.resolve()));dups.append(item);print('DUPLICATE',item,flush=True)
                    continue
                arrays,info=load_episode(folder)
                current={k:(v.shape[1:],v.dtype) for k,v in arrays.items()}
                if spec is not None:
                    require(set(current)==set(spec),'Sample keys incompatible')
                    for key,(shape,dtype) in current.items():
                        require(shape==spec[key][0],'Trailing shape incompatible: '+key)
                        # Unicode IDs/JSON have content-dependent storage widths; explicitly widen those two keys.
                        string_width=key in ('sample_id','mapping_json') and dtype.kind==spec[key][1].kind=='U'
                        require(dtype==spec[key][1] or string_width,'Dtype incompatible: '+key)
                        if dtype!=spec[key][1]:dtype_warnings.append(dict(episode_id=eid,key=key,from_dtype=str(dtype),other_dtype=str(spec[key][1]),policy='lossless Unicode width widening'))
                else:spec=current
                info['sample_start']=sum(x['sample_count'] for x in included);info['sample_end']=info['sample_start']+info['sample_count']
                seen[eid]=str(folder.resolve());included.append(info);chunks.append(arrays)
            except Exception as exc:rejected.append(dict(path=str(folder),reason=f'{type(exc).__name__}: {exc}'))
    report=dict(accepted_episodes=len(included),duplicates=dups,rejected=rejected,dtype_warnings=dtype_warnings,duplicate_policy=duplicates)
    dump(output/'aggregation_report.json',report)
    require(not (dups and duplicates=='error'),'Duplicate episode IDs; see aggregation_report.json')
    require(bool(chunks),'No accepted compatible episodes')
    merged={key:np.concatenate([x[key] for x in chunks],axis=0) for key in spec}
    require(len(set(merged['sample_id'].tolist()))==len(merged['sample_id']),'Cross-episode sample IDs collide')
    (output/'va_dataset').mkdir();np.savez_compressed(output/'va_dataset/samples.npz',**merged)
    dump(output/'training_manifest.json',dict(schema_version=2,sample_contract='canonical-va-samples-v2',episodes=included,
        sample_count=len(merged['sample_id']),samples_sha256=digest(output/'va_dataset/samples.npz'),duplicates=dups,
        compatibility='numeric dtypes and trailing shapes exact; Unicode widths widened without truncation'))
    print(f'Accepted: {len(included)}; duplicates: {len(dups)}; rejected: {len(rejected)}; samples: {len(merged["sample_id"])}',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duplicates',choices=['first','error'],default='first');args=p.parse_args();aggregate(args.roots,args.output,args.duplicates)

if __name__=='__main__':main()
