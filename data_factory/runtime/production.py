"""Production gates and compact accepted artifacts; original sampling function is unchanged."""
import json
import math
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read,dump,load_npz,lines,index,require,digest,snapshot_hash,ROOT
from .compiler import plan_samples
from .versioning import canonical_hash,recorded_sources


def metadata_fields(m,count,physical,position_bin_size=None):
    xy=list(m['case']['block_xy']);arm=m.get('task_arm','right')
    require(arm in ('left','right'),'Invalid task_arm')
    position_bin=None
    if position_bin_size is not None:
        require(np.isfinite(position_bin_size) and position_bin_size>0,'position_bin_size must be positive')
        position_bin=f'x{math.floor(xy[0]/position_bin_size)}_y{math.floor(xy[1]/position_bin_size)}'
    return dict(episode_id=m['episode_id'],producer_type=m['producer_type'],case=dict(m['case']),
        block_xy=xy,block_yaw_deg=float(m['case'].get('block_yaw_deg',0)),task_arm=arm,
        task_arm_source='metadata' if not m.get('task_arm_defaulted') else 'legacy default right',
        position_bin=position_bin,position_bin_size_m=position_bin_size,
        sample_count=count,samples=count,physical_success=bool(physical),expert_valid=bool(physical))


def qa_decision(version,renderer,episode_id,source_hash,registry,rate=0.):
    require(0<=rate<=1,'QA sample rate must be in [0,1]')
    key=canonical_hash(dict(code=version['code_fingerprint'],runtime=version['canonical_runtime_hash'],
        renderer={k:renderer[k] for k in ('GL_VENDOR','GL_RENDERER','GL_VERSION')}))
    if not registry.exists() or key not in read(registry).get('qualified',{}):return True,'first batch or code/runtime/renderer change',key
    certificate=read(registry)['qualified'][key]; evidence=__import__('pathlib').Path(certificate['accepted_episode'])/'validation_summary.json'
    if not evidence.is_file() or digest(evidence)!=certificate['summary_sha256'] or read(evidence).get('expert_valid') is not True:
        return True,'missing or changed QA certificate',key
    score=int(canonical_hash([episode_id,source_hash,key])[:16],16)/2**64
    return score<rate,'deterministic sampled QA' if score<rate else 'qualified code; production basic checks',key


def mark_qa(registry,key,episode):
    import fcntl,os
    registry.parent.mkdir(parents=True,exist_ok=True)
    with (registry.parent/(registry.name+'.lock')).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        value=read(registry) if registry.exists() else {'qualified':{}}
        value['qualified'][key]=dict(accepted_episode=str(episode.resolve()),summary_sha256=digest(episode/'validation_summary.json'))
        tmp=registry.with_suffix('.tmp');dump(tmp,value);os.replace(tmp,registry)


def basic_validate(folder,a,m,physical):
    recorded_sources(folder,read(folder/'provenance.json'))
    require(physical.get('replay_complete') is True and physical.get('reference_frame_coverage') is True,'Native execution incomplete')
    require(physical.get('cursor_missing_ticks')==0 and not physical.get('failure'),'Native timing/cursor failed')
    o=load_npz(folder/'observations.npz');r=load_npz(folder/'references.npz');e=load_npz(folder/'sonic_encoder.npz');d=load_npz(folder/'sonic_decoder.npz');p=load_npz(folder/'physics.npz')
    oi=index(o,'observation_id');ri=index(r,'reference_frame_id');di=index(d,'decision_id');ei=index(e,'decision_id');pi=index(p,'sim_step_id')
    require(set(di)==set(ei),'Encoder/decoder identity sets differ')
    for tab in (o,r,e,d,p):
        for k,v in tab.items():
            if v.dtype.kind in 'fiu':require(np.isfinite(v).all(),'Nonfinite '+k)
    require(d['exact_decoder_input'].shape==(len(di),994) and e['exact_encoder_input'].shape==(len(ei),1762),'Native model input widths differ')
    require(np.allclose(p['post_time']-p['pre_time'],.005,atol=1e-10,rtol=0),'Physics timing differs')
    require(np.all(np.diff(p['sim_step_id'])==1),'Missing physics steps')
    for did,j in di.items():
        k=ei[did]
        for key in ('observation_id','control_tick_id','reference_frame_id','token_id'):require(d[key][j]==e[key][k],'Cross-decision '+key)
        require(str(d['observation_id'][j]) in oi,'Unknown physical observation')
        require(np.array_equal(d['exact_decoder_input'][j,:64],e['current_post_fsq_64d_token'][k]),'Current token/decoder mismatch')
    commands=lines(folder/'commands.jsonl');pubs=np.fromfile(folder/'publications.bin',dtype='<u8').reshape(-1,3)
    pubmap={int(x[1]):x for x in pubs};receipts={x['publication_id']:x for x in lines(folder/'native_receipts.jsonl')}
    consumed={}
    for j,cid in enumerate(p['command_id']):
        if p['playing'][j]:consumed.setdefault(str(cid),[]).append(int(p['sim_step_id'][j]))
    known=set()
    for c in commands:
        require(c['command_id'] not in known,'Duplicate command');known.add(c['command_id'])
        steps=consumed.get(c['command_id'],[]);require(steps==c['consuming_sim_step_ids'],'Command consumption mapping differs')
        if c['phase']=='active':require(bool(steps),'Active command not consumed')
        j=di[c['decision_id']];require(np.array_equal(c['motor_values'],d['native_motor_command'][j]),'Mapped command mismatch')
        for sid in steps:
            k=pi[sid];pid=int(p['publication_id'][k]);require(pid in pubmap and pid in receipts,'Missing native publication/receipt')
            require(int(pubmap[pid][0])==receipts[pid]['command_number']==int(d['native_decision_number'][j]),'DDS identity mismatch')
            require(int(pubmap[pid][2])<=receipts[pid]['receipt_time'],'Receipt precedes publication')
            require(np.array_equal(p['applied_motor_values'][k],d['native_motor_command'][j]),'Actual DDS command differs')
    require(set(consumed)<=known,'Unknown active command')
    images=lines(folder/'rgb/frames.jsonl')
    for image in images:
        j=oi[image['observation_id']]
        require(image['snapshot_sha256']==snapshot_hash(o['full_qpos'][j],o['full_qvel'][j],o['current_applied_ctrl'][j],o['simulation_time'][j]),'RGB observation hash mismatch')
        require(image['sim_step_id']==int(o['sim_step_id'][j]) and image['simulation_time']==float(o['simulation_time'][j]),'RGB observation identity mismatch')
        require(digest(folder/image['path'])==image['image_sha256'],'RGB content changed')
    samples=plan_samples(folder);require(bool(samples),'No complete samples')
    ev=physical['canonical_evaluator'];criterion=ev['criterion']
    require(criterion['minimum_lift_m']==.10 and criterion['minimum_continuous_hold_s']==2.,'Canonical physical thresholds changed')
    require(ev.get('task_arm',m['task_arm'])==m['task_arm'],'Evaluator task arm mismatch')
    physical_ok=bool(physical['physical_success'] and physical['cube_max_lift']>=.10 and ev['final_continuous_hold_seconds']>=2. and not physical['fall_detected'] and physical['opposing_contact_valid'])
    return dict(reference_schema_valid=True,native_execution_completed=True,basic_timing_provenance_valid=True,
        physical_success=physical_ok,sample_mapping_valid=True,sample_count=len(samples),
        validation_level='production_basic',renderer_snapshot_identity_valid=True,
        full_reconstruction_performed=False)


def pack_samples(dataset):
    """Add self-contained RGB + explicit ID mapping. Existing numeric keys/shapes are untouched."""
    arrays=load_npz(dataset/'samples.npz');mapping=lines(dataset/'sample_mapping.jsonl')
    arrays['rgb']=np.stack([np.asarray(Image.open(dataset/s['rgb_path']).convert('RGB')) for s in mapping])
    arrays['mapping_json']=np.asarray([json.dumps(s,sort_keys=True,separators=(',',':')) for s in mapping])
    require(arrays['rgb'].shape==(len(mapping),240,320,3),'Packed RGB shape mismatch')
    np.savez_compressed(dataset/'samples.npz',**arrays)
    check=load_npz(dataset/'samples.npz')
    require(set(check)==set(arrays) and all(np.array_equal(check[k],v) for k,v in arrays.items()),'Packed sample readback failed')
    return {'contract':'canonical-va-samples-v2','sample_count':len(mapping),'samples_sha256':digest(dataset/'samples.npz'),
        'arrays':{k:{'dtype':str(v.dtype),'trailing_shape':list(v.shape[1:])} for k,v in arrays.items()}}


def compact_success(out):
    """Only this invocation's accepted staging artifacts; never touches pre-existing outputs."""
    summary=read(out/'validation_summary.json');require(summary['expert_valid'] is True,'Cannot compact failed evidence')
    keep={'va_dataset','training_manifest.json','validation_summary.json','provenance.json'}
    for p in out.iterdir():
        if p.name not in keep:
            require(not p.is_symlink(),'Refuse pruning symlink')
            if p.is_dir():shutil.rmtree(p)
            else:p.unlink()
    for p in (out/'va_dataset').iterdir():
        if p.name!='samples.npz':
            require(not p.is_symlink(),'Refuse pruning symlink')
            if p.is_dir():shutil.rmtree(p)
            else:p.unlink()


def training_success(out):
    """Publish a compact-but-V4-complete production artifact atomically.

    The source reference, canonical time-indexed tables, sample mapping and
    referenced RGB frames are retained.  Forensic traces remain omitted.
    """
    summary=read(out/'validation_summary.json');require(summary['expert_valid'] is True,'Cannot retain failed evidence')
    required=(
        'source_reference/metadata.json','source_reference/provenance.json',
        'source_reference/reference.npz','source_reference/reference_validation.json',
        'canonical_replay/metadata.json','canonical_replay/provenance.json',
        'canonical_replay/sonic_encoder.npz','canonical_replay/references.npz',
        'canonical_replay/observations.npz','va_dataset/metadata.json',
        'va_dataset/sample_mapping.jsonl','va_dataset/samples.npz',
        'training_manifest.json','validation_summary.json','provenance.json',
        'validation/reference_validation.json','validation/replay_validation.json',
        'validation/temporal_validation.json','validation/dataset_validation.json',
        'validation/renderer.json','REPORT.md')
    mapping=lines(out/'va_dataset/sample_mapping.jsonl')
    rels=list(required)
    for row in mapping:
        rel=str(row['rgb_path'])
        require(rel and not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Unsafe RGB path')
        rels.append('va_dataset/'+rel)
    stage=out/'.training_retention_staging'
    if stage.exists():shutil.rmtree(stage)
    try:
        for rel in dict.fromkeys(rels):
            src=out/rel;require(src.is_file() and not src.is_symlink(),'Missing training retention file: '+rel)
            dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        # Check the canonical tables and packed sample before pruning.
        for rel in ('canonical_replay/sonic_encoder.npz','canonical_replay/references.npz','canonical_replay/observations.npz','va_dataset/samples.npz'):
            require(load_npz(stage/rel),'Unreadable training retention array: '+rel)
        require(lines(stage/'va_dataset/sample_mapping.jsonl')==mapping,'Training mapping changed during staging')
        for child in list(out.iterdir()):
            if child.name==stage.name:continue
            if child.is_symlink():raise ValueError('Refuse pruning symlink')
            if child.is_dir():shutil.rmtree(child)
            else:child.unlink()
        for child in list(stage.iterdir()):
            target=out/child.name
            if target.exists():
                if target.is_dir():shutil.rmtree(target)
                else:target.unlink()
            child.rename(target)
        stage.rmdir()
    except Exception:
        if stage.exists():shutil.rmtree(stage)
        raise
def accepted_episode(folder):
    """Safe resume/index gate: an expert flag alone is never an integrity certificate."""
    manifest=read(folder/'training_manifest.json');require(len(manifest['episodes'])==1,'Expected one episode per accepted directory')
    e=manifest['episodes'][0];require(e.get('expert_valid') is True,'Not accepted')
    data=folder/e.get('path','va_dataset')/'samples.npz'
    require(data.resolve().is_relative_to(folder.resolve()),'Unsafe dataset path')
    require(data.is_file(),'Accepted samples missing')
    if (folder/'validation_summary.json').exists():
        summary=read(folder/'validation_summary.json');require(summary.get('expert_valid') is True,'Summary disagrees with manifest')
        require(digest(data)==summary['sample_schema']['samples_sha256'],'Accepted samples checksum mismatch')
    else:
        validity=read(folder/'validation/dataset_validation.json');require(validity.get('expert_valid') is True,'Legacy validity missing/failed')
    return e,data
