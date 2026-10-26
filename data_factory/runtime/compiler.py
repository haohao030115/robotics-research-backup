"""VA horizon belongs here, never in the replay loop. All joins use explicit IDs."""
import shutil
import numpy as np
from .common import load_npz, read, lines, index, require, dump, jsonl

STATE_FIELDS=('body_q','body_dq','hand_q','hand_dq','root_pos','root_quat','root_velocity')


def plan_samples(folder):
    o=load_npz(folder/'observations.npz'); r=load_npz(folder/'references.npz'); e=load_npz(folder/'sonic_encoder.npz')
    oi=index(o,'observation_id'); ri=index(r,'reference_frame_id'); ei=index(e,'token_id')
    by_tick={}
    for j,tid in enumerate(r['control_tick_id']):
        require(str(tid) not in by_tick,'Duplicate reference control tick')
        by_tick[str(tid)]=j
    token_by_ref={}
    for j,rid in enumerate(e['reference_frame_id']):
        require(str(rid) not in token_by_ref,'Duplicate token reference')
        token_by_ref[str(rid)]=j
    seq={int(n):j for j,n in enumerate(r['reference_index'])}
    require(len(seq)==len(r['reference_index']),'Duplicate execution/reference index')
    images=lines(folder/'rgb/frames.jsonl'); mappings=[]
    episode=read(folder/'metadata.json')['episode_id']
    def same_segment(anchor,lo,hi):
        return all(k in seq and str(r['phase'][seq[k]])=='active' for k in range(lo,hi+1))
    for image in images:
        oid=image['observation_id']; require(oid in oi,'RGB has unknown observation')
        j=oi[oid]; tid=str(o['control_tick_id'][j]); require(tid in by_tick,'Observation has no reference')
        k=by_tick[tid]; t=int(r['reference_index'][k])
        if str(o['phase'][j])!='active' or not same_segment(t,t,t+6): continue
        start=t
        while start>0 and start-1 in seq and str(r['phase'][seq[start-1]])=='active': start-=1
        # Repeat first active observation only; never borrow standby/reset state.
        hs=[max(start,t-2),max(start,t-1),t]; future=list(range(t+2,t+7))
        history_ids=[str(r['observation_id'][seq[x]]) for x in hs]
        future_refs=[str(r['reference_frame_id'][seq[x]]) for x in future]
        require(all(x in oi for x in history_ids),'History references missing observation')
        require(all(x in token_by_ref for x in future_refs),'Future token missing')
        token_rows=[token_by_ref[x] for x in future_refs]
        for x,z in zip(future,token_rows):
            y=seq[x]
            require(str(e['decision_id'][z])==str(r['decision_id'][y]) and str(e['control_tick_id'][z])==str(r['control_tick_id'][y]),'Body/hand future decision mismatch')
        mappings.append(dict(sample_id=f'{episode}:sample:{len(mappings)+1:06d}',episode_id=episode,observation_id=oid,
            rgb_frame_id=image['rgb_frame_id'],rgb_path=image['path'],state_history_observation_ids=history_ids,
            history_valid_mask=[t-2>=start,t-1>=start,True],future_reference_frame_ids=future_refs,
            future_execution_indices=future,body_token_ids=[str(e['token_id'][z]) for z in token_rows],
            hand_ref_ids=[str(r['hand_ref_id'][seq[x]]) for x in future],anchor_execution_index=t))
    return mappings


def compile_samples(folder,out,validity):
    require(validity.get('expert_valid') is True,'Refuse training export: expert_valid != true')
    mappings=plan_samples(folder); require(bool(mappings),'No complete active samples')
    o=load_npz(folder/'observations.npz'); r=load_npz(folder/'references.npz'); e=load_npz(folder/'sonic_encoder.npz')
    oi=index(o,'observation_id'); hi=index(r,'hand_ref_id'); zi=index(e,'token_id')
    out.mkdir(); (out/'rgb').mkdir()
    arrays={'sample_id':np.asarray([s['sample_id'] for s in mappings]),
            'history_valid_mask':np.asarray([s['history_valid_mask'] for s in mappings],bool),
            'body_target':np.asarray([[e['current_post_fsq_64d_token'][zi[z]] for z in s['body_token_ids']] for s in mappings]),
            'hand_target':np.asarray([[r['hand_ref_q'][hi[h]] for h in s['hand_ref_ids']] for s in mappings])}
    for field in STATE_FIELDS:
        arrays['state_history_'+field]=np.asarray([[o[field][oi[x]] for x in s['state_history_observation_ids']] for s in mappings])
    np.savez_compressed(out/'samples.npz',**arrays)
    for s in mappings:
        shutil.copy2(folder/s['rgb_path'],out/s['rgb_path'])
    jsonl(out/'sample_mapping.jsonl',mappings)
    dump(out/'metadata.json',dict(schema_version=1,sample_count=len(mappings),future_offsets=[2,3,4,5,6],
        history='three observations, repeat first active observation with validity mask',state_fields=STATE_FIELDS,
        target='post-FSQ release encoder 64D token and absolute hand reference radians',rgb_hz=10,control_hz=50,
        body_target_shape=[len(mappings),5,64],hand_target_shape=[len(mappings),5,14],expert_valid=True))
    return mappings
