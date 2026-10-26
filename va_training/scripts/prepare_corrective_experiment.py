"""Audit the 160-row prototype and derive immutable K=5 labels with provenance."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from datasets.tokens import load_token_episodes,STATE_WIDTHS
from diagnostics.teacher import CorrectiveTeacher
from diagnostics.corrective import phase_match,quantiles
from sonic.release import digest
from utils.config import dump_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prototype',required=True);p.add_argument('--rollout',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--round',choices=['corrective_round1','corrective_round2'],default='corrective_round1')
    p.add_argument('--phase-scales',default='{"body_q":0.1,"body_dq":1.0,"eef_pos":0.05,"hand_q":0.2}')
    args=p.parse_args(); source=Path(args.rollout); proto=Path(args.prototype); out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    r=json.loads((source/'report.json').read_text()); pr=json.loads((proto/'report.json').read_text())
    assert digest(source/'report.json')==pr['provenance']['rollout_report_sha256']
    import torch
    config=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['config'];k=config['model']['token_chunk_size'];offset=config['data']['target_offset']
    assert k==5 and offset==2
    data=dict(config['data']);data['episode_ids']=[r['episode']];e=load_token_episodes(data)[0]
    z=np.load(proto/'corrective_samples.npz');live=np.load(source/'live_observations.npz');physics=np.load(source/'physics.npz')
    initial=np.loadtxt(source/'policy_logs/base_quat.csv',delimiter=',',skiprows=1)[0,5:]
    teacher=CorrectiveTeacher(config['sonic'],e)
    native_quat=np.loadtxt(Path(e.provenance['token_label_source']).parent/'base_quat.csv',delimiter=',',skiprows=1)[:len(e.tokens),5:]
    native=np.array([teacher.token(i,q,native_quat[0]) for i,q in enumerate(native_quat)])
    assert np.array_equal(native,e.tokens), 'Teacher must reproduce every native token exactly'
    anchors=z['capture_anchor'];verified=[]
    for i,t in enumerate(anchors):
        indexes=np.maximum(np.arange(t-2,t+1),0)
        for key in STATE_WIDTHS:
            if key!='projected_gravity':assert np.array_equal(z[key][i],live[key][indexes]), key
        label=teacher.token(int(t)+2,z['base_quat'][i,-1],initial)
        assert np.array_equal(label,z['target_sonic_token'][i])
        assert np.array_equal(z['target_hand_ref'][i],e.hand_ref[t+2])
        verified.append(True)
    valid=anchors+offset+k<=len(e.tokens); anchors=anchors[valid]
    tokens=np.array([[teacher.token(int(t)+offset+j,z['base_quat'][i,-1],initial) for j in range(k)] for i,t in enumerate(z['capture_anchor']) if valid[i]])
    targets=anchors[:,None]+offset+np.arange(k); hands=e.hand_ref[targets]
    # Every 200-tick block holds out its final 50 ticks, with a 10-tick boundary guard.
    val=(anchors%200>=150); train=(anchors%200<140); guard=~(train|val)
    arrays={name:z[name][valid] for name in ['rgb','history_mask',*STATE_WIDTHS]}
    arrays.update(capture_anchor=anchors,execution_index=targets,target_sonic_token=tokens,target_hand_ref=hands,train_mask=train,val_mask=val,guard_mask=guard)
    np.savez_compressed(out/'corrective_k5.npz',**arrays)
    expert={name:np.load(e.path/(name+'.npy')) for name in ['body_q','body_dq','eef_pos','hand_q']}
    weights=json.loads(args.phase_scales); assert set(weights)>=set(expert) and all(v>0 for v in weights.values())
    matched=phase_match(physics,expert,weights);lag=matched-np.arange(len(matched))
    np.savez_compressed(out/'phase_diagnostic.npz',matched_expert_index=matched,nominal_index=np.arange(len(matched)),lag_ticks=lag)
    phase={name:quantiles(lag[a:b]) for name,a,b in [('all',0,800),('reach',100,400),('closure',400,500),('lift',500,800)]}
    provenance=[]
    for i,t in enumerate(anchors):
        provenance.append({'sample_id':f'{args.round}:{e.id}:{t}','source_rollout':str(source),'source_report_sha256':digest(source/'report.json'),
            'source_episode_id':e.source_episode_id,'learner_tick':int(t),'rgb_source':str(proto/'corrective_samples.npz'),
            'rgb_rule':'head_rgb re-render of actual learner qpos/qvel at t; no nominal image substitution',
            'structured_state_source':str(source/'live_observations.npz'),'history_ticks':list(map(int,np.maximum(np.arange(t-2,t+1),0))),
            'body_label_source':'frozen release G1 encoder; actual measured base quaternion at t, recorded initial heading, delta_heading=0',
            'measured_base_quaternion_wxyz':z['base_quat'][np.flatnonzero(z['capture_anchor']==t)[0],-1].tolist(),
            'initial_base_quaternion_wxyz':initial.tolist(),'delta_heading_rad':0.0,
            'hand_label_source':str(e.path/'hand_ref_q.npy'),'intended_reference_ticks':targets[i].tolist(),
            'reference_folder':str(teacher.reference),'alignment':'post-step learner t -> intended control t+2..t+6; simple nominal hand schedule',
            'split':'train' if train[i] else 'val' if val[i] else 'guard','matched_expert_index':int(matched[t]),'matched_minus_nominal_ticks':int(lag[t])})
    dump_json(out/'samples.json',provenance)
    dump_json(out/'audit.json',{'round':args.round,'original_prototype_samples':len(z['capture_anchor']),'prototype_shapes':{n:list(z[n].shape) for n in z.files},
        'prototype_all_rows_state_and_labels_verified':all(verified),'native_encoder_exact_rows':len(native),'source_rollout':str(source),
        'source_episode_id':e.source_episode_id,'samples':len(anchors),'train':int(train.sum()),'val':int(val.sum()),'guard':int(guard.sum()),
        'excluded_incomplete_future_anchors':z['capture_anchor'][~valid].tolist(),'shapes':{n:list(v.shape) for n,v in arrays.items()},
        'phase_scales':weights,'phase_lag_ticks':phase,'phase_rule':'matched_expert_index - nominal_index; full reference search; diagnostic only',
        'split_limitation':'Time-block heldout with 10-tick guard, single source learner rollout; not independent trajectory generalization',
        'nominal_tokens_not_equivalent_fraction':float((tokens!=e.tokens[targets]).any(-1).mean()),
        'token_vs_nominal_mae':float(abs(tokens-e.tokens[targets]).mean()),'body_label_scope':'Motion-intention post-FSQ token, not full corrective control actions',
        'sonic':r['provenance']['sonic'],'inputs_sha256':{str(p):digest(p) for p in [proto/'corrective_samples.npz',source/'physics.npz',source/'live_observations.npz',out/'corrective_k5.npz']}})
    print('Audited',len(z['capture_anchor']),'rows; K5',len(anchors),'train',train.sum(),'val',val.sum(),'guard',guard.sum(),flush=True)


if __name__=='__main__':main()
