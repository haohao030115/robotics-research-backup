"""Frozen decoder control errors, alignment comparisons, observability interventions."""
import argparse, copy, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.tokens import load_token_episodes, TokenDataset, STATE_WIDTHS
from losses.token import TokenMetrics
from trainers.token_training import move
from diagnostics.control import ReleaseDecoder
from utils.checkpoint import load_policy
from utils.runtime import initialize, cleanup
from utils.config import dump_json
from sonic.release import bundle_provenance, digest, snap_token


def summary(error):
    a=np.asarray(error)
    return dict(mae=float(abs(a).mean()),rmse=float(np.sqrt((a*a).mean())),max_abs=float(abs(a).max()),
                per_dimension_mae=abs(a).mean(0).tolist(),per_dimension_rmse=np.sqrt((a*a).mean(0)).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--oracle-episode',required=True); p.add_argument('--device',default='musa',choices=['musa','cpu'])
    args=p.parse_args(); runtime=initialize(args.device,38)
    try:
        policy,saved=load_policy(args.checkpoint,runtime.device); config=saved['config']
        sonic=bundle_provenance(config['sonic']); assert sonic==saved['provenance']['sonic']
        out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
        episodes=load_token_episodes(config['data']); mapping={e.id:e for e in episodes}
        split=json.loads((Path(args.checkpoint).parents[1]/'split.json').read_text())
        selected=[mapping[i] for i in split['val']]; data=TokenDataset(selected,128,3,1,1)
        samples=[data[i] for i in range(len(data))]; loader=DataLoader(samples,batch_size=32)
        predictions=[]; ablations={}
        interventions={'current':[], 'without_dq':['body_dq','hand_dq'],
                       'without_base_attitude':['base_quat','projected_gravity'],
                       'without_previous_action':['previous_action'], 'one_frame':[],
                       'without_hand_state':['hand_q','hand_dq'], 'without_rgb':[]}
        with torch.inference_mode():
            for name,fields in interventions.items():
                metrics=TokenMetrics()
                for batch in loader:
                    image,state,mask,token,hand=move(batch,runtime.device)
                    for field in fields: state[field]=getattr(policy,'state_'+field+'_mean').expand_as(state[field])
                    if name=='one_frame': mask=mask.clone(); mask[:,:-1]=False
                    if name=='without_rgb': image=policy.image_mean[None,:,None,None].expand_as(image)
                    output=policy(image,state,mask); metrics.add(output,token,hand)
                    if name=='current': predictions.extend(zip(output['sonic_token'].cpu().numpy(),output['hand_ref'].cpu().numpy()))
                ablations[name]=metrics.report()
        pred=np.array([r[0] for r in predictions]); hands=np.array([r[1] for r in predictions])
        gt=np.stack([s['target_sonic_token'].numpy() for s in samples]); wrong=(snap_token(pred)!=gt).sum(-1)
        metrics={'token':summary(pred-gt),'whole_token_fsq_accuracy':float((wrong==0).mean()),
                 'wrong_coordinate_counts':{str(k):int((wrong==k).sum()) for k in np.unique(wrong)},
                 'wrong_coordinates_mean':float(wrong.mean()),'wrong_coordinates_p50_p95_p99':np.percentile(wrong,[50,95,99]).tolist()}
        decoder=ReleaseDecoder(config['sonic']); acts=[]; checks=[]; offsets={}; predacts=[]; gtacts=[]
        import mujoco
        model=mujoco.MjModel.from_xml_path(config['sonic']['robot_xml']); d=mujoco.MjData(model)
        motor_ids=np.array([model.joint(n).qposadr[0] for n in decoder.parameters['motor_names']])
        root=model.joint('floating_base_joint').qposadr[0]; right=model.body('right_wrist_yaw_link').id
        left=model.body('left_wrist_yaw_link').id; wrist=[]; orientations=[]; leftw=[]
        for s,z in zip(samples,pred):
            e=mapping[s['episode_id']]; t=s['state_index']; f=t+1
            a=decoder.decode(e,f,e.tokens[f]); b=decoder.decode(e,f,z)
            gtacts.append(a); predacts.append(b); acts.append(b-a)
            raw=decoder.logs(e)['action']; index=int(e.control_metadata[f,0])
            checks.append(a-raw[index+1])
            poses=[]
            for target in [decoder.target(a),decoder.target(b)]:
                d.qpos[root:root+3]=np.load(e.path/'root_pos.npy')[f]
                d.qpos[root+3:root+7]=e.state['base_quat'][f]
                d.qpos[motor_ids]=target
                for j,n in enumerate(e.hand_order): d.qpos[model.joint(n).qposadr[0]]=e.hand_ref[f,j]
                mujoco.mj_forward(model,d)
                poses.append((d.xpos[right].copy(),d.xquat[right].copy(),d.xpos[left].copy()))
            wrist.append(np.linalg.norm(poses[0][0]-poses[1][0])*100)
            dot=np.clip(abs(np.dot(poses[0][1],poses[1][1])),0,1); orientations.append(np.degrees(2*np.arccos(dot)))
            leftw.append(np.linalg.norm(poses[0][2]-poses[1][2])*100)
        checks=np.array(checks)
        metrics['native_decoder_reconstruction']=summary(checks)
        if abs(checks).max()>.002: raise RuntimeError('Decoder reconstruction does not match native logs')
        metrics['raw_action']=summary(acts); target_error=decoder.target(predacts)-decoder.target(gtacts)
        metrics['motor_target_radians']=summary(target_error)
        metrics['motor_joint_order']=decoder.parameters['motor_names']
        metrics['raw_action_joint_order']=selected[0].body_order
        metrics['wrist_fk']={name:dict(mean=float(np.mean(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))
                            for name,a in [('right_position_cm',wrist),('right_orientation_deg',orientations),('left_position_cm',leftw)]}
        # Adjacent visual decisions only, never differences across episode boundaries.
        temporal=[]
        for i in range(1,len(samples)):
            if samples[i]['episode_id']==samples[i-1]['episode_id']: temporal.append(np.asarray(acts[i])-np.asarray(acts[i-1]))
        metrics['adjacent_decision_action_error_change']=summary(temporal)
        for offset in range(6):
            errors=[]; herrors=[]
            for s,z,h in zip(samples,pred,hands):
                e=mapping[s['episode_id']]; f=s['state_index']+offset
                if f<len(e.tokens): errors.append(z-e.tokens[f]); herrors.append(h-e.hand_ref[f])
            offsets[str(offset)]={'samples':len(errors),'token_mae':float(abs(np.array(errors)).mean()),'hand_mae':float(abs(np.array(herrors)).mean())}
        np.savez_compressed(out/'heldout_predictions.npz',token=pred,gt_token=gt,hand=hands,
                            residual=pred-gt,action_gt=gtacts,action_pred=predacts,
                            episode_id=[s['episode_id'] for s in samples],state_index=[s['state_index'] for s in samples])
        # Dense nominal-state predictions use the latest available 10 Hz expert image.
        e=mapping[args.oracle_episode]; dataset=TokenDataset([e],128,3,1,1)
        dataset.index=[(0,int(np.searchsorted(e.anchors,t,side='right')-1),t) for t in range(len(e.tokens)-1)]
        dense=[dataset[i] for i in range(len(dataset))]; dz=[]; dh=[]
        with torch.inference_mode():
            for batch in DataLoader(dense,batch_size=32):
                image,state,mask,_,_=move(batch,runtime.device); r=policy(image,state,mask)
                dz.extend(r['sonic_token'].cpu().numpy()); dh.extend(r['hand_ref'].cpu().numpy())
        np.savez_compressed(out/'nominal_predictions.npz',tokens=np.concatenate([e.tokens[:1],dz]),
                            hands=np.concatenate([e.hand_ref[:1],dh]),gt_tokens=e.tokens,
                            dense_state_hz=50,causal_image_hz=10)
        report={'control_metrics':metrics,'alignment_current_checkpoint':offsets,'observability_interventions':ablations,
                'observability_method':'Frozen-checkpoint mean replacement/masking; NOT retrained feature ablations',
                'alignment_method':'Same offset-1 checkpoint compared to native labels at offsets 0..5; not proof of optimal training offset',
                'fk_method':'Motor targets at fixed expert base and GT hands; kinematic target sensitivity, not dynamic wrist error',
                'provenance':{'sonic':sonic,'checkpoint_sha256':digest(args.checkpoint),'dataset':saved['provenance']}}
        dump_json(out/'report.json',report); print({k:v for k,v in metrics.items() if k in ['whole_token_fsq_accuracy','wrong_coordinates_mean','raw_action','motor_target_radians','wrist_fk','native_decoder_reconstruction']},flush=True)
    finally: cleanup(runtime)


if __name__=='__main__': main()
