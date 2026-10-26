"""Zero-training diagnostics; all comparisons use authored absolute target ticks."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from v4.dataset import read, dump, sha, gravity, progress_labels
from v4.evaluation import load
from v4.inference import Inference


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def stats(a):
    a=np.asarray(a)
    return dict(n=int(a.size),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()),mean=float(a.mean()))


def differences(delta,mask):
    x=delta[...,mask]
    return dict(mae=stats(np.abs(x).mean(-1)),rms=stats(np.sqrt(np.mean(x*x,-1))),l2=stats(np.linalg.norm(x,axis=-1)))


def consistency(p,mask):
    if not np.all(np.diff(p['start_tick'])==5):raise ValueError('Nonadjacent authored plan starts')
    h=p['hand_ref'];over=h[:-1,5:]-h[1:,:45];internal=np.diff(h,axis=1)
    result={}
    for name,m in [('all14',np.ones(14,bool)),('moving_right',mask)]:
        result[name]=dict(boundary=differences(over[:,0],m),overlap=differences(over,m),internal=differences(internal,m))
    return result,over,internal


def closure(x,initial,direction):
    return np.clip(((x-initial)*direction).sum(-1)/(direction@direction),0,1)


def combine(p,tick,decay,available,exclude=None):
    if decay is not None and decay<=0:raise ValueError('Positive recency decay required')
    indices=np.flatnonzero((p['start_tick']<=tick)&(p['start_tick']+50>tick)&(available<=tick))
    if exclude is not None:indices=indices[indices!=exclude]
    if not len(indices):raise ValueError('No valid hand target')
    if decay is None:return p['hand_ref'][indices[-1],tick-p['start_tick'][indices[-1]]].astype(float)
    ages=(p['start_tick'][indices[-1]]-p['start_tick'][indices])/5
    weights=np.exp(-decay*ages);weights/=weights.sum()
    return np.sum(p['hand_ref'][indices,tick-p['start_tick'][indices]]*weights[:,None],axis=0)


def stream_info(p,mask):
    c,_,_=consistency(p,mask);phase=p['phase_logits'].argmax(-1);idx=np.flatnonzero(np.diff(phase)!=0)+1
    c.update(plans=len(phase),phase_transitions=[dict(plan=int(i),anchor=int(p['anchor'][i]),time_seconds=float(p['anchor'][i]/50),previous=int(phase[i-1]),current=int(phase[i])) for i in idx],
        progress_step=stats(np.abs(np.diff(p['closure_progress']))),body_same_tick_overlap_mae=stats(np.abs(p['body_tokens'][:-1,5:]-p['body_tokens'][1:,:5]).mean(-1)),
        body_slot0_replan_change_mae=stats(np.abs(np.diff(p['body_tokens'][:,0],axis=0)).mean(-1)))
    return c


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/workspace/group2/outputs/va_v4_overfit/kimodo')
    parser.add_argument('--weak-decay',type=float,default=2.);parser.add_argument('--medium-decay',type=float,default=.7);args=parser.parse_args()
    base=Path(args.root);out=base/'diagnostics_plan_consistency';out.mkdir(exist_ok=True)
    checkpoint=base/'checkpoint.pt';before=sha(checkpoint);torch.set_num_threads(1)
    model,saved=load(checkpoint);policy=Inference(model);source=Path(saved['manifest']).parent
    s=arrays(source/'va_dataset/samples.npz');mapping=[json.loads(x) for x in (source/'va_dataset/sample_mapping.jsonl').read_text().splitlines()]
    refs=arrays(source/'canonical_replay/references.npz');names=read(source/'source_reference/metadata.json')['hand_joint_names']
    _,_,moving,initial,direction=progress_labels(refs['hand_ref_q'],names)
    run=base/'normal_gpu';plans=arrays(run/'predictions.npz');physical=arrays(run/'physics.npz')
    timeline=read(run/'control_timeline.json');timings=read(run/'inference_timing.json')
    ticks=np.array([r['actual_tick'] for r in timeline]);applied=np.array([r['hand_target_applied'] for r in timeline]);measured=physical['hand_q']
    available=np.array([r['available_tick'] for r in timings]);frame_index={int(t):i for i,t in enumerate(physical['frame'])}
    c,over,internal=consistency(plans,moving);dump(out/'overlap_metrics.json',dict(units='radians; MAE/RMS/L2 reduced over named joints, then summarized over pairs/slots',moving_joint_names=np.array(names)[moving].tolist(),learner_saved=c))
    b=np.abs(over[:,0,:,]).mean(-1);bm=np.abs(over[:,0,moving]).mean(-1)
    events=[]
    for i in range(len(b)):
        k=i+1;anchor=int(plans['anchor'][k]);q=frame_index.get(anchor)
        events.append(dict(old_plan_id=i,new_plan_id=k,intended_tick=int(plans['start_tick'][k]),time_seconds=float(plans['start_tick'][k]/50),
            boundary_mae_all14=float(b[i]),boundary_mae_moving_right=float(bm[i]),old_target=plans['hand_ref'][i,5].tolist(),new_target=plans['hand_ref'][k,0].tolist(),
            phase=int(plans['phase_logits'][k].argmax()),progress=float(plans['closure_progress'][k]),capture_anchor=anchor,
            body_q=None if q is None else physical['body_q'][q].tolist(),right_wrist_pose=None if q is None else physical['right_wrist_pose'][q].tolist()))
    largest=sorted(events,key=lambda e:e['boundary_mae_moving_right'],reverse=True)[:10]
    dump(out/'boundary_jumps.json',dict(definition='old h[5] versus new h[0], same absolute tick, never consecutive-time samples',events=events,largest=largest))
    ensembles={};exec_targets={}
    for name,decay in [('none',None),('weak',args.weak_decay),('medium',args.medium_decay)]:
        targets=np.array([combine(plans,int(t),decay,available) for t in ticks]);exec_targets[name]=targets
        # Compare ensemble just before/after the arriving plan at the SAME tick.
        revision=np.array([combine(plans,int(t),decay,available)-combine(plans,int(t),decay,available,exclude=k) for k,t in enumerate(plans['start_tick']) if k>0])
        progress=closure(targets,initial,direction);found=np.flatnonzero(progress>.05)
        metrics={label:dict(same_tick_boundary_revision=differences(revision,m),executed_step=differences(np.diff(targets,axis=0),m),
                    target_velocity=differences(np.diff(targets,axis=0)*50,m),target_acceleration=differences(np.diff(targets,n=2,axis=0)*50**2,m),
                    target_jerk=differences(np.diff(targets,n=3,axis=0)*50**3,m)) for label,m in [('all14',np.ones(14,bool)),('moving_right',moving)]}
        ensembles[name]=dict(decay_per_plan=decay,formula='exp(-decay * age_in_5_tick_plan_intervals); newest highest; same tick only',
            metrics=metrics,closure_onset_tick=int(ticks[found[0]]) if len(found) else None,peak_closure_progress=float(progress.max()),final_closure_progress=float(progress[-1]),
            max_abs_target_from_initial=(np.abs(targets-initial).max(0)).tolist())
    if not np.allclose(exec_targets['none'],applied,atol=1e-7,rtol=0):raise ValueError('Saved plans do not reproduce actually applied targets')
    ensemble_output=dict(settings=ensembles,body_unchanged=True,body_source='Existing recorded body tokens; no body averaging or new body execution',
        baseline_applied_target_max_abs_difference=float(np.abs(exec_targets['none']-applied).max()),live_run=False,
        physical_improvement_claim=False,measured_hand={label:dict(step=differences(np.diff(measured,axis=0),m),jerk=differences(np.diff(measured,n=3,axis=0)*50**3,m)) for label,m in [('all14',np.ones(14,bool)),('moving_right',moving)]})
    dump(out/'temporal_ensemble_offline.json',ensemble_output);np.savez_compressed(out/'ensemble_targets.npz',tick=ticks,measured=measured,**exec_targets)
    print('A and A2 saved; reconstructing observation streams',flush=True)
    # Every available canonical decision sample, including nine horizon-tail samples;
    # these need no padded supervision because inference needs only current inputs.
    expert_rgb=[];expert_body=[];expert_hand=[]
    for i,row in enumerate(mapping):
        expert_rgb.append(np.array(Image.open(source/'va_dataset'/row['rgb_path']).convert('RGB')))
        q=s['state_history_root_quat'][i,-1]
        expert_body.append(np.r_[s['state_history_body_q'][i,-1],s['state_history_body_dq'][i,-1],q,s['state_history_root_velocity'][i,-1,3:],gravity(q)])
        expert_hand.append(np.r_[s['state_history_hand_q'][i,-1],s['state_history_hand_dq'][i,-1]])
    def infer_stream(rgb,bodies,hands,anchors,starts):
        predictions=[policy.predict(r,b,h)[0] for r,b,h in zip(rgb,bodies,hands)]
        return dict(**{k:np.array([p[k] for p in predictions]) for k in predictions[0]},anchor=np.array(anchors),start_tick=np.array(starts))
    expert=infer_stream(expert_rgb,expert_body,expert_hand,[r['anchor_execution_index'] for r in mapping],[r['future_execution_indices'][0] for r in mapping])
    np.savez_compressed(out/'expert_predictions.npz',**expert)
    # Match the original worker: fresh MjData + saved qpos/qvel + mj_forward.
    # No mj_step, physics rollout, expert-state injection or new data collection.
    root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
    from canonical_va.common import PROJECT, ROBOT
    sys.path.insert(0,str(PROJECT/'scripts'))
    from scene import load_scene
    contract=read(root/'canonical_va/runtime_contract.json')
    excluded={'dl-group2/scripts/expert_trajectory.py','dl-group2/scripts/run_sonic_grasp.py'}
    mismatches={rel:dict(expected=h,actual=sha(root/rel)) for rel,h in contract['source_hashes'].items() if sha(root/rel)!=h}
    if set(mismatches)-excluded:raise ValueError('Rendering dependency changed: '+str(mismatches))
    dump(out/'render_asset_validation.json',dict(changed_unused_controller_files=mismatches,render_assets_match=True,no_controller_imported=True))
    import mujoco
    from OpenGL import GL
    render_model,_=load_scene(supported=False,robot_path=ROBOT,scene_path=PROJECT/'scenes/tabletop.xml')
    render_model.opt.timestep=.005
    render_data=mujoco.MjData(render_model);renderer=mujoco.Renderer(render_model,height=240,width=320)
    gpu=dict(vendor=GL.glGetString(GL.GL_VENDOR).decode(),renderer=GL.glGetString(GL.GL_RENDERER).decode(),version=GL.glGetString(GL.GL_VERSION).decode())
    dump(out/'renderer.json',gpu)
    if 'Moore Threads' not in gpu['vendor'] or 'MTT S4000' not in gpu['renderer']:raise RuntimeError('MTT S4000 required')
    learner_rgb=[];learner_body=[];learner_hand=[];indices=[]
    for i,anchor in enumerate(plans['anchor']):
        if int(anchor) not in frame_index:continue
        k=frame_index[int(anchor)];render_data.qpos[:]=physical['full_qpos'][k];render_data.qvel[:]=physical['full_qvel'][k]
        mujoco.mj_forward(render_model,render_data);renderer.update_scene(render_data,camera='head_rgb');learner_rgb.append(renderer.render().copy())
        q=physical['root_quat'][k];learner_body.append(np.r_[physical['body_q'][k],physical['body_dq'][k],q,physical['root_velocity'][k,3:],gravity(q)])
        learner_hand.append(np.r_[physical['hand_q'][k],physical['hand_dq'][k]]);indices.append(i)
    learner=infer_stream(learner_rgb,learner_body,learner_hand,plans['anchor'][indices],plans['start_tick'][indices]);renderer.close()
    np.savez_compressed(out/'learner_predictions.npz',**learner)
    np.savez_compressed(out/'learner_reconstructed_observations.npz',rgb=learner_rgb,body=learner_body,hand=learner_hand,anchor=learner['anchor'])
    parity={k:dict(mae=float(np.abs(learner[k]-plans[k][indices]).mean()),max_abs=float(np.abs(learner[k]-plans[k][indices]).max())) for k in ('body_tokens','hand_ref','phase_logits','closure_progress')}
    dump(out/'observation_replay_validation.json',dict(checkpoint_sha256=before,expert_anchors=len(expert['anchor']),learner_anchors=len(indices),
        missing_learner_anchors=[int(a) for a in plans['anchor'] if int(a) not in frame_index],learner_rgb='Reconstructed from recorded capture-anchor qpos/qvel using exact worker camera/resolution and GPU renderer; original RGB was not saved',
        original_startup_observation_unavailable=True,parity_to_saved_predictions=parity))
    print('Inference streams saved; parity='+json.dumps(parity),flush=True)
    stability=dict(expert_all=stream_info(expert,moving),expert_training_horizon_subset=stream_info({k:v[:152] for k,v in expert.items()},moving),
        learner_reinferred=stream_info(learner,moving),learner_saved_all=stream_info(plans,moving),parity=parity,
        caveat='Expert capture anchors 0,5,...; learner captures 3,8,... plus unavailable startup -2. Each stream uses its own authored absolute target ticks; no artificial cross-stream shift.')
    # Fixed-RGB local proprio sensitivity, four deterministic perturbations per early observation.
    # This probes the function, not a simulated physically consistent sensor disturbance.
    rng=np.random.default_rng(4);noise_b=rng.normal(size=(4,68));noise_h=rng.normal(size=(4,28))
    scales_b=np.r_[np.full(29,.001),np.full(29,.01),np.zeros(4),np.full(3,.001),np.zeros(3)]
    scales_h=np.r_[np.full(14,.001),np.full(14,.01)]
    sensitivity={}
    for name,p,rgb,bodies,hands in [('expert',expert,expert_rgb,expert_body,expert_hand),('learner',learner,learner_rgb,learner_body,learner_hand)]:
        records=[]
        for i in np.flatnonzero((p['anchor']>=0)&(p['anchor']<=50)):
            for nb,nh in zip(noise_b,noise_h):
                z,_=policy.predict(rgb[i],bodies[i]+nb*scales_b,hands[i]+nh*scales_h)
                records.append(dict(hand_mae=float(np.abs(z['hand_ref']-p['hand_ref'][i]).mean()),body_mae=float(np.abs(z['body_tokens']-p['body_tokens'][i]).mean()),
                    progress_delta=float(abs(z['closure_progress']-p['closure_progress'][i])),phase_flip=bool(z['phase_logits'].argmax()!=p['phase_logits'][i].argmax())))
        sensitivity[name]={k:stats([r[k] for r in records]) for k in ('hand_mae','body_mae','progress_delta')}
        sensitivity[name]['phase_flips']=sum(r['phase_flip'] for r in records)
    stability['fixed_rgb_proprio_sensitivity']=dict(noise='Four fixed-seed Gaussian directions; sigma joint q=.001 rad, joint dq=.01 rad/s, base angular velocity=.001 rad/s; quat/gravity/RGB unchanged',results=sensitivity)
    dump(out/'expert_vs_learner_stability.json',stability)
    # Early target metrics from explicit canonical future_execution_indices only.
    target_by_tick={int(t):s['body_target'][i,j] for i,r in enumerate(mapping) for j,t in enumerate(r['future_execution_indices'])}
    post=arrays(source/'canonical_replay/post_states.npz');post_index={x:i for i,x in enumerate(post['observation_id'])}
    early={}
    for name,p in [('expert',expert),('learner',plans)]:
        rows=[]
        for i,a in enumerate(p['anchor']):
            if a>50:continue
            valid=[(j,int(p['start_tick'][i]+j)) for j in range(10) if int(p['start_tick'][i]+j) in target_by_tick]
            mae=float(np.mean([np.abs(p['body_tokens'][i,j]-target_by_tick[t]).mean() for j,t in valid])) if valid else None
            rows.append(dict(plan=i,anchor=int(a),time_seconds=float(a/50),start_tick=int(p['start_tick'][i]),body_token_mae=mae,body_slots_compared=len(valid),
                hand_slot0_progress=float(closure(p['hand_ref'][i,0],initial,direction)),phase=int(p['phase_logits'][i].argmax()),auxiliary_progress=float(p['closure_progress'][i]),
                boundary_mae=None if i==0 else float(np.abs(p['hand_ref'][i-1,5]-p['hand_ref'][i,0]).mean())))
        early[name]=rows
    physical_errors=[]
    for row in mapping:
        tick=row['anchor_execution_index']
        if tick>50:continue
        e=post_index[row['observation_id']];l=frame_index[tick]
        dot=abs(float(post['right_wrist_pose'][e,3:]@physical['right_wrist_pose'][l,3:]))
        physical_errors.append(dict(tick=tick,time_seconds=tick/50,body_q_rms=float(np.sqrt(np.mean((physical['body_q'][l]-post['body_q'][e])**2))),
            wrist_position_error=float(np.linalg.norm(physical['right_wrist_pose'][l,:3]-post['right_wrist_pose'][e,:3])),wrist_orientation_error=float(2*np.arccos(np.clip(dot,0,1)))))
    early['physical_errors']=physical_errors;dump(out/'early_rollout.json',early)
    # Plots use absolute intended tick for handoffs, not observation timestamps.
    sys.path.insert(0,str(root/'outputs/va_v2_diagnosis/plot_dependencies'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=plans['start_tick'][1:]/50
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(x,b,label='all14');ax.plot(x,bm,label='moving right');ax.set(xlabel='Intended handoff time (s)',ylabel='Same-tick boundary MAE (rad)');ax.legend();fig.tight_layout();fig.savefig(out/'boundary_jump.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    for ax,mask,label in zip(axes,[np.ones(14,bool),moving],['all14','moving right']):
        im=ax.imshow(np.abs(over[...,mask]).mean(-1).T,origin='lower',aspect='auto',extent=[x[0],x[-1],-.5,44.5]);ax.set(xlabel='New plan start time (s)',ylabel='Overlap slot j',title=label);fig.colorbar(im,ax=ax,label='MAE rad')
    fig.tight_layout();fig.savefig(out/'overlap_heatmap.png');plt.close(fig)
    event=largest[0];k=event['new_plan_id'];t=event['intended_tick'];joints=np.flatnonzero(moving)[:3]
    fig,axes=plt.subplots(len(joints),1,figsize=(12,9),sharex=True)
    for ax,j in zip(axes,joints):
        for i,label in [(k-1,'old plan'),(k,'new plan')]:ax.plot((plans['start_tick'][i]+np.arange(50))/50,plans['hand_ref'][i,:,j],label=label)
        ax.plot(ticks/50,applied[:,j],label='actual applied');ax.plot(physical['frame']/50,measured[:,j],label='measured q');ax.axvline(t/50,color='black',ls=':');ax.set(title=names[j],ylabel='rad',xlim=((t-10)/50,(t+35)/50));ax.legend()
    axes[-1].set_xlabel('Absolute control time (s)');fig.tight_layout();fig.savefig(out/'selected_plan_handoffs.png');plt.close(fig)
    ec,eo,_=consistency(expert,moving);lc,lo,_=consistency(learner,moving)
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    for p,o,label in [(expert,eo,'expert observations'),(learner,lo,'learner observations')]:
        axes[0].plot(p['start_tick'][1:]/50,np.abs(o[:,0,:,]).mean(-1),label=label)
        axes[1].plot(p['start_tick'][1:]/50,np.abs(o).mean((1,2)),label=label)
    axes[0].set_ylabel('Boundary MAE rad');axes[1].set_ylabel('45-slot mean MAE rad');axes[1].set_xlabel('Own-stream intended start time (s)')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'expert_vs_learner_overlap.png');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for p,label in [(expert,'expert observations'),(learner,'learner observations')]:
        axes[0].step(p['anchor']/50,p['phase_logits'].argmax(-1),label=label);axes[1].plot(p['anchor']/50,p['closure_progress'],label=label)
        axes[2].plot(p['start_tick'][1:]/50,np.abs(p['body_tokens'][:-1,5:]-p['body_tokens'][1:,:5]).mean((1,2)),label=label)
    axes[0].set_ylabel('Phase');axes[1].set_ylabel('Auxiliary progress');axes[2].set_ylabel('Body same-tick MAE');axes[2].set_xlabel('Time (s)')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'phase_progress_body.png');plt.close(fig)
    if sha(checkpoint)!=before:raise RuntimeError('Checkpoint changed')
    dump(out/'provenance.json',dict(checkpoint_sha256=before,training=False,model_changed=False,sonic_launched=False,physics_steps=0,
        inputs={str(p):sha(p) for p in [run/'predictions.npz',run/'physics.npz',source/'va_dataset/samples.npz']},script_sha256=sha(__file__)))
    print('Diagnostics complete: '+str(out),flush=True)


if __name__=='__main__':main()
