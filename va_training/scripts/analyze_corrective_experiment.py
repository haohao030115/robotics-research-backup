"""Paired physical shift curves, phase diagnostics and prespecified effect sizes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from diagnostics.phase import phase_match,quantiles
from scripts.analyze_token_rollouts import orientation
from sonic.release import digest
from utils.config import dump_json

MODELS=['original','nominal_ft','body25','body50','hand25','hand50','combined25','combined50','combined25_scratch']


def first_cross(v,threshold):
    ix=np.flatnonzero(v>threshold)
    return float(ix[0]/50) if len(ix) else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    out=root/'analysis';out.mkdir(parents=True,exist_ok=True);audit=json.loads((root/'round1/audit.json').read_text());scales=audit['phase_scales']
    corrective=np.load(root/'round1/corrective_k5.npz');source_physics=np.load(Path(audit['source_rollout'])/'physics.npz')
    corrective_train={k:source_physics[k][corrective['capture_anchor'][corrective['train_mask']]] for k in scales}
    reports={};plots={};bootstrap={};constants={}
    for folder in sorted((root/'rollouts').iterdir()):
        if not (folder/'report.json').exists():continue
        r=json.loads((folder/'report.json').read_text());model=next(m for m in sorted(MODELS,key=len,reverse=True) if folder.name.startswith(m+'_'));ep=r['episode'];name=f'{model}:{ep}'
        epath=Path(r['provenance']['dataset']['token_label_source']).parents[3];actual=np.load(folder/'physics.npz')
        expert={k:np.load(epath/(k+'.npy')) for k in ['body_q','body_dq','hand_q','root_quat','eef_pos','eef_quat','cube_pos']}
        assert len(actual['body_q'])==len(expert['body_q'])==800
        errors={k:np.sqrt(((actual[k]-expert[k])**2).mean(-1)) for k in ['body_q','body_dq','hand_q']}
        errors['base_orientation_deg']=orientation(actual['root_quat'],expert['root_quat'])
        errors['wrist_position_cm']=np.linalg.norm(actual['eef_pos']-expert['eef_pos'],axis=-1)*100
        errors['wrist_orientation_deg']=orientation(actual['eef_quat'],expert['eef_quat'])
        errors['cube_position_cm']=np.linalg.norm(actual['cube_pos']-expert['cube_pos'],axis=-1)*100
        errors['body_state_scaled']=np.sqrt((errors['body_q']/.1)**2+(errors['body_dq']/1.)**2)
        matched=phase_match(actual,expert,scales);lag=matched-np.arange(800)
        distances=np.zeros((800,len(corrective_train['body_q'])))
        for field,scale in scales.items():
            distances+=((actual[field][:,None,:]-corrective_train[field][None,:,:])**2).mean(-1)/(scale**2)
        nearest_corrective=np.sqrt(distances.min(-1));errors['nearest_corrective_train_scaled']=nearest_corrective
        contact=json.loads((folder/'contacts.json').read_text());ncontact=0
        for rows in contact:
            names=[row['body'] for row in rows if row['normal_force_n']>1e-4]
            ncontact+=int(any('right_hand_thumb' in v for v in names) and any('right_hand_index' in v or 'right_hand_middle' in v for v in names))
        hashes={p.name:digest(p) for p in (folder/'bootstrap/initial_pose').iterdir()}
        constant={k:r['provenance'][k] for k in ['sonic','runtime']};constant['criterion']=r['criterion']
        if ep in bootstrap:assert hashes==bootstrap[ep] and constant==constants[ep], 'Unequal physical/runtime controls'
        bootstrap[ep]=hashes;constants[ep]=constant
        phase={label:quantiles(lag[a:b]) for label,a,b in [('all',0,800),('reach',100,400),('closure',400,500),('lift',500,800)]}
        hand_ref=np.load(epath/'hand_ref_q.npy')
        applied_hand=np.load(folder/'live_observations.npz')['hand_ref']
        errors['hand_reference_mae_rad']=abs(applied_hand-hand_ref).mean(-1)
        result={'model':model,'episode':ep,'task':{k:r[k] for k in ['closed_loop_pass','max_lift_m','max_continuous_hold_seconds','maximum_cube_height_m','fall_steps']},
            'contact_frames':ncontact,'wrist_over_5cm_seconds':first_cross(errors['wrist_position_cm'],5),
            'body_q_over_0_1rad_seconds':first_cross(errors['body_q'],.1),
            'peak_wrist_cm':float(errors['wrist_position_cm'].max()),'mean_wrist_before_closure_cm':float(errors['wrist_position_cm'][:400].mean()),
            'integrated_body_state_error':float(errors['body_state_scaled'].sum()/50),
            'integrated_body_q_rad_seconds':float(errors['body_q'].sum()/50),
            'hand_q_rms_closure_rad':float(errors['hand_q'][400:500].mean()),
            'hand_reference_mae_closure_rad':float(errors['hand_reference_mae_rad'][400:500].mean()),
            'mean_base_orientation_deg':float(errors['base_orientation_deg'].mean()),'mean_wrist_orientation_deg':float(errors['wrist_orientation_deg'].mean()),
            'phase_lag_ticks':phase,'phase_hand_target_disagreement_closure_rad':float(abs(hand_ref[matched[400:500]]-hand_ref[400:500]).mean()),
            'corrective_state_coverage':{'nearest_scaled_distance':quantiles(nearest_corrective),
                'within_unit_distance_fraction':float((nearest_corrective<=1).mean()),
                'closure_within_unit_distance_fraction':float((nearest_corrective[400:500]<=1).mean()),
                'meaning':'Distance to nearest of 112 corrective training snapshots, same physical scales as phase diagnostic; not a learned manifold or visual coverage metric'},
            'buffer_stale_control_ticks':r['buffer_stale_control_ticks'],'execution_deadline_misses':r['execution_deadline_misses'],
            'prediction_count':r.get('prediction_count'),'bootstrap_hashes':hashes}
        np.savez_compressed(out/(folder.name+'_errors.npz'),**errors,phase_lag_ticks=lag,matched_expert_index=matched)
        reports[name]=result;plots.setdefault(ep,{})[model]=errors
    comparisons={}
    # "Measurable improvement" is fixed before looking at results; two-episode evidence, not statistical proof.
    for model in MODELS[2:]:
        if not all(f'{model}:{ep}' in reports for ep in plots):continue
        results=[]
        for ep in plots:
            current=reports[f'{model}:{ep}'];baseline=reports[f'nominal_ft:{ep}']
            wr=current['mean_wrist_before_closure_cm']/max(baseline['mean_wrist_before_closure_cm'],1e-9)
            br=current['integrated_body_state_error']/max(baseline['integrated_body_state_error'],1e-9)
            hr=current['hand_q_rms_closure_rad']/max(baseline['hand_q_rms_closure_rad'],1e-9)
            results.append({'episode':ep,'wrist_ratio':wr,'body_integral_ratio':br,'closure_hand_ratio':hr,'task_pass':current['task']['closed_loop_pass']})
        comparisons[model]={'per_episode':results,'mean_wrist_ratio':float(np.mean([v['wrist_ratio'] for v in results])),
            'mean_body_integral_ratio':float(np.mean([v['body_integral_ratio'] for v in results])),
            'mean_closure_hand_ratio':float(np.mean([v['closure_hand_ratio'] for v in results]))}
    available=[m for m in MODELS if (root/f'models/{m}/report.json').exists()]
    offline={m:json.loads((root/f'models/{m}/report.json').read_text())['metrics'] for m in available}
    eligible=[];h1_evidence=[]
    for model,c in comparisons.items():
        prior=offline['nominal_ft'];now=offline[model]
        nominal_ok=now['nominal_validation']['token_mae']<=prior['nominal_validation']['token_mae']*1.2 and now['nominal_validation']['hand_mae']<=prior['nominal_validation']['hand_mae']*1.2
        improved_head=(now['corrective_validation']['token_mae']<prior['corrective_validation']['token_mae']*.9 if model.startswith('body') else
                       now['corrective_validation']['hand_mae']<prior['corrective_validation']['hand_mae']*.9 if model.startswith('hand') else
                       now['corrective_validation']['token_mae']<prior['corrective_validation']['token_mae']*.9 and now['corrective_validation']['hand_mae']<prior['corrective_validation']['hand_mae']*.9)
        drift_help=((c['mean_wrist_ratio']<=.9 and c['mean_body_integral_ratio']<=1.1) or
                    (c['mean_closure_hand_ratio']<=.9 and c['mean_wrist_ratio']<=1.1 and c['mean_body_integral_ratio']<=1.1))
        c.update(nominal_not_materially_degraded=bool(nominal_ok),corrective_offline_improved=bool(improved_head),drift_improved=bool(drift_help))
        if not model.endswith('_scratch') and improved_head and (drift_help or any(v['task_pass'] for v in c['per_episode'])):h1_evidence.append(model)
        if nominal_ok and improved_head and (drift_help or any(v['task_pass'] for v in c['per_episode'])):eligible.append(model)
    best=min(eligible,key=lambda m:comparisons[m]['mean_wrist_ratio']+comparisons[m]['mean_body_integral_ratio']+comparisons[m]['mean_closure_hand_ratio']) if eligible else None
    dump_json(out/'report.json',{'rollouts':reports,'comparisons':comparisons,'offline':offline,'round2_justified':bool(best),'round2_model':best,
        'h1_supporting_models':h1_evidence,'hypothesis_scope':'H1 evidence requires corrective offline reduction and actual drift/task improvement; offline-only gain does not establish H1. Round2 additionally preserves nominal metrics.',
        'thresholds':{'body_q_rms_rad':.1,'wrist_cm':5,'nominal_relative_degradation_max':.2,'corrective_error_reduction_min':.1,'drift_reduction_min':.1},
        'phase_scales':scales,'physical_constants':constants,'limitation':'One deterministic rollout per model/episode, fixed seed; effect sizes are not significance estimates'})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=['body_q','body_dq','base_orientation_deg','wrist_position_cm','wrist_orientation_deg','hand_q','body_state_scaled','cube_position_cm']
    for ep,curves in plots.items():
        fig,axes=plt.subplots(4,2,figsize=(14,13),sharex=True)
        for ax,field in zip(axes.flat,fields):
            for model in MODELS:
                if model not in curves:continue
                ax.plot(np.arange(800)/50,curves[model][field],label=model,linewidth=1,alpha=.85)
            ax.set_ylabel(field);ax.axvspan(8,10,color='orange',alpha=.1);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8,ncol=2)
        fig.suptitle(ep+'; closure 8–10 s');fig.tight_layout();fig.savefig(out/(ep+'_shift.png'),dpi=160);fig.savefig(out/(ep+'_shift.svg'));plt.close(fig)
    print('Analysis complete. Bounded round2 justified:',best,flush=True)


if __name__=='__main__':main()
