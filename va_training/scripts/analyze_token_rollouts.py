"""Trajectory distribution shift and native consumption timing from diagnostic rollouts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.config import dump_json


def percentiles(a):
    a=np.asarray(a); return dict(p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),max=float(a.max())) if len(a) else None


def orientation(a,b):
    a=a/np.linalg.norm(a,axis=-1,keepdims=True); b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.degrees(2*np.arccos(np.clip(abs((a*b).sum(-1)),0,1)))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--rollouts',nargs='+',required=True)
    p.add_argument('--output',required=True); args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    reports={}; plot_data={}
    for path in args.rollouts:
        path=Path(path); r=json.loads((path/'report.json').read_text()); episode=Path(r['provenance']['dataset']['token_label_source']).parents[3]
        actual=np.load(path/'physics.npz'); online=np.load(path/'live_observations.npz'); frames=online['reference_frame']; n=len(frames)
        expert={k:np.load(episode/(k+'.npy'))[frames] for k in ['body_q','body_dq','root_quat','eef_pos','hand_q','cube_pos']}
        errors={k:np.sqrt(((actual[k]-expert[k])**2).mean(-1)) for k in ['body_q','body_dq','hand_q']}
        errors['root_orientation_deg']=orientation(actual['root_quat'],expert['root_quat'])
        errors['wrist_position_cm']=np.linalg.norm(actual['eef_pos']-expert['eef_pos'],axis=-1)*100
        errors['cube_position_cm']=np.linalg.norm(actual['cube_pos']-expert['cube_pos'],axis=-1)*100
        fullq=np.load(episode/'body_q.npy'); nearest=[]
        for q in actual['body_q']: nearest.append(int(np.argmin(((fullq-q)**2).mean(-1))))
        nearest=np.array(nearest); errors['nearest_body_q_rms']=np.sqrt(((actual['body_q']-fullq[nearest])**2).mean(-1))
        errors['nearest_phase_tick']=nearest; errors['phase_lag_ticks']=frames-nearest
        pred=np.load(path/'predictions.npz'); anchors=pred['capture_anchor']; valid=(anchors>=0)&(anchors+1<len(fullq))
        gt=np.loadtxt(episode/'validation'/r['provenance']['dataset']['selected_validation_run']/'policy_logs/token_state.csv',delimiter=',',skiprows=1)[:len(fullq),5:]
        if pred['token'].ndim == 3:
            target=pred['intended_execution_index'][valid,None]+np.arange(pred['token'].shape[1])
            in_range=target<len(gt)
            err=abs(pred['token'][valid]-gt[np.minimum(target,len(gt)-1)])*in_range[...,None]
            tokenerr=err.sum((1,2))/(in_range.sum(1)*64)
        else:
            tokenerr=abs(pred['token'][valid]-gt[pred['intended_execution_index'][valid]]).mean(-1)
        timing=json.loads((path/'timing.json').read_text()); live=[x for x in timing if x['anchor']>=0]
        applied=[x for x in live if 'applied_index' in x and 'consumed_monotonic' in x]
        summary={'consumed_token_nominal_timeline_mae':float(abs(online['token']-gt[frames]).mean()),
                 'trajectory':{k:percentiles(a) for k,a in errors.items() if k!='nearest_phase_tick'},
                 'token_intended_timeline_mae':float(tokenerr.mean()) if len(tokenerr) else None,
                 'token_windows':{},'state_windows':{},'task':{k:r.get(k) for k in ['closed_loop_pass','max_lift_m','max_continuous_hold_seconds','fall_steps','maximum_cube_height_m','final_cube_displacement_m']}}
        for lo,hi in [(0,100),(100,400),(400,500),(500,800)]:
            key=f'{lo}:{hi}'; window=(anchors[valid]>=lo)&(anchors[valid]<hi)
            summary['token_windows'][key]=float(tokenerr[window].mean()) if window.any() else None
            summary['state_windows'][key]={k:float(a[(frames>=lo)&(frames<hi)].mean()) for k,a in errors.items() if k not in ['nearest_phase_tick','phase_lag_ticks']}
        summary['timing']={'inference_ms':percentiles([x['inference_seconds']*1000 for x in live]),
           'render_ms':percentiles([x['render_seconds']*1000 for x in live]),
           'whole_prediction_ms':percentiles([x['total_seconds']*1000 for x in live]),
           'observation_to_native_snapshot_ms':percentiles([(x['consumed_monotonic']-x['requested'])*1000 for x in applied]),
           'prediction_to_native_snapshot_ms':percentiles([(x['consumed_monotonic']-x['inference_end'])*1000 for x in applied]),
           'enqueue_to_native_snapshot_ms':percentiles([(x['consumed_monotonic']-x['enqueued'])*1000 for x in applied if 'enqueued' in x]),
           'observation_to_hand_command_ms':percentiles([(x['hand_applied_monotonic']-x['requested'])*1000 for x in applied if 'hand_applied_monotonic' in x]),
           'capture_to_consumption_ticks':{str(k):sum(x['applied_index']-x['anchor']==k for x in applied) for k in sorted(set(x['applied_index']-x['anchor'] for x in applied))},
           'next_tick_miss_rate':float(np.mean([x['applied_index']>x['execution_index'] for x in applied])) if applied else None,
           'unique_predictions_applied':len(applied)+int(bool(timing)),'held_pair_ticks':n-(len(applied)+int(bool(timing))) if applied else None,
           'dropped_predictions':sum('discard_reason' in x for x in live),'body_hand_skew_ticks':r['max_body_hand_skew_ticks'],
           'clock_semantics':'Native time_monotonic_ms is the pre-action state log timestamp, a control snapshot proxy; exact input-copy time is not instrumented inside frozen native code.'}
        if (path/'dds_commands.npz').exists():
            from diagnostics.control import cpp_parameters
            parameters=cpp_parameters(r['provenance']['va_config']['sonic']['repository'])
            raw=np.loadtxt(path/'policy_logs/action.csv',delimiter=',',skiprows=1)[:n+1,5:]
            targets=raw[1:][:,parameters['isaaclab_to_mujoco']]*parameters['g1_action_scale']+parameters['default_angles']
            dds=np.load(path/'dds_commands.npz'); times=dds['received_monotonic']; values=dds['motor_target']
            ticks=json.loads((path/'control_timeline.json').read_text()); matched=[]; arrivals=[]
            for tick,target in zip(ticks,targets):
                native=tick['native_snapshot_monotonic']; indexes=np.flatnonzero((times>=native-.0001)&(times<native+.05))
                exact=indexes[np.max(abs(values[indexes]-target),axis=1)<2e-6]
                if len(exact):
                    arrival=times[exact[0]]; arrivals.append((arrival-native)*1000)
                    matched.append((tick['hand_applied_monotonic']-arrival)*1000)
            summary['timing']['dds_action_target_matches']=len(matched)
            summary['timing']['native_snapshot_to_body_dds_ms']=percentiles(arrivals)
            summary['timing']['hand_minus_body_dds_ms']=percentiles(matched)
        contacts=json.loads((path/'contacts.json').read_text()); contactframes=[]
        for frame,rows in zip(frames,contacts):
            names=[c['body'] for c in rows if c['normal_force_n']>1e-4]
            if any('right_hand_thumb' in s for s in names) and any('right_hand_index' in s or 'right_hand_middle' in s for s in names): contactframes.append(int(frame))
        summary['grasp_contact_frames']=len(contactframes); summary['first_grasp_contact_tick']=contactframes[0] if contactframes else None
        name=path.name if path.name!='A' else 'phase1_A'; reports[name]=summary
        np.savez_compressed(out/(name+'_errors.npz'),reference_frame=frames,token_capture_anchor=anchors[valid],token_mae=tokenerr,**errors)
        plot_data[name]=(frames,errors,anchors[valid],tokenerr)
    dump_json(out/'report.json',reports)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(13,13),sharex=True)
    fields=['body_q','body_dq','root_orientation_deg','wrist_position_cm','hand_q','cube_position_cm','nearest_body_q_rms','token_mae']
    for ax,field in zip(axes.flat,fields):
        for name,(frames,errors,anchors,te) in plot_data.items():
            ax.plot((anchors if field=='token_mae' else frames)/50,te if field=='token_mae' else errors[field],label=name,linewidth=1)
        ax.set_ylabel(field); ax.axvspan(8,10,alpha=.08,color='red'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7); axes[-1,0].set_xlabel('Native control time (s)'); axes[-1,1].set_xlabel('Native control time (s)')
    fig.tight_layout(); fig.savefig(out/'distribution_shift.png',dpi=160); plt.close(fig)


if __name__=='__main__': main()
