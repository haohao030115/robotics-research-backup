"""Forensics for real asynchronous native events; no four-step or zero-latency assumptions."""
import threading
from types import SimpleNamespace
import numpy as np
from PIL import Image
from .common import ROOT,PROJECT,SONIC,BUNDLE,load_npz,read,lines,index,require,digest,snapshot_hash
from .native_scene import SnapshotReader,mujoco
from .policy import session,infer,parameters
from .compiler import plan_samples




def expected_native_reference(arrays,key,frames,playing):
    # Original GatherMotionJointVelocitiesMultiFrame explicitly zeroes standby velocities.
    expected=arrays[key][frames].reshape(-1).astype(np.float32)
    return np.zeros_like(expected) if key=='body_ref_dq' and not playing else expected


def verify_receipt_edge(command_id, command_number, publication_id, publications, receipts, consumption_end_ns):
    require(publication_id in publications and publication_id in receipts, 'Consumed command has no publish/receipt')
    pub=publications[publication_id]; receipt=receipts[publication_id]
    require(pub['command_id']==command_id and receipt['command_number']==command_number, 'Publication/receipt belongs to other action')
    require(pub['monotonic_ns']<=receipt['receipt_time']<=consumption_end_ns, 'Publish/receipt/consume causality mismatch')


def verify(folder,a,m):
    o=load_npz(folder/'observations.npz');r=load_npz(folder/'references.npz');e=load_npz(folder/'sonic_encoder.npz');d=load_npz(folder/'sonic_decoder.npz')
    p=load_npz(folder/'physics.npz');post=load_npz(folder/'post_states.npz');raw=load_npz(folder/'native_physics.npz')
    commands=lines(folder/'commands.jsonl');events=lines(folder/'timeline.jsonl');images=lines(folder/'rgb/frames.jsonl')
    receipts=lines(folder/'native_receipts.jsonl');oi=index(o,'observation_id');ei=index(e,'decision_id');di=index(d,'decision_id');ri=index(r,'reference_frame_id');pi=index(p,'sim_step_id');si=index(post,'post_state_id')
    report={}; failures=[];reader=SnapshotReader(render=True)
    def check(key,fn):
        try:
            report[key]={'valid':True};report[key].update(fn() or {})
            if not report[key]['valid']:failures.append(key)
        except Exception as exc:
            report[key]=dict(valid=False,error=f'{type(exc).__name__}: {exc}');failures.append(key)
    try:
        def schema():
            require(len(commands)==len(oi)==len(ei)==len(di)==len(ri),'Missing/duplicate decision data')
            require(set(ei)==set(di),'Encoder/decoder decision sets differ')
            for tab in (o,r,e,d,p,post):
                for k,v in tab.items():
                    require(len(v)==len(next(iter(tab.values()))),'Table length mismatch')
                    if v.dtype.kind in 'fiu': require(np.isfinite(v).all(),'Nonfinite '+k)
            require(e['exact_encoder_input'].shape==(len(ei),1762) and d['exact_decoder_input'].shape==(len(di),994),'Model input schema')
            require(d['current_raw_29d_action'].shape==(len(di),29),'Action schema')
            require(len({x['event_id'] for x in events})==len(events),'Duplicate event IDs')
            provenance=read(folder/'provenance.json')
            from .versioning import recorded_sources
            recorded_sources(folder,provenance)
            runtime=provenance['runtime'];require(digest(runtime['binary'])==runtime['binary_sha256'],'Changed native binary')
            return dict(total_decisions=len(di),active_decisions=int(sum(r['phase']=='active')),physics_steps=len(pi))
        check('schema',schema)
        def vision():
            expected={str(r['observation_id'][j]) for j in range(len(r['phase'])) if r['phase'][j]=='active' and int(r['source_reference_index'][j])%5==0}
            require({x['observation_id'] for x in images}==expected and len(images)==len(expected),'RGB coverage/identity mismatch')
            errors=[]
            for image in images:
                j=oi[image['observation_id']]
                require(int(o['sim_step_id'][j])==image['sim_step_id'] and float(o['simulation_time'][j])==image['simulation_time'],'RGB physical timestamp mismatch')
                require(snapshot_hash(o['full_qpos'][j],o['full_qvel'][j],o['current_applied_ctrl'][j],o['simulation_time'][j])==image['snapshot_sha256']==str(o['snapshot_sha256'][j]),'RGB snapshot hash mismatch')
                require(digest(folder/image['path'])==image['image_sha256'],'RGB changed')
                reader.restore(o['integration_state'][j]);image2=reader.render();saved=np.asarray(Image.open(folder/image['path']).convert('RGB'))
                errors.append(int(np.max(abs(image2.astype(int)-saved.astype(int)))))
            matched=sum(x==0 for x in errors)
            return dict(valid=matched==len(errors) and bool(errors),matched=matched,total=len(errors),max_pixel_error=max(errors,default=None))
        # check wrapper expects details without valid; retain failed-match statistics explicitly below.
        check('vision_alignment',vision)
        def reconstruction(tab,inp,out,filename):
            model=session(filename);errors=[]
            for x,y in zip(tab[inp],tab[out]): errors.append(float(np.max(abs(infer(model,x)-y))))
            tolerance=1e-5;matched=sum(x<=tolerance for x in errors)
            result=dict(valid=matched==len(errors) and bool(errors),matched=matched,total=len(errors),max_absolute_error=max(errors,default=None),
                atol=tolerance,backend='independent ONNX Runtime CPU vs native Torch/MUSA FP32',decision_shift=0)
            return result
        for key,tab,inp,target,name in [('encoder_reconstruction',e,'exact_encoder_input','current_post_fsq_64d_token','model_encoder.onnx'),
                                      ('decoder_reconstruction',d,'exact_decoder_input','current_raw_29d_action','model_decoder.onnx')]:
            check(key,lambda:reconstruction(tab,inp,target,name))
        def decisions():
            by_native={int(v):j for j,v in enumerate(d['native_decision_number'])};par=parameters()
            motor_names=par['motor_names']; motor_qa=[reader.model.joint(n).qposadr[0] for n in motor_names];motor_va=[reader.model.joint(n).dofadr[0] for n in motor_names]
            import yaml,re
            config=yaml.safe_load((BUNDLE/'observation_config.yaml').read_text())
            source=(SONIC/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp').read_text()
            dims={k:int(v) for k,v in re.findall(r'\{"([^"]+)",\s*(\d+),',source)};offset={};n=0
            for item in config['encoder']['encoder_observations']:
                if item['enabled']:offset[item['name']]=(n,dims[item['name']]);n+=dims[item['name']]
            maximum_input_error=0.
            for decision,u in di.items():
                v=ei[decision];j=oi[str(d['observation_id'][u])];rr=ri[str(d['reference_frame_id'][u])]
                for key in ('observation_id','decision_id','control_tick_id','reference_frame_id'):
                    require(str(d[key][u])==str(e[key][v])==str(r[key][rr])==str(o[key][j]),'Cross-decision ID '+key)
                require(np.array_equal(d['exact_decoder_input'][u,:64],e['current_post_fsq_64d_token'][v]) and np.array_equal(d['current_64d_token'][u],e['current_post_fsq_64d_token'][v]),'Token/current input mismatch')
                require(str(d['token_id'][u])==str(e['token_id'][v]),'Token ID mismatch')
                require(np.array_equal(d['previous_action'][u],d['exact_decoder_input'][u,935:964]),'Previous action not in current decoder input')
                num=int(d['native_decision_number'][u])
                if num-1 in by_native: require(np.array_equal(d['previous_action'][u],d['current_raw_29d_action'][by_native[num-1]]),'Previous action causal identity mismatch')
                # Current measured motor state must be the LowState snapshot actually consumed by native code.
                expected_q=(o['full_qpos'][j,motor_qa].astype(np.float32).astype(float)-par['default_angles'])[par['mujoco_to_isaaclab']].astype(np.float32)
                expected_dq=o['full_qvel'][j,motor_va].astype(np.float32)[par['mujoco_to_isaaclab']]
                require(np.allclose(expected_q,d['exact_decoder_input'][u,355:384],atol=1e-6,rtol=0),'Native input not bound to exact q snapshot')
                require(np.array_equal(expected_dq,d['exact_decoder_input'][u,645:674]),'Native input not bound to exact dq snapshot')
                frame=int(r['source_reference_index'][rr]);active=r['phase'][rr]!='standby'
                frames=np.minimum(frame+np.arange(10)*5,len(a['time_s'])-1) if active else np.full(10,frame)
                for field,key in [('motion_joint_positions_10frame_step5','body_ref_q'),('motion_joint_velocities_10frame_step5','body_ref_dq')]:
                    start,width=offset[field];expected=expected_native_reference(a,key,frames,active);actual=e['exact_encoder_input'][v,start:start+width]
                    err=float(np.max(abs(expected-actual)));maximum_input_error=max(maximum_input_error,err)
                    require(err<=1e-6,'Native reference consumption mismatch '+key)
                for key in ('body_ref_q','body_ref_dq','root_ref_pos','root_ref_quat','hand_ref_q'): require(np.array_equal(r[key][rr],a[key][frame]),'Reference mutated '+key)
            return dict(matched=len(di),total=len(di),shift=0,max_reference_input_error=maximum_input_error)
        check('no_off_by_one',decisions)
        def trace():
            receipt_by_pub={x['publication_id']:x for x in receipts};pubs={x['publication_id']:x for x in events if x['event']=='COMMAND_PUBLISH'}
            used=[];counts=[];cids={x['command_id'] for x in commands}
            for c in commands:
                u=di[c['decision_id']];require(c['command_id']==str(d['command_id'][u]),'Command/action ID mismatch')
                require(np.array_equal(c['motor_values'],d['native_motor_command'][u]),'Mapped native command changed')
                actual=[int(p['sim_step_id'][j]) for j in range(len(p['sim_step_id'])) if str(p['command_id'][j])==c['command_id']]
                require(actual==c['consuming_sim_step_ids'],'Incorrect consuming step set')
                counts.append(len(actual))
                for sid in actual:
                    j=pi[sid];pubid=int(p['publication_id'][j]);require(pubid in pubs and pubid in receipt_by_pub,'Consumed command has no publish/receipt')
                    verify_receipt_edge(c['command_id'],int(d['native_decision_number'][u]),pubid,pubs,receipt_by_pub,int(raw['end_ns'][j]))
                    receipt=receipt_by_pub[pubid]
                    require(pubs[pubid]['command_id']==c['command_id'],'Publication belongs to other action')
                    require(pubs[pubid]['monotonic_ns']<=receipt['receipt_time']<=int(raw['end_ns'][j]),'Publish/receipt/consume causality mismatch')
                    require(np.array_equal(p['applied_motor_values'][j],d['native_motor_command'][u]),'Actual selected DDS fields differ from native command')
                    used.append(sid)
                if actual:
                    k=si[c['post_state_id']];last=pi[actual[-1]]
                    require(np.array_equal(post['integration_state'][k],p['post_integration_state'][last]),'Command post-state mismatch')
                if c['phase']=='active': require(bool(actual),'Active command was never consumed')
            require(len(used)==len(set(used)),'Physics consumed multiple body commands')
            for j in range(len(p['sim_step_id'])):
                require(p['body_control_source'][j]==('native_DDS' if p['playing'][j] else 'startup_WBC'),'Body control source mismatch')
                if p['playing'][j]:require(str(p['command_id'][j]) in cids,'Unidentified active command')
                else:require(str(p['command_id'][j]) not in cids,'Startup WBC mislabeled as consumed DDS command')
                frame=int(p['hand_reference_index'][j]);require(np.array_equal(p['hand_ref_q'][j],a['hand_ref_q'][frame]),'Hand timing/intent changed')
            return dict(commands=len(commands),consuming_steps=len(used),min_consuming_steps=min(counts),max_consuming_steps=max(counts),
                transport='native_DDS',fixed_four_step_assumption=False,hand_cursor_logged_independently=True,
                startup_wbc_steps=int(sum(~p['playing'])),selected_dds_distinguished_from_effective_body_control=True)
        check('command_trace',trace)
        def transitions():
            from expert_trajectory import JointMap,PhysicalGraspEvaluator
            from hand_trajectory_controller import HandTrajectoryController
            from run_sonic_grasp import TabletopDDS
            par=parameters();motor=JointMap(reader.model,par['motor_names'],29)
            reader.restore(raw['pre_integration_state'][0]);evaluator=PhysicalGraspEvaluator(reader.model,reader.data,task_arm=m.get("task_arm","right"))
            # Reuse the ORIGINAL MuJoCo command application method for reconstruction.
            sim=TabletopDDS.__new__(TabletopDDS);sim.model=reader.model;sim.data=reader.data;sim.motor=motor
            sim.body=reader.body;sim.hand=reader.hand;sim.publish=lambda:None;sim.startup_wbc=None
            sim.final_band_enabled=False;sim.startup_support=False;sim.support_mode='none';sim.command_received_steps=0;sim.evaluator=evaluator
            sim.hand_controller=HandTrajectoryController(reader.model,a['time_s'],a['hand_ref_q'])
            errors=[];qualifying_table=0
            for j,sid in enumerate(p['sim_step_id']):
                if j:require(np.array_equal(p['pre_integration_state'][j],p['post_integration_state'][j-1]),'Broken physical state chain')
                if not p['playing'][j]:continue
                mujoco.mj_setState(reader.model,reader.data,p['pre_integration_state'][j],reader.state_spec)
                # mj_setState does not restore derived body poses/bias terms required by step().
                mujoco.mj_forward(reader.model,reader.data)
                values=p['applied_motor_values'][j]
                msg=SimpleNamespace(motor_cmd=[SimpleNamespace(q=float(v[0]),dq=float(v[1]),kp=float(v[2]),kd=float(v[3]),tau=float(v[4])) for v in values])
                sim.bridge=SimpleNamespace(low_cmd_lock=threading.Lock(),low_cmd_received=True,low_cmd=msg)
                _,_,_,metrics=sim.step(float(a['time_s'][int(p['hand_reference_index'][j])]),True,True)
                err=max(float(np.max(abs(reader.data.qpos-p['post_qpos'][j]))),float(np.max(abs(reader.data.qvel-p['post_qvel'][j]))),float(np.max(abs(reader.data.ctrl-p['applied_ctrl'][j]))))
                errors.append(err);qualifying_table+=int(metrics['valid_hold'] and metrics['cube_table_contact'])
            maxerror=max(errors,default=float('inf'));require(maxerror<=1e-8,'Original step reconstruction mismatch '+str(maxerror))
            for c in commands:
                if not c['consuming_sim_step_ids']:continue
                j=oi[c['observation_id']];sid=int(o['sim_step_id'][j]);require(np.array_equal(o['integration_state'][j],p['pre_integration_state'][pi[sid]]),'Observation not the recorded published state')
                require(c['intervening_sim_step_ids']==list(range(sid,c['first_consuming_sim_step'])),'Missing inference/transport-latency steps')
            actual=evaluator.report(reader.data);saved=read(folder/'replay_validation.json')
            require(abs(actual['max_lift_m']-saved['cube_max_lift'])<1e-9 and abs(actual['max_continuous_hold_seconds']-saved['qualifying_hold_duration'])<1e-9,'Physical metrics differ from original-step evidence')
            success=bool(actual['physical_grasp_success'] and not actual['fall_steps'] and saved['replay_complete'])
            require(success==saved['physical_success'],'Physical success changed')
            report['physical_reconstruction']=dict(valid=True,physical_success=success,steps=len(errors),qualifying_cube_table_steps=qualifying_table)
            return dict(matched=len(errors),total=len(errors),max_absolute_error=maxerror,method='original TabletopDDS.step, original hand PD, actual selected DDS command')
        check('state_transition',transitions)
        if 'physical_reconstruction' not in report:report['physical_reconstruction']=dict(valid=False,error='State reconstruction failed')
        def future():
            samples=plan_samples(folder);require(bool(samples),'No valid future horizons')
            zi=index(e,'token_id');hi=index(r,'hand_ref_id')
            for sample in samples:
                body=[str(e['reference_frame_id'][zi[x]]) for x in sample['body_token_ids']]
                hand=[str(r['reference_frame_id'][hi[x]]) for x in sample['hand_ref_ids']]
                require(body==hand==sample['future_reference_frame_ids'],'Body/hand target reference IDs differ')
                t=sample['anchor_execution_index'];require([int(r['source_reference_index'][ri[x]]) for x in body]==list(range(t+2,t+7)),'Future offsets changed')
            return dict(samples=len(samples),future_offsets=[2,3,4,5,6],body_target_reference_ids_equal_hand_target_reference_ids=True)
        check('future_target_alignment',future)
    finally:reader.close()
    report['failures']=failures
    return report
