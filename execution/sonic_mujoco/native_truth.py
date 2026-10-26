"""Materialize ID-linked native trace. No timestamp nearest-neighbor or row alignment."""
import time
import numpy as np
from PIL import Image
from .common import dump,read,load_npz,lines,table,jsonl,digest,require
from .native_scene import SnapshotReader


def materialize(out,a,m,physical):
    materialize_started=time.perf_counter(); render_seconds=0.; disk_seconds=0.
    from .native_replay import DECISION_DTYPE
    trace=out/'decisions.bin'; require(trace.exists(),'Native decision trace missing')
    require(trace.stat().st_size%DECISION_DTYPE.itemsize==0,'Partial native decision record')
    native=np.fromfile(trace,dtype=DECISION_DTYPE)
    raw=load_npz(out/'native_physics.npz'); receipts=lines(out/'native_receipts.jsonl')
    pub=np.fromfile(out/'publications.bin',dtype='<u8').reshape(-1,3)
    source_by_seq={int(n):j for j,n in enumerate(raw['observation_sequence'])}
    receipt_by_pub={x['publication_id']:x for x in receipts}
    publications={int(x[1]):x for x in pub}
    consumed={}
    for j,c in enumerate(raw['command_number']):
        # Original TabletopDDS applies startup WBC after reading DDS; that overrides body control.
        # A selected DDS message is not an effective body command until playback starts.
        if raw['playing'][j]: consumed.setdefault(int(c),[]).append(j)
    rows={k:[] for k in ('observations','references','sonic_encoder','sonic_decoder','post_states','physics')}
    commands=[]; images=[]; events=[]; reader=SnapshotReader(render=True); episode=m['episode_id']
    def identity(kind,n): return f'{episode}:{kind}:{int(n):08d}'
    def event(kind,ns,**kw): events.append(dict(episode_id=episode,event=kind,monotonic_ns=int(ns),**kw))
    active_frames=[]
    try:
        for x in native:
            h=x['header']; did,seq,tick,frame,playing=map(int,h[:5]); require(seq in source_by_seq,'Native consumed unknown LowState identity')
            j=source_by_seq[seq]; source_step=int(raw['sim_step_id'][j])
            oid=identity('observation',seq); cid=identity('command',did); rid=identity('reference',did); zid=identity('token',did); hid=identity('handref',did)
            ids=dict(observation_id=oid,control_tick_id=identity('tick',did),decision_id=identity('decision',did),reference_frame_id=rid)
            phase=m['phases'][frame] if playing else 'standby'
            reader.restore(raw['pre_integration_state'][j]); snap=reader.snapshot(oid,source_step)
            rows['observations'].append(dict(snap,**ids,phase=phase,lowstate_sequence=seq,lowstate_tick=tick))
            event('OBSERVE',h[5],**ids,sim_step_id=source_step,lowstate_sequence=seq)
            if playing:
                active_frames.append(frame)
                if frame%5==0:
                    rgbid=identity('rgb',did); path=f'rgb/{did:08d}.png'
                    render_start=time.perf_counter(); pixels=reader.render();render_seconds+=time.perf_counter()-render_start
                    disk_start=time.perf_counter();Image.fromarray(pixels).save(out/path);disk_seconds+=time.perf_counter()-disk_start
                    images.append(dict(rgb_frame_id=rgbid,observation_id=oid,sim_step_id=source_step,simulation_time=snap['simulation_time'],path=path,
                        snapshot_sha256=snap['snapshot_sha256'],image_sha256=digest(out/path),render_mode='offline_exact_restore'))
                    event('RGB_SNAPSHOT_BINDING',h[5],**ids,rgb_frame_id=rgbid,sim_step_id=source_step)
            rows['references'].append(dict(**ids,hand_ref_id=hid,reference_index=frame if playing else -did-1,source_reference_index=frame,
                simulation_time=snap['simulation_time'],phase=phase,**{k:a[k][frame] for k in ('hand_ref_q','body_ref_q','body_ref_dq','root_ref_pos','root_ref_quat')}))
            rows['sonic_encoder'].append(dict(**ids,token_id=zid,exact_encoder_input=x['encoder_input'],current_encoder_output=x['token'],current_post_fsq_64d_token=x['token']))
            rows['sonic_decoder'].append(dict(**ids,command_id=cid,token_id=zid,exact_decoder_input=x['decoder_input'],current_64d_token=x['token'],
                proprio_history=x['decoder_input'][64:],previous_action=x['previous_action'],current_raw_29d_action=x['action'],
                postprocessed_body_target=x['command'][0],native_motor_command=x['command'].T,native_decision_number=did))
            event('ENCODE',h[7],**ids,start_ns=int(h[6]),token_id=zid)
            event('DECODE',h[9],**ids,start_ns=int(h[8]),command_id=cid,token_id=zid)
            steps=consumed.get(did,[]); sim_ids=[int(raw['sim_step_id'][v]) for v in steps]
            native_pubs=[int(v[1]) for v in pub if int(v[0])==did]
            received=[receipt_by_pub[v] for v in native_pubs if v in receipt_by_pub]
            pid=identity('poststate',did) if steps else None
            command=dict(**ids,command_id=cid,post_state_id=pid,token_id=zid,hand_ref_id=hid,
                publish_time=int(publications[native_pubs[0]][2]) if native_pubs else None,
                receipt_time=received[0]['receipt_time'] if received else None,publication_ids=native_pubs,
                received_publication_ids=[v['publication_id'] for v in received],
                first_consuming_sim_step=sim_ids[0] if sim_ids else None,last_consuming_sim_step=sim_ids[-1] if sim_ids else None,
                consuming_sim_step_ids=sim_ids,intervening_sim_step_ids=list(range(source_step,sim_ids[0])) if sim_ids else [],
                body_target=x['command'][0].tolist(),motor_values=x['command'].T.tolist(),hand_ref_q=a['hand_ref_q'][frame].tolist(),
                time_unit='monotonic_ns',transport='native_DDS',phase=phase)
            commands.append(command)
            if steps:
                end=steps[-1]; reader.restore(raw['post_integration_state'][end]); post=reader.snapshot(pid,int(raw['sim_step_id'][end])+1)
                rows['post_states'].append(dict(post,post_state_id=pid,observation_id=oid,decision_id=ids['decision_id'],command_id=cid))
                event('POST_STATE',raw['end_ns'][end],**ids,command_id=cid,post_state_id=pid,sim_step_id=int(raw['sim_step_id'][end])+1)
        command_by_number={int(x['header'][0]):identity('decision',int(x['header'][0])) for x in native}
        for v in pub:
            event('COMMAND_PUBLISH',v[2],command_id=identity('command',v[0]),publication_id=int(v[1]))
        for v in receipts:
            event('COMMAND_RECEIPT',v['receipt_time'],command_id=identity('command',v['command_number']),publication_id=v['publication_id'])
        for j,sid in enumerate(raw['sim_step_id']):
            number=int(raw['command_number'][j]); selected_cid=identity('command',number)
            source='native_DDS' if raw['playing'][j] else 'startup_WBC'
            cid=selected_cid if raw['playing'][j] else identity('startup_wbc',sid)
            reader.restore(raw['post_integration_state'][j])
            rows['physics'].append(dict(sim_step_id=int(sid),command_id=cid,selected_dds_command_id=selected_cid,body_control_source=source,publication_id=int(raw['publication_id'][j]),
                decision_id=command_by_number.get(number,'') if raw['playing'][j] else '',pre_time=raw['pre_time'][j],post_time=raw['post_time'][j],
                pre_integration_state=raw['pre_integration_state'][j],post_integration_state=raw['post_integration_state'][j],
                applied_ctrl=raw['applied_ctrl'][j],post_qpos=reader.data.qpos.copy(),post_qvel=reader.data.qvel.copy(),
                playing=raw['playing'][j],hand_ref_q=raw['hand_ref_q'][j],hand_reference_index=raw['hand_reference_index'][j],
                applied_motor_values=raw['applied_motor_values'][j]))
            event('APPLY',raw['begin_ns'][j],command_id=cid,selected_dds_command_id=selected_cid,body_control_source=source,sim_step_id=int(sid),publication_id=int(raw['publication_id'][j]))
            event('PHYSICS_STEP',raw['end_ns'][j],command_id=cid,sim_step_id=int(sid))
        for name,values in rows.items():
            if values: table(out/(name+'.npz'),values)
        jsonl(out/'commands.jsonl',commands);jsonl(out/'rgb/frames.jsonl',images)
        events.sort(key=lambda x:x['monotonic_ns'])
        for n,event_row in enumerate(events): event_row['event_id']=identity('event',n+1)
        jsonl(out/'timeline.jsonl',events)
        physical.update(control_ticks=len(active_frames),native_total_decisions=len(native),reference_frame_coverage=active_frames==list(range(len(a['time_s']))))
        physical['replay_complete'] &= physical['reference_frame_coverage']
        physical['physical_success'] &= physical['replay_complete']
        physical.setdefault('stage_timings',{}).update(rgb_rendering=render_seconds,disk_writing=disk_seconds,trace_materialization=time.perf_counter()-materialize_started-render_seconds-disk_seconds)
        dump(out/'replay_validation.json',physical)
    finally: reader.close()
    return physical
