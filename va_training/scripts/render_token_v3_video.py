"""Render actual recorded closed-loop physics, without stepping or changing inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from diagnostics.phase import phase_match,quantiles
from sonic.release import digest
from utils.config import dump_json


def sustained_cross(values, threshold, count=5):
    hits=np.convolve((np.asarray(values)>threshold).astype(int),np.ones(count,dtype=int),mode='valid')
    indexes=np.flatnonzero(hits==count)
    return int(indexes[0]) if len(indexes) else None


def phase(index):
    return 'prepare' if index<100 else 'reach' if index<400 else 'closure' if index<500 else 'lift/hold'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--learner',required=True);p.add_argument('--expert',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stride',type=int,default=2);args=p.parse_args()
    if args.stride<1 or 50%args.stride:raise ValueError('Stride must divide 50 Hz')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    paths={'expert':Path(args.expert),'learner':Path(args.learner)}
    reports={k:json.loads((v/'report.json').read_text()) for k,v in paths.items()}
    assert reports['expert']['closed_loop_pass'], 'Require successful live oracle comparison'
    for key in ['sonic','runtime']:assert reports['expert']['provenance'][key]==reports['learner']['provenance'][key]
    assert reports['expert']['episode']==reports['learner']['episode']
    assert reports['expert']['criterion']==reports['learner']['criterion']
    hashes={k:{f.name:digest(f) for f in (v/'bootstrap/initial_pose').iterdir()} for k,v in paths.items()}
    assert hashes['expert']==hashes['learner'], 'Initialization differs'
    physics={k:np.load(v/'physics.npz') for k,v in paths.items()};online={k:np.load(v/'live_observations.npz') for k,v in paths.items()}
    n=len(physics['learner']['qpos']);assert 0<n<=len(physics['expert']['qpos'])==800
    # An aborted V3 run is shown only for its actually recorded prefix.
    # Never pad, repeat, or invent physics to make a 16-second success video.
    physics={k:{name:a[name][:n] for name in a.files} for k,a in physics.items()}
    online={k:{name:a[name][:n] for name in a.files} for k,a in online.items()}
    for k in paths:np.testing.assert_array_equal(online[k]['reference_frame'],np.arange(n))
    r=reports['learner'];config=r['provenance']['va_config'];ep=Path(r['provenance']['dataset']['token_label_source']).parents[3]
    reference={k:np.load(ep/(k+'.npy')) for k in ['body_q','body_dq','hand_q','eef_pos']}
    scales={'body_q':.1,'body_dq':1.,'eef_pos':.05,'hand_q':.2}
    right=[i for i,name in enumerate(r['provenance']['hand_joint_order']) if name.startswith('right_')]
    assert len(right)==7
    wrist=np.linalg.norm(physics['learner']['eef_pos']-physics['expert']['eef_pos'],axis=-1)*100
    divergence=sustained_cross(wrist,5)
    hand_measured={k:physics[k]['hand_q'][:,right].mean(-1) for k in paths}
    hand_ref={k:online[k]['hand_ref'][:,right].mean(-1) for k in paths}
    # Above initial right-hand mean by 0.10 rad for five consecutive 50-Hz ticks.
    closure={k:{kind:sustained_cross(values,values[0]+.1) for kind,values in [('measured',hand_measured[k]),('reference',hand_ref[k])]} for k in paths}
    nearest={k:phase_match(physics[k],reference,scales) for k in paths}
    for k in paths:
        a=physics[k];z=online[k]
        np.savez_compressed(out/(k+'_synchronized.npz'),control_step=np.arange(n),control_time_s=np.arange(n)/50,
            simulation_time_s=z['sim_timestamps'],simulation_elapsed_s=z['sim_timestamps']-z['sim_timestamps'][0],
            body_q=a['body_q'],body_dq=a['body_dq'],hand_q=a['hand_q'],hand_dq=z['hand_dq'],
            body_token_consumed=z['token'],hand_reference_applied=z['hand_ref'],previous_raw_action=z['previous_action'],
            cube_pos=a['cube_pos'],cube_quat=a['cube_quat'],wrist_pos=a['eef_pos'],wrist_quat=a['eef_quat'],
            root_pos=a['root_pos'],root_quat=a['root_quat'],nearest_expert_index=nearest[k],phase_lag_ticks=nearest[k]-np.arange(n),
            hand_mean_right=hand_measured[k],hand_reference_mean_right=hand_ref[k],qpos=a['qpos'],qvel=a['qvel'],ctrl=a['ctrl'])
    sys.path.insert(0,config['sonic']['repository']);sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'dl-group2/scripts'))
    import scene,mujoco,cv2
    from PIL import Image,ImageDraw,ImageFont
    meta=json.loads((ep/'metadata.json').read_text())
    initial_cube_z=float(meta['task_config']['cube_initial_position'][2])
    model,data=scene.load_scene(supported=False,robot_path=Path(config['sonic']['robot_xml']),root_height=meta.get('source_root_height_m'))
    renderer=mujoco.Renderer(model,height=480,width=640)
    for cam in ['task_overview','head_rgb','wrist_rgb']:assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,cam)>=0
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',16)
    except OSError:font=ImageFont.load_default()
    fps=50/args.stride;writers={k:cv2.VideoWriter(str(out/(k+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,(960,624)) for k in paths}
    writers['comparison']=cv2.VideoWriter(str(out/'expert_vs_v3.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,624))
    if not all(w.isOpened() for w in writers.values()):raise RuntimeError('MP4 encoder unavailable')
    def panel(k,i):
        a=physics[k];data.qpos[:]=a['qpos'][i];data.qvel[:]=a['qvel'][i];data.ctrl[:]=a['ctrl'][i];data.time=float(online[k]['sim_timestamps'][i])
        mujoco.mj_forward(model,data) # No integration, contact control, or reference playback.
        canvas=np.zeros((624,960,3),dtype=np.uint8)
        for cam,x,y,small in [('task_overview',0,0,False),('head_rgb',640,0,True),('wrist_rgb',640,240,True)]:
            renderer.update_scene(data,camera=cam);rgb=renderer.render().copy()
            if small:rgb=cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA)
            canvas[y:y+rgb.shape[0],x:x+rgb.shape[1]]=rgb
        img=Image.fromarray(canvas);draw=ImageDraw.Draw(img)
        for text,x,y in [((('V3 | TIMING FAILED' if not r.get('timing_contract_pass',False) else 'V3') if k=='learner' else 'EXPERT')+' | actual physics',8,8),('head_rgb',648,8),('wrist_rgb',648,248)]:
            draw.rectangle((x-3,y-2,x+len(text)*10,y+20),fill=(10,10,10));draw.text((x,y),text,font=font,fill='white')
        z=float(a['cube_pos'][i,2]);lift=z-initial_cube_z
        lines=[f'control {i/50:5.2f}s | sim {data.time:7.3f}s | step {i:03d}/799',
            f'nominal plan: {phase(i)} | matched: {nearest[k][i]:03d} ({phase(nearest[k][i])})',
            f'cube z {z:.3f}m | lift {lift:+.3f}m | phase lag {nearest[k][i]-i:+d} ticks',
            f'right hand mean {hand_measured[k][i]:.3f}rad | applied ref {hand_ref[k][i]:.3f}rad',
            f'wrist separation {wrist[i]:.2f}cm | matched phase is diagnostic only']
        for j,text in enumerate(lines):draw.text((10,486+26*j),text,font=font,fill='white')
        return np.asarray(img)
    count=0
    try:
        for i in range(0,n,args.stride):
            panels={k:panel(k,i) for k in paths}
            for k,img in panels.items():writers[k].write(cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            pair=np.concatenate([panels['expert'],panels['learner']],axis=1)
            writers['comparison'].write(cv2.cvtColor(pair,cv2.COLOR_RGB2BGR));count+=1
            if i in [0,400] or (divergence is not None and i==((divergence+args.stride-1)//args.stride)*args.stride):Image.fromarray(pair).save(out/f'comparison_step_{i:03d}.png')
    finally:
        renderer.close()
        for w in writers.values():w.release()
    verification={}
    for file in ['expert.mp4','learner.mp4','expert_vs_v3.mp4']:
        capture=cv2.VideoCapture(str(out/file));frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));actual_fps=capture.get(cv2.CAP_PROP_FPS)
        readable=0
        while True:
            ok,_=capture.read()
            if not ok:break
            readable+=1
        capture.release();assert readable==frames==count and abs(actual_fps-fps)<1e-6
        verification[file]={'frames':frames,'fps':actual_fps,'duration_seconds':frames/actual_fps,'all_frames_decoded':True,'sha256':digest(out/file)}
    summary={'episode':r['episode'],'checkpoint':r['provenance']['checkpoint'],'checkpoint_sha256':r['provenance']['checkpoint_sha256'],
        'recorded_video_control_ticks':n,'expected_control_ticks':800,'video_truncated_by_abort':n<800,
        'v3_timing_contract_pass':r.get('timing_contract_pass',False),'v3_stale_ticks':r.get('buffer_stale_control_ticks'),
        'task':{k:{name:reports[k][name] for name in ['closed_loop_pass','max_lift_m','max_continuous_hold_seconds','fall_steps']} for k in paths},
        'first_visible_divergence_s':None if divergence is None else divergence/50,'divergence_rule':'wrist separation >5 cm for five consecutive 50-Hz control states; visual proxy, not a manually judged onset',
        'hand_closure_start_s':{k:{name:None if ix is None else ix/50 for name,ix in vals.items()} for k,vals in closure.items()},
        'closure_rule':'right seven-joint mean exceeds its own initial value by 0.10 rad for five ticks; measured and applied reference reported separately',
        'phase_labels':'nominal plan windows from prior diagnosis; nearest-state index is diagnostic only, not inferred ground-truth task phase',
        'phase_matching_scales':scales,'learner_phase_lag_ticks':quantiles(nearest['learner']-np.arange(n)),
        'rendered_from':'fresh live VA + native SONIC physics qpos/qvel at recorded post-step states; no mj_step or generated reference trajectory used in rendering',
        'cameras':['task_overview','head_rgb','wrist_rgb'],'video_stride':args.stride,'video_control_steps':list(range(0,n,args.stride)),
        'initial_cube_height_m':initial_cube_z,'codec':'MP4V (OpenCV existing FFmpeg backend)','videos':verification,'bootstrap_hashes':hashes,
        'sources':{k:{str(paths[k]/name):digest(paths[k]/name) for name in ['report.json','physics.npz','live_observations.npz']} for k in paths},
        'raw_model_predictions':str(paths['learner']/'predictions.npz'),'raw_native_logs':str(paths['learner']/'policy_logs')}
    dump_json(out/'report.json',summary)
    print(json.dumps({k:summary[k] for k in ['episode','checkpoint','task','first_visible_divergence_s','hand_closure_start_s','videos']},indent=2),flush=True)


if __name__=='__main__':main()
