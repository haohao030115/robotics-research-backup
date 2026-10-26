"""Validate native release teacher, then create separate learner-state corrective labels."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from datasets.tokens import load_token_episodes,projected_gravity
from diagnostics.teacher import CorrectiveTeacher
from sonic.release import bundle_provenance,digest
from utils.config import dump_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rollout',required=True);p.add_argument('--output',required=True)
    p.add_argument('--target-offset',type=int,default=2);args=p.parse_args()
    path=Path(args.rollout);r=json.loads((path/'report.json').read_text()); config=r['provenance']['va_config']
    data=dict(config['data']);data['episode_ids']=[r['episode']];e=load_token_episodes(data)[0]
    teacher=CorrectiveTeacher(config['sonic'],e)
    native_quat=np.loadtxt(Path(e.provenance['token_label_source']).parent/'base_quat.csv',delimiter=',',skiprows=1)[:len(e.tokens),5:]
    native_teacher=np.array([teacher.token(f,q,native_quat[0]) for f,q in enumerate(native_quat)])
    error=abs(native_teacher-e.tokens)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    validation={'token_mae':float(error.mean()),'max_abs':float(error.max()),'whole_token_exact_fraction':float((error.max(-1)<=1e-6).mean())}
    dump_json(out/'native_reconstruction.json',validation)
    if error.max()>1e-6: raise RuntimeError('Native encoder preprocessing mismatch; do not emit corrective labels')
    online=np.load(path/'live_observations.npz');physics=np.load(path/'physics.npz')
    frames=online['reference_frame'];anchors=frames[(frames%5==0)&(frames+args.target_offset<len(e.tokens))]
    fields=['body_q','body_dq','hand_q','hand_dq','base_quat','base_ang_vel','previous_action']
    state={k:online[k] for k in fields};state['projected_gravity']=projected_gravity(state['base_quat'])
    histories={k:[] for k in state};masks=[];tokens=[];hands=[];rgb=[]
    live_initial=np.loadtxt(path/'policy_logs/base_quat.csv',delimiter=',',skiprows=1)[0,5:]
    sys.path.insert(0,str(Path(config['sonic']['repository'])));sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'dl-group2/scripts'))
    import scene,mujoco
    scene.ROBOT=Path(config['sonic']['robot_xml'])
    meta=json.loads((e.path/'metadata.json').read_text());model,mjdata=scene.load_scene(supported=False,robot_path=scene.ROBOT,root_height=meta.get('source_root_height_m'))
    renderer=mujoco.Renderer(model,height=240,width=320)
    try:
        for anchor in anchors:
            target=int(anchor)+args.target_offset
            tokens.append(teacher.token(target,state['base_quat'][anchor],live_initial));hands.append(e.hand_ref[target])
            indexes=np.arange(anchor-2,anchor+1);masks.append(indexes>=0);indexes=np.maximum(indexes,0)
            for k in state:histories[k].append(state[k][indexes])
            mjdata.qpos[:]=physics['qpos'][anchor];mjdata.qvel[:]=physics['qvel'][anchor]
            mujoco.mj_forward(model,mjdata);renderer.update_scene(mjdata,camera='head_rgb');rgb.append(renderer.render().copy())
    finally: renderer.close()
    np.savez_compressed(out/'corrective_samples.npz',rgb=rgb,capture_anchor=anchors,
        execution_index=anchors+args.target_offset,history_mask=masks,target_sonic_token=tokens,target_hand_ref=hands,**histories)
    nominal=e.tokens[anchors+args.target_offset]; corrective=np.array(tokens)
    dump_json(out/'report.json',{'prototype_pass':True,'samples':len(anchors),'native_validation':validation,
        'episode_id':e.id,'source_episode_id':e.source_episode_id,
        'split_rule':'Retain the original source-episode split when aggregating corrective rows',
        'corrective_vs_nominal_token_mae':float(abs(corrective-nominal).mean()),
        'corrective_vs_nominal_whole_token_changed_fraction':float((abs(corrective-nominal).max(-1)>1e-6).mean()),
        'teacher_mode':'release G1: intended future reference q/dq and relative anchor orientation using learner measured quaternion',
        'teacher_hand':'Nominal intended future hand schedule; not a contact-aware recovery policy',
        'rgb_method':'Re-render recorded learner qpos/qvel in the identical scene; diagnostic prototype, not live collection',
        'no_expert_regeneration':True,'no_training_performed':True,
        'provenance':{'sonic':bundle_provenance(config['sonic']),'rollout_report_sha256':digest(path/'report.json'),
                      'dataset':e.provenance,'reference_folder':str(teacher.reference),'encoder_offsets':teacher.offsets,
                      'target_offset':args.target_offset}})
    print(validation,'corrective samples',len(anchors),flush=True)


if __name__=='__main__':main()
