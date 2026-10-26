#!/usr/bin/env python3
"""Run actual constrained Kimodo; preserve original generated arrays and raw DOFs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(os.environ.get('KIMODO_ROOT', Path(os.environ.get('RESEARCH_ROOT', Path(__file__).resolve().parents[3])) / 'kimodo-t6/g1-integration')).resolve()
sys.path.insert(0,str(Path(__file__).resolve().parent))

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--gpu',type=int,default=1);ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--segmented',action='store_true');ap.add_argument('--constraint-cfg',type=float,default=2.)
    ap.add_argument('--spatial-conditioned-prompt', action='store_true',
                    help='Append the deterministic case-derived spatial descriptor to all five stage prompts')
    ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    if a.steps<=0 or a.gpu<0 or not np.isfinite(a.constraint_cfg) or a.constraint_cfg<=0:ap.error('Invalid steps/GPU/guidance')
    a.out.mkdir(parents=True,exist_ok=False)
    os.environ['MUSA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    # Resolve the validated local G1 checkpoint explicitly. Without this,
    # Kimodo's loader falls back to a Hugging Face lookup even in offline
    # va-train runs and the new workflow cannot be reproducible.
    os.environ.update(TEXT_ENCODER_MODE='local',TEXT_ENCODER_DEVICE='cpu',
                      CHECKPOINT_DIR=str(ROOT/'checkpoints'), LOCAL_CACHE='true',
                      TEXT_ENCODERS_DIR=str(ROOT/'task8_qualification_v2/assets/text_encoders'),
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    torch.set_num_threads(8)
    import torch_musa
    from kimodo.skeleton import G1Skeleton34
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.motion_rep.conditioning import build_condition_dicts
    from kimodo.constraints import EndEffectorConstraintSet
    from kimodo_pose_constraint_codec import load_pose_constraints
    task=json.loads((a.task/'task.json').read_text());frames=task['source_min_frames']
    assert frames==487 and task['source_fps']==30 and task['duration_s']==16.2
    converter=MujocoQposConverter(G1Skeleton34())
    conditions,checks=load_pose_constraints(a.task/'pose_constraints.json',converter)
    indices,values=build_condition_dicts(conditions)
    for key,entries in indices.items():
        for value in entries:
            # End-effector indexes are (frame,joint) pairs; bound only the frame column.
            frame_index=value[:,0] if value.ndim==2 else value
            assert value.dtype==torch.long and torch.all((frame_index>=0)&(frame_index<frames)),key
    for key,entries in values.items():
        for value in entries:assert torch.isfinite(value).all(),key
    meta=dict(task_id=task['task_id'],model_name='Kimodo-G1-RP-v1',fps=30,source_fps=30,
              requested_frames=frames,source_qpos_joint_order=task['source_qpos_joint_order'],prompt=task['prompt'],seed=task['seed'],
              num_denoising_steps=a.steps,cfg_weight=[2.,a.constraint_cfg],post_processing=False,physical_gpu=a.gpu,logical_device='musa:0',
              source_hashes={str(p.resolve()):digest(p) for p in [a.task/'task.json',a.task/'task_plan.npz',a.task/'pose_constraints.json']},
              model_generation_run=False,physical_replay_run=False,coordinate_validation=checks,
              condition_keys=list(indices),condition_max_frame=max(int((x[:,0] if x.ndim==2 else x).max()) for entries in indices.values() for x in entries))
    write(a.out/'generation.json',meta)
    if a.dry_run:print(json.dumps(meta,indent=2));return
    if not torch.musa.is_available():raise RuntimeError('MUSA unavailable')
    from kimodo import load_model
    print('Loading actual Kimodo checkpoint and local CPU text encoder',flush=True)
    start=time.monotonic();model,resolved=load_model('g1',device='musa:0',return_resolved_name=True)
    model_load_elapsed=time.monotonic()-start
    assert model.fps==30
    # Keep codec validation on CPU; constraints themselves move through the SDK to the actual model device.
    # SDK crop_move recreates end-effector index tensors on CPU while preserving
    # GPU frame indexes/data. This sidecar subclass aligns only tensor placement.
    class DeviceSafeEndEffector(EndEffectorConstraintSet):
        def crop_move(self,start,end):
            return super().crop_move(start,end).to(device='musa:0')
    if a.segmented:
        conditions=[DeviceSafeEndEffector(c.skeleton,c.frame_indices,c.global_joints_positions,c.global_joints_rots,
                    c.smooth_root_2d,joint_names=c.joint_names) if isinstance(c,EndEffectorConstraintSet) else c for c in conditions]
    for c in conditions:c.to(device='musa:0')
    meta['condition_device_adapter']='DeviceSafeEndEffector.crop_move' if a.segmented else None
    checkpoint=ROOT/'checkpoints/Kimodo-G1-RP-v1'
    meta['checkpoint_hashes']={name:digest(checkpoint/name) for name in ('config.yaml','model.safetensors')}
    meta['resolved_model']=resolved;meta['model_load_seconds']=model_load_elapsed
    torch.manual_seed(task['seed']);torch.musa.manual_seed_all(task['seed']);np.random.seed(task['seed'])
    print(f'Generating {frames} genuine frames with {a.steps} denoising steps',flush=True)
    prompts=task['prompt'];lengths=frames
    if a.segmented:
        prompts=['A standing person slowly raises the right hand outside the front edge of a table.',
                 'A standing person reaches the right hand over the table and lowers it beside a red cube.',
                 'A standing person steadily grasps a red cube with the right hand.',
                 'A standing person slowly lifts a red cube with the right hand.',
                 'A standing person holds a red cube steadily in the right hand above the table.']
        lengths=[120,120,60,60,127]
        if a.spatial_conditioned_prompt:
            from research.shared_adapter.spatial_prompt import VERSION, condition_prompts, descriptor
            prompts=condition_prompts(prompts, task['case']['id'])
            spatial_prompt_mode=VERSION
            spatial_prompt_descriptor=descriptor(task['case']['id'])
        else:
            spatial_prompt_mode=None
            spatial_prompt_descriptor=None
    else:
        spatial_prompt_mode=None
        spatial_prompt_descriptor=None
    meta['multi_prompt']=a.segmented;meta['segment_prompts']=prompts;meta['segment_frames']=lengths
    meta['spatial_prompt_mode']=spatial_prompt_mode
    meta['spatial_prompt_descriptor']=spatial_prompt_descriptor
    model_calls=0
    original_generate=model._generate
    def counted_generate(*args,**kwargs):
        nonlocal model_calls
        model_calls+=1
        return original_generate(*args,**kwargs)
    model._generate=counted_generate
    torch.musa.synchronize()
    generation_start=time.monotonic()
    with torch.inference_mode():
        raw=model(prompts,lengths,num_denoising_steps=a.steps,constraint_lst=conditions,
                  cfg_weight=[2.,a.constraint_cfg],multi_prompt=a.segmented,num_samples=1 if a.segmented else None,
                  num_transition_frames=5,post_processing=False,return_numpy=True)
    torch.musa.synchronize()
    generation_elapsed=time.monotonic()-generation_start
    meta.update(model_api_call_count=1,model_call_count=model_calls,warm_up=False)
    if a.segmented:
        if not all(np.asarray(v).shape[0]==1 for v in raw.values()):raise ValueError('Unexpected multi-prompt batch')
        raw={key:np.asarray(value)[0] for key,value in raw.items()}
    expected={'local_rot_mats':(34,3,3),'root_positions':(3,),'posed_joints':(34,3),'global_rot_mats':(34,3,3),
              'smooth_root_pos':(3,),'foot_contacts':(4,),'global_root_heading':(2,)}
    for key,tail in expected.items():
        arr=np.asarray(raw[key])
        if arr.shape!=(frames,*tail) or not np.isfinite(arr).all():raise ValueError(f'Invalid actual model output {key}: {arr.shape}')
    np.savez_compressed(a.out/'raw_motion.npz',**raw)
    tensor={key:torch.as_tensor(raw[key]) for key in ('local_rot_mats','root_positions')}
    q=converter.to_qpos(tensor['local_rot_mats'][None],tensor['root_positions'][None],root_quat_w_first=True,mujoco_rest_zero=False).detach().cpu().numpy()[0]
    if q.shape!=(frames,36) or not np.isfinite(q).all():raise ValueError('Invalid original forward converter output')
    np.savetxt(a.out/'raw_motion.csv',q,delimiter=',',fmt='%.12g')
    meta.update(model_generation_run=True,actual_frames=len(q),actual_source_last_timestamp_s=(len(q)-1)/30,
                generation_wall_seconds=generation_elapsed,generation_process_load_and_generation_seconds=time.monotonic()-start,peak_allocated_gpu_bytes=torch.musa.max_memory_allocated(),
                raw_sha256=digest(a.out/'raw_motion.npz'),raw_qpos_sha256=digest(a.out/'raw_motion.csv'))
    write(a.out/'generation.json',meta);print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
