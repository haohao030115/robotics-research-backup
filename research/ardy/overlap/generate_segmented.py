"""Kimodo-style conditional overlap transitions using native ARDY sampling."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
ARDY = Path(os.environ.get('ARDY_REPO', WORKSPACE / 'workspace/fuyuhan/ardy')).resolve()
sys.path.insert(0, str(ARDY))
from ardy.constraints import FullBodyConstraintSet, EndEffectorConstraintSet
from ardy.exports.mujoco import MujocoQposConverter
from ardy.model import load_model
from ardy.motion_rep.tools import length_to_mask
from ardy.tools import seed_everything, to_numpy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    device = 'cuda:0'
    model = load_model('ARDY-G1-RP-25FPS-Horizon52', device=device,
                       checkpoints_dir=str(ARDY/'checkpoints'))
    fps = model.motion_rep.fps
    steps = cfg['diffusion_steps']
    if fps != 25 or not 1 <= steps <= model.diffusion.num_base_steps:
        raise ValueError('Expected G1 25Hz checkpoint and a supported diffusion schedule')
    rep = model.motion_rep
    # Like Kimodo's sidecar codec, preserve local rotation matrices instead of
    # serializing through axis-angle and validate the canonical qpos roundtrip.
    saved = np.load(args.source/'pose_matrices.npz')
    local = torch.as_tensor(saved['local_rot_mats'], dtype=torch.float32, device=device)
    roots = torch.as_tensor(saved['root_positions'], dtype=torch.float32, device=device)
    converter = MujocoQposConverter(model.skeleton)
    restored = to_numpy(converter.dict_to_qpos({'local_rot_mats': local[None],
                                               'root_positions': roots[None]}, device))[0]
    roundtrip = float(np.max(abs(restored-saved['canonical_qpos'])))
    if roundtrip > 1e-6:
        raise ValueError(f'Saved matrix roundtrip exceeds Kimodo tolerance: {roundtrip}')
    global_rot, positions, _ = model.skeleton.fk(local, roots)
    indices = torch.tensor(saved['frame_indices'], dtype=torch.long)
    rows = torch.tensor([0, 5], dtype=torch.long)
    constraints = [FullBodyConstraintSet(model.skeleton, indices[rows], positions[rows], global_rot[rows]),
                   EndEffectorConstraintSet(model.skeleton, indices, positions, global_rot, None,
                       joint_names=['RightHand', 'LeftHand', 'LeftFoot', 'RightFoot', 'Hips'])]
    transition = cfg['transition_frames']
    prompts, lengths = cfg['prompts'], cfg['segment_frames']
    if len(prompts) != 5 or lengths != [100, 100, 50, 50, 106] or transition != 5:
        raise ValueError('Five canonical segments with complete 16.2s endpoint coverage required')
    patch = model.num_frames_per_token
    history = ((int(10*fps)-model.gen_horizon_len)//patch)*patch
    seed_everything(cfg['seed'])
    # Encode each distinct stage once. Motion history is conveyed by fullbody
    # plus hand/foot transition constraints, exactly as Kimodo _multiprompt.
    text_features = [model._encode_text([text]) for text in prompts]
    generated, current_frame, stage_log = None, 0, []
    stem = args.source/'reference/motion/ardy_motion'
    stage_dir = args.source/'segments'
    stage_dir.mkdir()
    start = time.monotonic()
    with torch.inference_mode():
        for index, (prompt, count, text_embedding) in enumerate(zip(prompts, lengths, text_features)):
            start_wall = time.monotonic()
            relevant = [c.crop_move(current_frame, current_frame+count) for c in constraints]
            segment_length = torch.tensor([count], device=device)
            observed, mask = rep.create_conditions_from_constraints_batched(
                relevant, segment_length, to_normalize=False, device=device)
            heading = torch.zeros(1, device=device)
            latest, init_history = None, None
            if generated is not None:
                latest = generated[:, -transition:]
                last = rep.inverse(latest, is_normalized=False)
                smooth_root = last['smooth_root_pos'][..., [0, 2]]
                # Global task constraints stay on their own timeline. Only
                # the generated tail occupies the prepended overlap frames.
                connection = [
                    FullBodyConstraintSet(model.skeleton, torch.arange(transition),
                        last['posed_joints'][0], last['global_rot_mats'][0], smooth_root[0]),
                    EndEffectorConstraintSet(model.skeleton, torch.arange(transition),
                        last['posed_joints'][0], last['global_rot_mats'][0], smooth_root[0],
                        joint_names=['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']),
                ]
                tail_length = torch.tensor([transition], device=device)
                tail_observed, tail_mask = rep.create_conditions_from_constraints_batched(
                    connection, tail_length, to_normalize=False, device=device)
                observed = torch.cat([tail_observed, observed], dim=1)
                mask = torch.cat([tail_mask, mask], dim=1)
                segment_length += transition
                # ARDY has a native autoregressive history input. Use the first
                # token of the five-frame overlap; its remaining frame and all
                # new frames are sampled under the incoming prompt. Native
                # history encoding owns planar recentering/restoration.
                history_frames = cfg['native_history_frames']
                if history_frames != patch or history_frames >= transition:
                    raise ValueError('Expected one ARDY token inside the overlap')
                init_history = rep.normalize(latest[:, :history_frames])
                heading = None
            total = int(segment_length.item())
            normalized = model([prompt], total, num_denoising_steps=steps,
                pad_mask=length_to_mask(segment_length), first_heading_angle=heading,
                motion_mask=mask, observed_motion=rep.normalize(observed)*mask,
                cfg_weight=tuple(cfg['cfg_weight']),
                crop_history_length=history if init_history is None else None,
                init_history_sequence=init_history,
                text_feat=text_embedding[0], text_pad_mask=text_embedding[1])
            motion = rep.unnormalize(normalized)
            # Save every native sample before any transition blending or IK.
            native = to_numpy(rep.inverse(motion, is_normalized=False))
            np.savez_compressed(stage_dir/f'{index:02d}_native.npz',
                                **{k: v[0] for k, v in native.items()}, fps=fps, text=prompt)
            if latest is None:
                generated = motion
            else:
                alpha = torch.linspace(1, 0, transition, device=device)[None, :, None]
                bridge = latest*alpha + motion[:, :transition]*(1-alpha)
                generated = torch.cat([generated[:, :-transition], bridge, motion[:, transition:]], dim=1)
            current_frame += count
            record = dict(index=index, prompt=prompt, start_frame=current_frame-count,
                end_frame_exclusive=current_frame, new_frames=count,
                overlap_frames=0 if latest is None else transition,
                native_sample_frames=total, denoising_steps=steps,
                native_history_frames=0 if init_history is None else cfg['native_history_frames'],
                autoregressive_windows=int(np.ceil((total-(0 if init_history is None else cfg['native_history_frames']))/model.gen_horizon_len)),
                wall_s=time.monotonic()-start_wall)
            stage_log.append(record)
            (args.source/'segment_progress.json').write_text(json.dumps(stage_log, indent=2)+'\n')
            print(json.dumps(record), flush=True)
        result = to_numpy(rep.inverse(generated, is_normalized=False))
        converter = MujocoQposConverter(model.skeleton)
        qpos = to_numpy(converter.dict_to_qpos(result, device))[0]
    if qpos.shape != (406, 36) or not np.isfinite(qpos).all():
        raise ValueError('Incomplete or invalid segmented model output')
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix('.npz'), **{k: v[0] for k, v in result.items()},
                        fps=fps, text=' | '.join(prompts), segment_prompts=np.array(prompts),
                        segment_frames=np.array(lengths))
    np.savetxt(stem.with_suffix('.csv'), qpos, delimiter=',', fmt='%.12g')
    meta = dict(model='ARDY-G1-RP-25FPS-Horizon52', source_fps=fps, seed=cfg['seed'],
        prompts=prompts, segment_frames=lengths, native_ardy_api_calls=5,
        autoregressive_window_count=sum(s['autoregressive_windows'] for s in stage_log),
        transition_frames=transition, transition_s=transition/fps,
        transition_method='Kimodo fullbody + end-effector tail conditioning and linear feature overlap blend; ARDY native token history owns planar recentering',
        native_history_frames=cfg['native_history_frames'],
        condition_codec='matrix-preserving canonical qpos-to-ARDY FK; same constraint families as Kimodo',
        condition_roundtrip_max_error=roundtrip,
        condition_sha256=hashlib.sha256((args.source/'pose_matrices.npz').read_bytes()).hexdigest(),
        checkpoint_hashes={name: hashlib.sha256((ARDY/'checkpoints/ARDY-G1-RP-25FPS-Horizon52'/name).read_bytes()).hexdigest()
                           for name in ('config.yaml','denoiser.safetensors','tokenizer.safetensors')},
        native_postprocessing=False, model_generated_hands=False, online_prompt=False,
        spatial_prompt_mode=cfg.get('spatial_prompt_mode'),
        spatial_prompt_descriptor=cfg.get('spatial_prompt_descriptor'),
        denoising_steps=steps, cfg_weight=cfg['cfg_weight'], actual_frames=len(qpos),
        last_timestamp_s=(len(qpos)-1)/fps, stages=stage_log, wall_s=time.monotonic()-start,
        raw_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (stem.with_suffix('.npz'), stem.with_suffix('.csv'))})
    (args.source/'generation.json').write_text(json.dumps(meta, indent=2)+'\n')
    spec_path = args.source/'setting.json'
    spec = json.loads(spec_path.read_text())
    spec.update(prompt=' | '.join(prompts), prompts=prompts, segmented=True,
                segment_frames=lengths, transition_frames=transition)
    spec_path.write_text(json.dumps(spec, indent=2)+'\n')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
