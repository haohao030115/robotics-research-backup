#!/usr/bin/env python3
"""Paired evidence for raw -> resampled -> adapted -> native motion streams.

This analysis is read-only with respect to source runs.  It uses the same source
episode seed at every stage, and emits stage diversity plus per-episode
transition distortion.  It deliberately treats producer generated motion as
body-only: neither Kimodo nor ARDY exports model-generated hand joints.  Hand
metrics therefore begin at the adapter's hand reference and continue through
native execution.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/unified_adapter_v2/joint_optimization_20261017"
RESULT_DIR = RUN / "analysis/paired_motion_diversity_v2"
ROBOT = ROOT / "GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml"
SCENE = ROOT / "dl-group2/scenes/tabletop.xml"

ARDY_SOURCE_NAMES = [
    'left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint',
    'left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint',
    'right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint',
    'right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
    'waist_yaw_joint','waist_roll_joint','waist_pitch_joint',
    'left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint',
    'left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint',
    'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint',
    'right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint',
]


def finite(a):
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("non-finite motion array")
    return a


def load_json(path):
    with open(path) as f:
        return json.load(f)


def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resample_qpos(q, fps, n=810):
    q = finite(q)
    t = np.arange(n, dtype=float) / 50.0
    source_t = np.arange(len(q), dtype=float) / float(fps)
    if q.ndim != 2 or q.shape[1] != 36:
        raise ValueError(f"expected qpos(T,36), got {q.shape}")
    if source_t[-1] < t[-1] - 1e-12:
        raise ValueError("raw stream does not cover 16.18 s")
    out = np.empty((n, 36), dtype=float)
    out[:, :3] = np.column_stack([np.interp(t, source_t, q[:, j]) for j in range(3)])
    out[:, 7:] = np.column_stack([np.interp(t, source_t, q[:, j]) for j in range(7, 36)])
    quat = q[:, [4, 5, 6, 3]]
    if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("raw root quaternion not unit")
    out[:, 3:7] = Slerp(source_t, Rotation.from_quat(quat))(t).as_quat()[:, [3, 0, 1, 2]]
    return out


def canonical_from_qpos(q, source_names, canonical_names):
    q = finite(q)
    if q.shape[1] != 36:
        raise ValueError(q.shape)
    ix = [source_names.index(n) for n in canonical_names]
    return np.column_stack([q[:, :7], q[:, 7 + np.asarray(ix)]])


def setup_fk():
    # The scene loader is used only for CPU FK; no renderer is initialized.
    sys.path[:0] = [str(ROOT / "dl-group2/scripts"), str(ROOT / "GR00T-WholeBodyControl")]
    import project_paths
    project_paths.default_robot_xml = lambda: ROBOT
    from scene import load_scene
    import mujoco
    m, d = load_scene(supported=False, robot_path=ROBOT)
    wrist = m.body('right_wrist_yaw_link').id
    body_names = [m.joint(n).name for n in CANONICAL_NAMES]
    body_adr = np.array([m.joint(n).qposadr[0] for n in CANONICAL_NAMES], dtype=int)
    return mujoco, m, d, wrist, body_adr


def qpos_fk_wrist(qpos, mujoco, model, data, wrist, body_adr):
    qpos = finite(qpos)
    if qpos.shape[1] != 36:
        raise ValueError(qpos.shape)
    out = np.empty((len(qpos), 3), dtype=float)
    for k, row in enumerate(qpos):
        data.qpos[:] = 0
        data.qpos[:7] = row[:7]
        data.qpos[body_adr] = row[7:]
        mujoco.mj_forward(model, data)
        out[k] = data.xpos[wrist]
    return out


def feature_from_qpos(qpos, source_names, mujoco, model, data, wrist, body_adr):
    canonical = canonical_from_qpos(qpos, source_names, CANONICAL_NAMES)
    return {
        'body': canonical[:, 7:],
        'root_pos': canonical[:, :3],
        'root_quat': canonical[:, 3:7],
        'wrist_pos': qpos_fk_wrist(canonical, mujoco, model, data, wrist, body_adr),
        'hand': None,
    }


def feature_from_reference(z, mujoco, model, data, wrist, body_adr, native=False):
    if native:
        body = finite(z['body_q']); hand = finite(z['hand_q'])
        root = finite(z['root_pos']); quat = finite(z['root_quat'])
        # Native recordings include eef_pos; use it only when complete and finite,
        # otherwise use the same MuJoCo FK path as the reference stage.
        if 'eef_pos' in z and np.asarray(z['eef_pos']).shape == (len(body), 3) and np.isfinite(z['eef_pos']).all():
            wrist = finite(z['eef_pos'])
        else:
            wrist = qpos_fk_wrist(np.column_stack([root, quat, body]), mujoco, model, data, wrist, body_adr)
    else:
        body = finite(z['body_ref_q']); hand = finite(z['hand_ref_q'])
        root = finite(z['root_ref_pos']); quat = finite(z['root_ref_quat'])
        wrist = qpos_fk_wrist(np.column_stack([root, quat, body]), mujoco, model, data, wrist, body_adr)
    return {'body': body, 'root_pos': root, 'root_quat': quat, 'wrist_pos': wrist, 'hand': hand}


def align_rows(x, n=810):
    x = finite(x)
    if len(x) == n:
        return x
    # qpos streams should use resample_qpos; this helper is for optional ablation
    # rows where only a 50Hz qpos CSV is available.
    old = np.linspace(0, 1, len(x)); new = np.linspace(0, 1, n)
    return np.column_stack([np.interp(new, old, x[:, j]) for j in range(x.shape[1])])


def stage_metrics(stages):
    out = {}
    for name, rows in stages.items():
        vals = {'frames': int(len(rows['body'])), 'body_dof': int(rows['body'].shape[1]),
                'hand_available': rows['hand'] is not None}
        for key, unit in [('body','rad'), ('wrist_pos','m'), ('hand','rad')]:
            a = rows.get(key)
            if a is None:
                vals[f'{key}_pairwise_rms'] = None
                vals[f'{key}_pairwise_max'] = None
                continue
            # Compare motion trajectories after each episode's first frame. This
            # separates strategy variation from a constant reset offset.
            x = np.asarray(a, dtype=float)
            x = x - x[:, :1]
            vals[f'{key}_pairwise_rms'] = unit_pairwise(x)
            vals[f'{key}_pairwise_max'] = unit_pairwise(x, mode='max')
        out[name] = vals
    return out


def unit_pairwise(arrays, mode='mean'):
    # arrays is (episodes, frames, dof...) or one episode? This function is
    # called with a stacked array by population_metrics; retained for clarity.
    a = np.asarray(arrays, dtype=float)
    if a.ndim < 3 or len(a) < 2:
        return None
    vals = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            d = np.sqrt(np.mean((a[i] - a[j]) ** 2))
            vals.append(float(d))
    if mode == 'max':
        return float(max(vals))
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
            'min': float(min(vals)), 'max': float(max(vals)), 'pairs': len(vals)}


def population_metrics(episodes, stage_name):
    vals = [e['stages'][stage_name] for e in episodes if stage_name in e['stages']]
    if len(vals) < 2:
        return {'episodes': len(vals), 'body_pairwise_rms': None, 'wrist_pairwise_rms': None, 'hand_pairwise_rms': None}
    result = {'episodes': len(vals)}
    for key in ('body', 'wrist_pos', 'hand'):
        available = [v[key] for v in vals if v.get(key) is not None]
        if len(available) < 2:
            result[f'{key}_pairwise_rms'] = None
            result[f'{key}_pairwise_max'] = None
            continue
        # Normalize all stage trajectories to per-episode displacement for
        # strategy diversity; also retain absolute RMS for compatibility with
        # the historical diversity report.
        x_abs = np.stack(available, axis=0)
        x = np.stack([a - a[:1] for a in available], axis=0)
        result[f'{key}_pairwise_rms'] = unit_pairwise(x)
        result[f'{key}_pairwise_max'] = unit_pairwise(x, mode='max')
        result[f'{key}_absolute_pairwise_rms'] = unit_pairwise(x_abs)
    return result


def rms(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def transition_metrics(stages):
    out = {}
    res = stages.get('resampled_g1'); ad = stages.get('adapted_reference'); nat = stages.get('native_executed')
    if res and ad:
        out['resampled_to_adapted_body_rms_rad'] = rms(ad['body'] - res['body'])
        out['resampled_to_adapted_body_max_rad'] = float(np.max(np.abs(ad['body'] - res['body'])))
        out['resampled_to_adapted_wrist_rms_m'] = rms(ad['wrist_pos'] - res['wrist_pos'])
        out['resampled_to_adapted_wrist_max_m'] = float(np.max(np.linalg.norm(ad['wrist_pos'] - res['wrist_pos'], axis=1)))
        out['resampled_to_adapted_root_rms_m'] = rms(ad['root_pos'] - res['root_pos'])
        out['adapted_hand_range_rad'] = float(np.max(ad['hand'], axis=0).max() - np.min(ad['hand'], axis=0).min()) if ad['hand'] is not None else None
    if ad and nat:
        out['adapted_to_native_body_rms_rad'] = rms(nat['body'] - ad['body'])
        out['adapted_to_native_body_max_rad'] = float(np.max(np.abs(nat['body'] - ad['body'])))
        out['adapted_to_native_wrist_rms_m'] = rms(nat['wrist_pos'] - ad['wrist_pos'])
        out['adapted_to_native_wrist_max_m'] = float(np.max(np.linalg.norm(nat['wrist_pos'] - ad['wrist_pos'], axis=1)))
        out['adapted_to_native_hand_rms_rad'] = rms(nat['hand'] - ad['hand']) if nat['hand'] is not None and ad['hand'] is not None else None
        out['adapted_to_native_hand_max_rad'] = float(np.max(np.abs(nat['hand'] - ad['hand']))) if nat['hand'] is not None and ad['hand'] is not None else None
    if res and nat:
        out['resampled_to_native_body_rms_rad'] = rms(nat['body'] - res['body'])
        out['resampled_to_native_wrist_rms_m'] = rms(nat['wrist_pos'] - res['wrist_pos'])
    return out


def load_entry(producer, seed, raw, raw_meta, adapted, native, source_names, mujoco, model, data, wrist, body_adr, adapter_label):
    raw = Path(raw)
    qraw = np.loadtxt(raw, delimiter=',', ndmin=2)
    fps = int(load_json(raw_meta).get('fps', load_json(raw_meta).get('source_fps', 30))) if raw_meta.suffix == '.json' else 30
    # generation.json for ARDY records source_fps; Kimodo records fps.
    gm = load_json(raw_meta)
    fps = int(gm.get('fps', gm.get('source_fps', fps)))
    raw_stage = feature_from_qpos(qraw, source_names, mujoco, model, data, wrist, body_adr)
    q50 = resample_qpos(qraw, fps)
    resampled_stage = feature_from_qpos(q50, source_names, mujoco, model, data, wrist, body_adr)
    z = np.load(adapted, allow_pickle=False)
    adapted_stage = feature_from_reference(z, mujoco, model, data, wrist, body_adr, native=False)
    nz = np.load(native, allow_pickle=False) if native else None
    native_stage = feature_from_reference(nz, mujoco, model, data, wrist, body_adr, native=True) if nz is not None else None
    stages = {'raw_generated': raw_stage, 'resampled_g1': resampled_stage, 'adapted_reference': adapted_stage}
    if native_stage is not None:
        stages['native_executed'] = native_stage
    result = {
        'producer': producer, 'seed': int(seed), 'adapter': adapter_label,
        'raw_fps': fps, 'raw_frames': int(len(qraw)), 'reference_frames': int(len(q50)),
        'source_paths': {'raw_generated': str(raw), 'raw_metadata': str(raw_meta), 'adapted_reference': str(adapted), 'native_executed': str(native) if native else None},
        'source_hashes': {k: sha256(v) for k,v in [('raw_generated',raw),('raw_metadata',raw_meta),('adapted_reference',adapted),('native_executed',native)] if v and Path(v).exists()},
        'stages': stages,
        'transition_metrics': transition_metrics(stages),
        'stage_metrics': stage_metrics(stages),
    }
    # Native result metadata is evidence, not used to define diversity.
    if native:
        rpath = Path(native).with_name('result.json')
        if rpath.exists():
            r = load_json(rpath)
            result['native_result'] = {k:r.get(k) for k in ('physical_success','expert_valid','max_lift_m','max_continuous_hold_seconds','support_contact_steps','body_tracking_rmse_rad','failure_reasons')}
    return result


def write_table(analysis, path):
    rows=[]
    for e in analysis['episodes']:
        t=e['transition_metrics']; rows.append({
            'producer':e['producer'],'seed':e['seed'],'adapter':e['adapter'],
            'resampled_to_adapted_body_rms_rad':t.get('resampled_to_adapted_body_rms_rad'),
            'resampled_to_adapted_wrist_rms_m':t.get('resampled_to_adapted_wrist_rms_m'),
            'adapted_to_native_body_rms_rad':t.get('adapted_to_native_body_rms_rad'),
            'adapted_to_native_wrist_rms_m':t.get('adapted_to_native_wrist_rms_m'),
            'adapted_to_native_hand_rms_rad':t.get('adapted_to_native_hand_rms_rad'),
            'physical_success':e.get('native_result',{}).get('physical_success') if e.get('native_result') else None,
            'strict_expert_valid':e.get('native_result',{}).get('expert_valid') if e.get('native_result') else None,
        })
    import csv
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['producer','seed']); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=RESULT_DIR); args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    # Canonical order is read from a Kimodo task contract; this is immutable for G1.
    global CANONICAL_NAMES
    task=load_json(RUN/'fresh_seed_batch/kimodo_seed7/task/task.json')
    CANONICAL_NAMES=task['body_joint_order']
    mujoco,model,data,wrist,body_adr=setup_fk()
    episodes=[]
    for seed in range(6,11):
        base=RUN/f'fresh_seed_batch/kimodo_seed{seed}'
        if seed==6:
            adapted=RUN/'kimodo_neutral_body_ablation/kimodo_seed6_approach_refine/reference/reference.npz'; native=RUN/'kimodo_neutral_body_ablation/kimodo_seed6_approach_refine/qualification/simulation.npz'
        else:
            adapted=RUN/f'kimodo_approach_refine_batch_v2/kimodo_seed{seed}/reference/reference.npz'; native=RUN/f'kimodo_approach_refine_batch_v2/kimodo_seed{seed}/qualification/simulation.npz'
        episodes.append(load_entry('kimodo',seed,base/'generation/raw_motion.csv',base/'generation/generation.json',adapted,native,task['source_qpos_joint_order'],mujoco,model,data,wrist,body_adr,'unified+approach_refine'))
    for seed in (7,8,9,10,12):
        src=RUN/f'ardy_seed_batch_v4/ardy_center_seed{seed}_unified/source'; base=RUN/f'ardy_seed_batch_v4/ardy_center_seed{seed}_calibrated_relief' if seed==7 else RUN/f'ardy_seed_batch_v4/ardy_center_seed{seed}_calibrated_control'
        episodes.append(load_entry('ardy',seed,src/'reference/motion/ardy_motion.csv',src/'generation.json',base/'replay/input_reference.npz',base/'replay/simulation.npz',ARDY_SOURCE_NAMES,mujoco,model,data,wrist,body_adr,'unified+calibration'+('+thumb_relief' if seed==7 else '')))
    analysis={'schema_version':'paired_motion_diversity_v2','run_date':'2026-10-17','scope':'matched accepted fresh expansion; no generation reruns; source streams preserved','episodes':episodes,'population_metrics':{}}
    for producer in ('kimodo','ardy'):
        subset=[e for e in episodes if e['producer']==producer]
        analysis['population_metrics'][producer]={stage:population_metrics(subset,stage) for stage in ('raw_generated','resampled_g1','adapted_reference','native_executed')}
        # Means of transition distortions expose where strategy variation is modified.
        keys=sorted({k for e in subset for k in e['transition_metrics']})
        analysis['population_metrics'][producer]['transition_means']={k:float(np.mean([e['transition_metrics'][k] for e in subset if e['transition_metrics'].get(k) is not None])) for k in keys}
        # Explicit retention ratios expose where cross-episode variation is lost.
        for metric in ('body_pairwise_rms','wrist_pos_pairwise_rms','hand_pairwise_rms'):
            rawv = analysis['population_metrics'][producer]['raw_generated'].get(metric)
            rawmean = rawv.get('mean') if isinstance(rawv, dict) else None
            if rawmean is None or rawmean <= 0:
                continue
            ratios = {}
            for stage in ('resampled_g1','adapted_reference','native_executed'):
                v = analysis['population_metrics'][producer][stage].get(metric)
                mean = v.get('mean') if isinstance(v, dict) else None
                ratios[stage] = None if mean is None else float(mean / rawmean)
            analysis['population_metrics'][producer].setdefault('retention_vs_raw', {})[metric] = ratios
        # Hand has no producer raw stream; report retention beginning at the
        # adapter reference whenever both adapted and native hand arrays exist.
        av = analysis['population_metrics'][producer]['adapted_reference'].get('hand_pairwise_rms')
        nv = analysis['population_metrics'][producer]['native_executed'].get('hand_pairwise_rms')
        if isinstance(av, dict) and av.get('mean', 0) > 0 and isinstance(nv, dict):
            analysis['population_metrics'][producer].setdefault('retention_vs_adapted', {})['hand_pairwise_rms'] = float(nv['mean'] / av['mean'])
    # Read-only targeted adapter ablations on the same raw episode where artifacts exist.
    ablations=[]
    kraw=RUN/'fresh_seed_batch/kimodo_seed6/generation/raw_motion.csv'; kmeta=RUN/'fresh_seed_batch/kimodo_seed6/generation/generation.json'
    for label, refdir, sim in [
        ('kimodo_seed6_preserve0',RUN/'kimodo_neutral_body_ablation/kimodo_seed6_preserve0/reference',RUN/'kimodo_neutral_body_ablation/kimodo_seed6_preserve0/qualification_retry/simulation.npz'),
        ('kimodo_seed6_preserve005',RUN/'kimodo_neutral_body_ablation/kimodo_seed6_preserve005/reference',RUN/'kimodo_neutral_body_ablation/kimodo_seed6_preserve005/qualification/simulation.npz'),
        ('kimodo_seed6_approach_refine',RUN/'kimodo_neutral_body_ablation/kimodo_seed6_approach_refine/reference',RUN/'kimodo_neutral_body_ablation/kimodo_seed6_approach_refine/qualification/simulation.npz'),
    ]:
        e=load_entry('kimodo',6,kraw,kmeta,refdir/'reference.npz',sim,task['source_qpos_joint_order'],mujoco,model,data,wrist,body_adr,label)
        ablations.append({'label':label,'same_raw_episode':'kimodo_seed6','transition_metrics':e['transition_metrics'],'native_result':e.get('native_result'),'paths':e['source_paths']})
    araw=RUN/'ardy_seed_batch_v4/ardy_center_seed7_unified/source/reference/motion/ardy_motion.csv'; ameta=RUN/'ardy_seed_batch_v4/ardy_center_seed7_unified/source/generation.json'
    # Existing no-calibration replay is a matched adapter control; calibrated relief
    # is the accepted variant.  Both are retained as produced artifacts.
    variants=[('ardy_seed7_unified_no_calibration',RUN/'ardy_seed_batch_v4/ardy_center_seed7_unified',RUN/'ardy_seed_batch_v4/ardy_center_seed7_unified/replay/simulation.npz'),('ardy_seed7_calibrated_relief',RUN/'ardy_seed_batch_v4/ardy_center_seed7_calibrated_relief',RUN/'ardy_seed_batch_v4/ardy_center_seed7_calibrated_relief/replay/simulation.npz')]
    for label, root, sim in variants:
        ref=root/'replay/input_reference.npz'
        if not ref.exists(): continue
        e=load_entry('ardy',7,araw,ameta,ref,sim,ARDY_SOURCE_NAMES,mujoco,model,data,wrist,body_adr,label)
        ablations.append({'label':label,'same_raw_episode':'ardy_seed7','transition_metrics':e['transition_metrics'],'native_result':e.get('native_result'),'paths':e['source_paths']})
    analysis['matched_adapter_ablations']=ablations
    # Do not serialize large trajectories; retain compact stage metrics and hashes.
    for e in analysis['episodes']:
        e.pop('stages',None)
    write_table(analysis,out/'paired_episode_metrics.csv')
    with open(out/'paired_motion_diversity.json','w') as f: json.dump(analysis,f,indent=2,allow_nan=False)
    readme='''# Paired motion diversity evidence\n\nThis directory was generated from existing native runs only. Each accepted Kimodo seed 6--10 and ARDY seed 7--10,12 is matched across the same raw generated stream, deterministic 50 Hz resampling, final adapted reference, and native SONIC simulation. Source files are hashed in `paired_motion_diversity.json`.\n\n`population_metrics` reports centered pairwise RMS after subtracting each episode’s first frame, so constant reset offsets do not masquerade as motion diversity; absolute pairwise RMS is retained under `*_absolute_pairwise_rms` for historical comparison. `transition_means` reports per-episode distortion at each boundary. Raw/resampled hand metrics are intentionally null: both producer generation records `model_generated_hands=false`; hand variation begins at the adapter hand reference. `matched_adapter_ablations` compares preserved same-seed Kimodo and ARDY variants without rerunning generation.\n'''
    (out/'README.md').write_text(readme)
    print(json.dumps({'output':str(out),'episodes':len(episodes),'ablations':len(ablations),'kimodo':5,'ardy':5},indent=2))


if __name__=='__main__':
    main()
