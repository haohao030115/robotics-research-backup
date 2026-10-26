"""Offline ARDY wrist projection, measured compensation and independent hands."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from prepare_conditions import initialize, ARDY, CONVERTER
from planned_grasp_reference import GraspReference


def smooth(t, a, b):
    u = np.clip((t-a)/(b-a), 0, 1)
    return u*u*(3-2*u)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--adapter', choices=['legacy', 'unified'], default='legacy')
    p.add_argument('--hand-profile', choices=['default','ardy','ardy_thumb_relief','ardy_thumb_relief_balanced','ardy_thumb_relief_medium','ardy_thumb_relief_strong','ardy_thumb_lead_strong','slow_close','early_preshape',
                                              'timing_jitter','coordination_mild','coordination_tiny'],
                   default='ardy', help='Opt-in shared hand policy for unified adaptation')
    p.add_argument('--preserve-generated-body', action='store_true',
                   help='Experimental: retain generated non-right-arm joints while adapting the wrist')
    p.add_argument('--preserve-body-weight', type=float, default=1.0,
                   help='Fraction of generated non-arm displacement to retain (0..1)')
    p.add_argument('--body-preservation-policy', choices=['legacy','bounded'], default='legacy',
                   help='Deterministic generated-body residual policy; bounded applies fixed smoothing, rate, and joint limits')
    p.add_argument('--task-conditioned-clearance', action='store_true',
                   help='Derive bounded approach clearance from table/cube edge margin')
    p.add_argument('--task-clearance-target-margin', type=float, default=0.10,
                   help='Geometry policy target edge margin (same for all positions)')
    p.add_argument('--geometry-approach-corridor', action='store_true',
                   help='Use shared cube/distal-hand geometry to retract the wrist through closure')
    p.add_argument('--geometry-grasp-height', action='store_true',
                   help='Opt in to the geometry probe vertical grasp offset (strictly validated before use)')
    p.add_argument('--opposition-centering', action='store_true',
                   help='Opt in to bounded horizontal distal-pair opposition centering during closure')
    p.add_argument('--opposition-centering-max-correction-m', type=float, default=0.03,
                   help='Shared geometry correction cap (metres), at most 0.03')
    a = p.parse_args()
    if not np.isfinite(a.preserve_body_weight) or not 0 <= a.preserve_body_weight <= 1:
        p.error('preserve-body-weight must be in [0,1]')
    if not np.isfinite(a.opposition_centering_max_correction_m) or not 0 < a.opposition_centering_max_correction_m <= 0.03:
        p.error('opposition-centering-max-correction-m must be in (0,0.03]')
    cfg = json.loads(a.config.read_text())['adapter']
    a.out.mkdir(parents=True, exist_ok=False)
    # Older native ARDY source bundles recorded the generated motion and
    # constraints but did not emit the optional generation/pose sidecars.
    # Keep the source bundle intact while allowing those already-valid raw
    # streams to pass through the unified adapter; provenance still points to
    # the raw motion and source hashes below.
    for name in ('setting.json', 'setting.npz', 'constraints.json', 'generation.json', 'pose_matrices.npz'):
        src = a.source/name
        if src.exists():
            shutil.copy2(src, a.out/name)
    spec = json.loads((a.source/'setting.json').read_text())
    model, data, _, _ = initialize(spec['case']['id'])
    spatial_clearance_m = 0.0
    spatial_clearance_info = None
    if a.task_conditioned_clearance:
        from research.shared_adapter import task_conditioned_clearance
        spatial_clearance_m, spatial_clearance_info = task_conditioned_clearance(
            model, spec['case']['block_xy'], data=data,
            target_edge_margin_m=a.task_clearance_target_margin)
    approach_retraction_m = 0.0
    approach_retraction_info = None
    grasp_height_offset_m = 0.0
    if a.geometry_approach_corridor:
        from research.shared_adapter.grasp_geometry import approach_retraction_m as _approach_retraction_m
        approach_retraction_m, approach_retraction_info = _approach_retraction_m(model)
        grasp_height_offset_m = float(approach_retraction_info['grasp_height_offset_m'])
    sys.path.insert(0, str(CONVERTER.parent))
    ms = importlib.util.spec_from_file_location('sonic_ardy_converter', CONVERTER)
    converter = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(converter)
    raw = a.source/'reference/motion/ardy_motion.csv'
    q = np.loadtxt(raw, delimiter=',', ndmin=2)
    with np.load(raw.with_suffix('.npz')) as npz:
        fps = float(npz['fps'])
    t = np.arange(810)/50
    source_t = np.arange(len(q))/fps
    if fps != 25 or len(q) != 406 or source_t[-1] < t[-1]:
        raise ValueError('Source must cover the full reference without padding or extrapolation')
    resampled = np.empty((len(t), 36))
    for col in list(range(3))+list(range(7,36)):
        resampled[:, col] = np.interp(t, source_t, q[:, col])
    resampled[:, 3:7] = Slerp(source_t, Rotation.from_quat(q[:, [4,5,6,3]]))(t).as_quat()[:, [3,0,1,2]]
    q = resampled
    setting = np.load(a.source/'setting.npz')
    goalp = setting['planned_wrist_pos']
    goalr = Rotation.from_quat(setting['planned_wrist_quat'][:, [1, 2, 3, 0]]).as_matrix()
    if len(q) != len(goalp):
        raise ValueError('Generated and planned duration differ')
    names = [j.get('name') for j in ET.parse(ARDY/'ardy/assets/skeletons/g1skel34/xml/g1.xml').findall('.//worldbody//joint') if j.get('type') != 'free']
    qa = np.array([model.joint(n).qposadr[0] for n in names])
    initial = np.r_[data.qpos[:7], data.qpos[qa]]
    output = np.tile(initial, (len(q), 1))
    # Build the hand trajectory before the optional geometry policy measures
    # distal-link opposition.  This keeps the policy tied to the actual hand
    # state exported for replay and leaves the default path unchanged.
    hn = spec['hand_joint_order']
    if a.adapter == 'unified':
        from research.shared_adapter.hand import HandMotionGenerator, profile_config
        hand, hand_info = HandMotionGenerator(profile_config(a.hand_profile)).generate(
            t, seed=int(spec.get('seed', 0)))
        hand_info['profile'] = a.hand_profile
    else:
        hand, hand_info = setting['hand_ref_q'].copy(), None
    early = cfg['hand_preclose_fraction']*smooth(t, *cfg['hand_preclose_s'])
    late = cfg['hand_preclose_fraction'] + (1-cfg['hand_preclose_fraction'])*smooth(t, cfg['hand_preclose_s'][1], cfg['close_end_s'])
    fraction = np.where(t < cfg['hand_preclose_s'][1], early, late)
    if a.adapter == 'legacy':
        for j, name in enumerate(hn):
            hand[:, j] = fraction*setting['hand_ref_q'][-1, j] if name.startswith('right_') else 0
    if hand.shape != (len(q), 14) or not np.isfinite(hand).all():
        raise ValueError('Invalid hand trajectory for opposition policy')
    hand_qa = np.array([model.joint(name).qposadr[0] for name in hn])
    opposition_anchor = None
    opposition_info = None
    opposition_correction = np.zeros(3)
    plan = GraspReference(model, data)
    armcols = [names.index(model.joint(int(j)).name)+7 for j in plan.arm]
    armset = set(armcols)
    bodycols = [7+i for i in range(len(names)) if 7+i not in armset]
    body_policy = None
    bounded_body = None
    if a.preserve_generated_body:
        source_body = q[:, 7:][:, [c-7 for c in bodycols]]
        initial_body = initial[bodycols]
        if a.body_preservation_policy == 'bounded':
            from research.shared_adapter.body import bounded_body_trajectory
            body_names = [names[c-7] for c in bodycols]
            lower = np.array([model.jnt_range[model.joint(n).id, 0] for n in body_names])
            upper = np.array([model.jnt_range[model.joint(n).id, 1] for n in body_names])
            bounded_body, body_policy = bounded_body_trajectory(
                source_body, initial_body, t, weight=a.preserve_body_weight,
                joint_lower=lower, joint_upper=upper)
        elif a.body_preservation_policy != 'legacy':
            raise ValueError('unknown body preservation policy '+str(a.body_preservation_policy))
    original = mujoco.MjData(model)
    start = data.xpos[plan.wrist].copy()
    startr = data.xmat[plan.wrist].reshape(3, 3).copy()
    calibration = json.loads(a.calibration.read_text()) if a.calibration else None
    if calibration:
        if calibration['case']['id'] != spec['case']['id']:
            raise ValueError('Tracking calibration is specific to its scene case')
        for path, sha in calibration['source_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError('Calibration evidence changed')
    rows = []
    for k, row in enumerate(q):
        original.qpos[:] = data.qpos
        original.qpos[:7] = row[:7]
        original.qpos[qa] = row[7:]
        if a.opposition_centering:
            original.qpos[hand_qa] = hand[k]
        mujoco.mj_forward(model, original)
        rawp = original.xpos[plan.wrist].copy()
        rawr = original.xmat[plan.wrist].reshape(3, 3).copy()
        residual = (Rotation.from_matrix(rawr)*Rotation.from_matrix(goalr[k]).inv()).as_rotvec()
        residual *= min(1., cfg['rotation_radius_rad']/max(np.linalg.norm(residual), 1e-12))
        projectedp = goalp[k] + np.clip(rawp-goalp[k], -cfg['position_halfwidth_m'], cfg['position_halfwidth_m'])
        projectedr = (Rotation.from_rotvec(residual)*Rotation.from_matrix(goalr[k])).as_matrix()
        w = float(smooth(t[k], *cfg['refinement_blend_s']))
        targetp = rawp+w*(projectedp-rawp)
        targetr = Slerp([0, 1], Rotation.from_matrix([rawr, projectedr]))([w]).as_matrix()[0]
        contact_correction = float(np.linalg.norm(targetp-rawp))
        contact_rotation_correction = float((Rotation.from_matrix(targetr)*Rotation.from_matrix(rawr).inv()).magnitude())
        bp, br = np.zeros(3), np.zeros(3)
        if calibration:
            v, cw = float(smooth(t[k], 10, 12)), float(smooth(t[k], 6, 8))
            bp = cw*((1-v)*np.array(calibration['close_position_bias_m']) + v*np.array(calibration['hold_position_bias_m']))
            br = cw*((1-v)*np.array(calibration['close_rotation_bias_rad']) + v*np.array(calibration['hold_rotation_bias_rad']))
            targetp -= bp
            targetr = (Rotation.from_rotvec(-br)*Rotation.from_matrix(targetr)).as_matrix()
        # Geometry-derived clearance is part of the same approach policy as
        # the base margin.  Decay both terms before hand closure; otherwise a
        # near-edge experiment retains the extra height during grasp and
        # becomes physically unreachable.
        clearance_weight = smooth(t[k], *cfg['clearance_blend_s'])*(1-smooth(t[k], *cfg['clearance_decay_s']))
        targetp[2] += (cfg['clearance_m'] + spatial_clearance_m)*clearance_weight
        corridor_weight = 0.0
        corridor_dir = np.zeros(2)
        if approach_retraction_m:
            block_xy = np.asarray(spec['case']['block_xy'], dtype=float)
            corridor_dir = targetp[:2] - block_xy
            norm = float(np.linalg.norm(corridor_dir))
            if norm > 1e-9:
                corridor_dir /= norm
                from research.shared_adapter.grasp_geometry import approach_retraction_weight
                corridor_weight = approach_retraction_weight(t[k])
                targetp[:2] += approach_retraction_m * corridor_weight * corridor_dir
                # The same immutable hand/cube probe supplies a bounded
                # vertical grasp offset.  It starts with the corridor and is
                # held through lift so the pads remain below the cube centre.
                if a.geometry_grasp_height:
                    targetp[2] += grasp_height_offset_m * float(smooth(t[k], 6.5, 7.5))
        if a.opposition_centering:
            from research.shared_adapter.grasp_geometry import opposition_centering_correction
            if opposition_anchor is None and t[k] >= 7.0:
                opposition_correction, opposition_info = opposition_centering_correction(
                    model, original, max_correction_m=a.opposition_centering_max_correction_m)
                opposition_anchor = int(k)
            opposition_weight = 0.0
            if opposition_anchor is not None:
                opposition_weight = float(smooth(t[k], 7.0, 8.0) *
                                         (1.0 - smooth(t[k], 10.0, 11.0)))
                targetp[:2] += opposition_correction[:2] * opposition_weight
        else:
            opposition_weight = 0.0
        entry = float(smooth(t[k], 0, cfg['entry_blend_s']))
        targetp = start+entry*(targetp-start)
        plan.rotation = Slerp([0, 1], Rotation.from_matrix([startr, targetr]))([entry]).as_matrix()[0]
        body_values = None
        if a.preserve_generated_body:
            blend = float(entry)
            source_body = row[7:][[c-7 for c in bodycols]]
            if bounded_body is not None:
                body_values = initial[bodycols] + blend*(bounded_body[k]-initial[bodycols])
            else:
                body_values = initial[bodycols] + blend*a.preserve_body_weight*(source_body-initial[bodycols])
            # Condition IK on the same preserved body that will be exported;
            # applying it after IK would invalidate the wrist solution.
            plan.plan.qpos[bodycols] = body_values
            mujoco.mj_forward(model, plan.plan)
        if k:
            plan.ik(targetp)
        mujoco.mj_forward(model, plan.plan)
        if a.preserve_generated_body:
            # Keep the producer's non-right-arm motion while the wrist/arm is
            # solved by bounded IK.  The entry blend preserves frame zero.
            output[k, bodycols] = body_values
        output[k, armcols] = plan.plan.qpos[plan.aq]
        rows.append(dict(frame=k, time_s=float(t[k]), raw_wrist_world_pos=rawp.tolist(),
                         planned_wrist_world_pos=goalp[k].tolist(), target_wrist_world_pos=targetp.tolist(),
                         position_error_m=float(np.linalg.norm(plan.plan.xpos[plan.wrist]-targetp)),
                         orientation_error_rad=float((Rotation.from_matrix(plan.plan.xmat[plan.wrist].reshape(3,3))*Rotation.from_matrix(plan.rotation).inv()).magnitude()),
                         contact_position_correction_m=contact_correction,
                         contact_rotation_correction_rad=contact_rotation_correction,
                         raw_to_target_correction_m=float(np.linalg.norm(targetp-rawp)),
                         approach_corridor_weight=float(corridor_weight),
                         approach_corridor_retraction_m=float(approach_retraction_m),
                         grasp_height_offset_m=float(grasp_height_offset_m),
                         opposition_centering_weight=float(opposition_weight),
                         opposition_centering_correction_m=opposition_correction.tolist(),
                         tracking_position_compensation_m=bp.tolist(), tracking_rotation_compensation_rad=br.tolist()))
    max_contact = max(r['contact_position_correction_m'] for r in rows)
    (a.out/'wrist_adaptation.json').write_text(json.dumps(rows)+'\n')
    if max_contact > cfg['maximum_position_correction_m']:
        raise ValueError(f'Kimodo-equivalent corridor correction exceeds 10cm: {max_contact}')
    if max(r['position_error_m'] for r in rows) > .02:
        raise ValueError('Kimodo-equivalent offline IK position error exceeds 2cm')
    ref = a.out/'reference/motion' 
    ref.mkdir(parents=True)
    jp, bp, bq = converter.sonic_kinematics(output, ARDY, CONVERTER.parents[1])
    motion = dict(joint_pos=jp, joint_vel=np.gradient(jp, .02, axis=0), body_pos_w=bp, body_quat_w=bq,
                  body_lin_vel_w=np.gradient(bp, .02, axis=0), body_ang_vel_w=converter.world_angular_velocity(bq),
                  _body_indexes=converter.BODY_INDEXES, time_step_total=len(q), fps=50, source_fps=fps,
                  source_npz=str(raw.with_suffix('.npz')), qpos_source=str(a.out/'adapted_qpos_50hz.csv'), prompt=spec['prompt'])
    if not converter.convert_single_motion('motion', motion, str(ref)):
        raise RuntimeError('SONIC export failed')
    for key, values in {'joint_pos': jp, 'joint_vel': np.gradient(jp,.02,axis=0),
                        'body_pos': bp, 'body_quat': bq,
                        'body_lin_vel': np.gradient(bp,.02,axis=0),
                        'body_ang_vel': converter.world_angular_velocity(bq)}.items():
        csv = ref/(key+'.csv')
        header = csv.open().readline().strip()
        np.savetxt(csv, values.reshape(len(t),-1), delimiter=',', fmt='%.12g', header=header, comments='')
    np.savetxt(a.out/'adapted_qpos_50hz.csv', output, delimiter=',', fmt='%.12g')
    for side in ('left', 'right'):
        parts = ['thumb_0', 'thumb_1', 'thumb_2', 'middle_0', 'middle_1', 'index_0', 'index_1']
        cols = [hn.index(side+'_hand_'+part+'_joint') for part in parts]
        np.savetxt(ref/(side+'_hand_pos.csv'), hand[:, cols], delimiter=',',
                   header=','.join(side+'_'+part for part in parts), comments='')
    quality = dict(frames=len(q), fps=50, max_ik_position_error_m=max(r['position_error_m'] for r in rows),
                   max_raw_to_target_correction_m=max(r['raw_to_target_correction_m'] for r in rows),
                   max_contact_position_correction_m=max_contact,
                   max_ik_orientation_error_rad=max(r['orientation_error_rad'] for r in rows),
                   source_last_timestamp_s=float(source_t[-1]), reference_last_timestamp_s=float(t[-1]),
                   source_padding_or_extrapolation=False,
                   model_generated_hands=False, online_feedback=False, adapter=cfg, calibration=calibration,
                   raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    quality['preserve_generated_body'] = bool(a.preserve_generated_body)
    quality['preserve_body_weight'] = float(a.preserve_body_weight)
    quality['body_preservation_policy'] = a.body_preservation_policy
    quality['body_preservation'] = body_policy
    quality['task_conditioned_clearance'] = bool(a.task_conditioned_clearance)
    quality['task_clearance_target_margin_m'] = float(a.task_clearance_target_margin)
    quality['spatial_clearance_m'] = float(spatial_clearance_m)
    quality['spatial_clearance_policy'] = spatial_clearance_info
    quality['geometry_approach_corridor'] = bool(a.geometry_approach_corridor)
    quality['geometry_approach_corridor_policy'] = approach_retraction_info
    quality['geometry_grasp_height'] = bool(a.geometry_grasp_height)
    quality['opposition_centering'] = bool(a.opposition_centering)
    quality['opposition_centering_anchor_frame'] = opposition_anchor
    quality['opposition_centering_policy'] = opposition_info
    quality['opposition_centering_max_correction_m'] = float(a.opposition_centering_max_correction_m)
    (a.out/'adaptation.json').write_text(json.dumps(quality, indent=2)+'\n')
    (a.out/'wrist_adaptation.json').write_text(json.dumps(rows)+'\n')
    (ref/'hand_plan.json').write_text(json.dumps(dict(generator='shared-hand-v1' if a.adapter == 'unified' else 'independent timed Dex3 schedule',
        preclose_s=cfg['hand_preclose_s'], preclose_fraction=cfg['hand_preclose_fraction'], close_end_s=cfg['close_end_s'],
        hand_generator=hand_info, hand_profile=a.hand_profile, adapter=a.adapter))+'\n')
    spec.update(source='Fresh ARDY with scene constraints, bounded offline wrist projection, independent timed hands; physical body controlled by SONIC',
                raw_source=str(raw.resolve()), generation_mode='fresh_segmented_ARDY', postprocessing=quality)
    (a.out/'setting.json').write_text(json.dumps(spec, indent=2)+'\n')
    print(json.dumps(quality, indent=2))


if __name__ == '__main__':
    main()
