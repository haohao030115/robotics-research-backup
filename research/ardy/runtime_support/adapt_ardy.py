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
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())['adapter']
    a.out.mkdir(parents=True, exist_ok=False)
    for name in ('setting.json', 'setting.npz', 'constraints.json', 'generation.json', 'pose_matrices.npz'):
        shutil.copy2(a.source/name, a.out/name)
    spec = json.loads((a.source/'setting.json').read_text())
    model, data, _, _ = initialize(spec['case']['id'])
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
    plan = GraspReference(model, data)
    armcols = [names.index(model.joint(int(j)).name)+7 for j in plan.arm]
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
        targetp[2] += cfg['clearance_m']*smooth(t[k], *cfg['clearance_blend_s'])*(1-smooth(t[k], *cfg['clearance_decay_s']))
        entry = float(smooth(t[k], 0, cfg['entry_blend_s']))
        targetp = start+entry*(targetp-start)
        plan.rotation = Slerp([0, 1], Rotation.from_matrix([startr, targetr]))([entry]).as_matrix()[0]
        if k:
            plan.ik(targetp)
        mujoco.mj_forward(model, plan.plan)
        output[k, armcols] = plan.plan.qpos[plan.aq]
        rows.append(dict(frame=k, time_s=float(t[k]), raw_wrist_world_pos=rawp.tolist(),
                         planned_wrist_world_pos=goalp[k].tolist(), target_wrist_world_pos=targetp.tolist(),
                         position_error_m=float(np.linalg.norm(plan.plan.xpos[plan.wrist]-targetp)),
                         orientation_error_rad=float((Rotation.from_matrix(plan.plan.xmat[plan.wrist].reshape(3,3))*Rotation.from_matrix(plan.rotation).inv()).magnitude()),
                         contact_position_correction_m=contact_correction,
                         contact_rotation_correction_rad=contact_rotation_correction,
                         raw_to_target_correction_m=float(np.linalg.norm(targetp-rawp)),
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
    hand = setting['hand_ref_q'].copy()
    early = cfg['hand_preclose_fraction']*smooth(t, *cfg['hand_preclose_s'])
    late = cfg['hand_preclose_fraction'] + (1-cfg['hand_preclose_fraction'])*smooth(t, cfg['hand_preclose_s'][1], cfg['close_end_s'])
    fraction = np.where(t < cfg['hand_preclose_s'][1], early, late)
    hn = spec['hand_joint_order']
    for j, name in enumerate(hn):
        hand[:, j] = fraction*setting['hand_ref_q'][-1, j] if name.startswith('right_') else 0
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
    (a.out/'adaptation.json').write_text(json.dumps(quality, indent=2)+'\n')
    (a.out/'wrist_adaptation.json').write_text(json.dumps(rows)+'\n')
    (ref/'hand_plan.json').write_text(json.dumps(dict(generator='independent timed Dex3 schedule',
        preclose_s=cfg['hand_preclose_s'], preclose_fraction=cfg['hand_preclose_fraction'], close_end_s=cfg['close_end_s']))+'\n')
    spec.update(source='Fresh ARDY with scene constraints, bounded offline wrist projection, independent timed hands; physical body controlled by SONIC',
                raw_source=str(raw.resolve()), generation_mode='fresh_segmented_ARDY', postprocessing=quality)
    (a.out/'setting.json').write_text(json.dumps(spec, indent=2)+'\n')
    print(json.dumps(quality, indent=2))


if __name__ == '__main__':
    main()
