"""Estimate offline wrist tracking bias from this ARDY scheme's own replay."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_conditions import initialize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Choose a new calibration output')
    spec = json.loads((a.run/'source_setting.json').read_text())
    m, d, _, _ = initialize(spec['case']['id'])
    wrist = m.body('right_wrist_yaw_link').id
    qa = np.array([m.joint(n).qposadr[0] for n in spec['body_joint_order']])
    actual = np.load(a.run/'simulation.npz')
    ref = np.load(a.run/'input_reference.npz')
    frame = actual['reference_frame']
    t = ref['timestamps'][frame]
    pos, rot = [], []
    for i, k in enumerate(frame):
        d.qpos[:] = actual['qpos'][i]
        mujoco.mj_forward(m, d)
        ap, ar = d.xpos[wrist].copy(), d.xmat[wrist].reshape(3, 3).copy()
        d.qpos[:] = spec['initial_qpos']
        d.qpos[:3] = ref['root_ref_pos'][k]
        d.qpos[3:7] = ref['root_ref_quat'][k]
        d.qpos[qa] = ref['body_ref_q'][k]
        mujoco.mj_forward(m, d)
        pos.append(ap-d.xpos[wrist])
        rot.append((Rotation.from_matrix(ar)*Rotation.from_matrix(d.xmat[wrist].reshape(3, 3)).inv()).as_rotvec())
    pos, rot = np.array(pos), np.array(rot)
    result = dict(kind='own_ARDY_replay_tracking_compensation', source_run=str(a.run.resolve()), case=spec['case'],
                  windows_s=dict(close=[7.5, 8.5], hold=[12, 16.18]), source_hashes={}, unclipped={})
    for phase, (start, end) in result['windows_s'].items():
        mask = (t >= start) & (t <= end)
        if not mask.any():
            raise ValueError('Incomplete calibration window '+phase)
        bp, br = np.median(pos[mask], axis=0), np.median(rot[mask], axis=0)
        result['unclipped'][phase] = dict(position_bias_m=bp.tolist(), rotation_bias_rad=br.tolist())
        bp *= min(1., .06/max(np.linalg.norm(bp), 1e-12))
        br *= min(1., .30/max(np.linalg.norm(br), 1e-12))
        result[phase+'_position_bias_m'] = bp.tolist()
        result[phase+'_rotation_bias_rad'] = br.tolist()
    for name in ('simulation.npz', 'input_reference.npz', 'provenance.json'):
        f = a.run/name
        result['source_hashes'][str(f.resolve())] = hashlib.sha256(f.read_bytes()).hexdigest()
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
