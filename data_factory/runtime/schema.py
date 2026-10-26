"""Fail closed before loading physics or policy. No producer-dependent semantics."""
import ast
import re
import numpy as np
from .common import SONIC, CPP, read, load_npz, require

UNITS = {'time_s': 's', 'body_ref_q': 'rad', 'body_ref_dq': 'rad/s',
         'root_ref_pos': 'm', 'root_ref_quat': 'unitless', 'hand_ref_q': 'rad'}

def joint_orders():
    tree = ast.parse((SONIC/'gear_sonic/envs/env_utils/joint_utils.py').read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('G1_ISAACLab_ORDER', 'G1_HAND_JOINTS'):
                    values[target.id] = ast.literal_eval(node.value)
    return values['G1_ISAACLab_ORDER'], values['G1_HAND_JOINTS']
def validate(folder):
    try:
        for filename in ('reference.npz','metadata.json','provenance.json','reference_validation.json'):
            require((folder/filename).is_file(), 'Missing '+filename)

        m, p, v = [read(folder/f) for f in (
            'metadata.json',
            'provenance.json',
            'reference_validation.json'
        )]
        require(all(isinstance(x, dict) for x in (m,p,v)),
                'JSON documents must be objects')

        # ------------------------------------------------------------
        # Backward-compatible metadata normalization
        # ------------------------------------------------------------

        m = dict(m)

        # producer_type:
        # new canonical format uses explicit producer_type.
        # Older Kimodo template used schema_name=kimodo_expert_reference.
        producer = m.get('producer_type')
        if producer is None:
            schema_name = m.get('schema_name')
            if schema_name == 'kimodo_expert_reference':
                producer = 'kimodo'
            elif schema_name == 'rule_expert_reference':
                producer = 'rule_based'
            elif schema_name == 'ardy_expert_reference':
                producer = 'ardy'

        require(producer in ('kimodo','rule_based','ardy'),
                'Unknown producer_type')
        m['producer_type'] = producer

        # quaternion convention alias
        quat_conv = m.get('quaternion_convention',
                          m.get('root_quat_order'))
        require(quat_conv == 'wxyz',
                'Expected explicit wxyz quaternion convention')
        m['quaternion_convention'] = quat_conv

        # root frame: old templates did not always declare it,
        # but their root reference is defined in world coordinates.
        root_pose_frame = m.get('root_pose_frame', 'world')
        require(root_pose_frame == 'world',
                'root_pose_frame must be world')
        m['root_pose_frame'] = root_pose_frame

        # unit compatibility
        units = m.get('units', {})
        if units != UNITS:
            legacy_units = {
                'time': 's',
                'position': 'm',
                'joint_position': 'rad',
                'joint_velocity': 'rad/s'
            }
            require(units == legacy_units,
                    'Explicit canonical units required')
            units = dict(UNITS)

        m['units'] = units

        require(m.get('reference_complete') is True,
                'reference_complete must be true')

        require(v.get('reference_schema_valid') is True,
                'Producer reference validation missing/failed')

        require(
            isinstance(m.get('episode_id'), str)
            and re.fullmatch(r'[A-Za-z0-9_-]+', m['episode_id']),
            'Invalid episode_id'
        )

        # ------------------------------------------------------------
        # The checks below remain STRICT
        # ------------------------------------------------------------

        bn, hn = joint_orders()

        require(
            m.get('body_joint_names') == bn and len(bn) == 29,
            'Canonical body joint order mismatch'
        )

        require(
            m.get('hand_joint_names') == hn and len(hn) == 14,
            'Canonical hand joint order mismatch'
        )

        a = load_npz(folder/'reference.npz')

        require(
            'time_s' in a and a['time_s'].ndim == 1,
            'Missing/invalid time_s'
        )

        t = a['time_s']
        n = len(t)

        require(n >= 7, 'At least seven reference frames required')

        shapes = {
            'time_s': (n,),
            'body_ref_q': (n,29),
            'body_ref_dq': (n,29),
            'root_ref_pos': (n,3),
            'root_ref_quat': (n,4),
            'hand_ref_q': (n,14)
        }

        for k, shape in shapes.items():
            require(
                k in a and a[k].shape == shape,
                'Invalid shape/missing '+k
            )

        for k, x in a.items():
            require(
                x.dtype.kind in 'fiu'
                and x.ndim > 0
                and len(x) == n
                and np.isfinite(x).all(),
                'Invalid dtype/length/nonfinite '+k
            )

        require(
            np.all(np.diff(t) > 0),
            'time_s must strictly increase'
        )

        hz = m.get('reference_hz')

        require(
            type(hz) in (int,float)
            and np.isfinite(hz)
            and hz == 50,
            'Canonical reference_hz must be 50'
        )

        require(
            abs(t[0]) < 1e-9
            and np.allclose(
                np.diff(t),
                1/hz,
                rtol=0,
                atol=1e-7
            ),
            'reference_hz disagrees with time_s or start != 0'
        )

        require(
            np.allclose(
                np.linalg.norm(a['root_ref_quat'], axis=1),
                1,
                rtol=0,
                atol=1e-5
            ),
            'Nonunit root quaternion'
        )

        require(
            np.allclose(
                a['hand_ref_q'][0],
                0,
                atol=1e-10,
                rtol=0
            ),
            'Canonical initial hand must be zero'
        )

        # Case compatibility:
        # canonical format uses case.block_xy;
        # older template may only contain cube_initial_pos.
        case = m.get('case')

        if not isinstance(case, dict):
            cube = m.get('cube_initial_pos')
            if isinstance(cube, (list, tuple)) and len(cube) >= 2:
                case = {
                    'id': m.get('case_name', 'legacy_case'),
                    'block_xy': [cube[0], cube[1]],
                    'block_yaw_deg': 0.0
                }
            else:
                case = {}

        require(
            isinstance(case.get('id'), str),
            'case.id required'
        )

        xy = np.asarray(case.get('block_xy'), dtype=float)

        require(
            xy.shape == (2,)
            and np.isfinite(xy).all()
            and np.isfinite(case.get('block_yaw_deg', 0)),
            'Invalid case XY/yaw'
        )

        m['case'] = case


        # Missing phases in legacy references simply means the whole
        # supplied reference is active.
        phases = m.get('phases', ['active'] * n)

        require(
            isinstance(phases, list)
            and len(phases) == n
            and all(
                x in ('active','reset','standby','inactive')
                for x in phases
            ),
            'Invalid phases'
        )

        m['phases'] = phases
        m['task_arm_defaulted'] = 'task_arm' not in m
        m['task_arm'] = m.get('task_arm','right')
        require(m['task_arm'] in ('left','right'),'task_arm must be left or right')

        return a, m, {
            'reference_schema_valid': True,
            'frames': n,
            'status': 'ACCEPT_REFERENCE'
        }

    except Exception as exc:
        raise ValueError('REJECT_REFERENCE: '+str(exc)) from exc