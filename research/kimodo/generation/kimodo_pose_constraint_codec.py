"""Matrix-preserving condition codec for the canonical-to-Kimodo adapter.

Use load_pose_constraints(), not Kimodo's CLI JSON loader, for this schema.
The original loader uses a biased axis-angle conversion (axis/(angle+1e-6));
this sidecar preserves verified matrices and builds the existing constraint types.
Frames/FPS are explicit method inputs, not a frozen VA data contract.
"""
import json
from pathlib import Path

import numpy as np

from kimodo_condition_coordinates import qpos_to_motion_dict_exact

SCHEMA = 'kimodo_canonical_pose_constraints_v1'


def make_pose_constraint_record(converter, qpos, frame_indices, source_fps,
                                fullbody_rows=(0,), purpose='task_conditions'):
    q = np.asarray(qpos, dtype=np.float32)
    frames = np.asarray(frame_indices)
    rows = np.asarray(fullbody_rows)
    if (frames.ndim != 1 or frames.dtype.kind not in 'iu' or len(frames) != len(q)
            or np.any(frames < 0) or np.any(np.diff(frames.astype(np.int64)) <= 0)):
        raise ValueError('Frame indices must be strictly increasing nonnegative integers')
    if (rows.ndim != 1 or len(rows) == 0 or rows.dtype.kind not in 'iu'
            or np.any(rows < 0) or np.any(rows >= len(q)) or len(np.unique(rows)) != len(rows)):
        raise ValueError('Fullbody rows must select unique valid pose rows')
    motion = qpos_to_motion_dict_exact(converter, q, source_fps)
    return {'schema':SCHEMA, 'purpose':purpose,
            'coordinate_contract':'canonical_raw_qpos_to_verified_model_local_matrices_to_model_FK',
            'direct_world_coordinate_copy':False,
            'source_fps':float(source_fps), 'frame_indices':frames.tolist(),
            'fullbody_rows':rows.tolist(),
            'joint_names':['RightHand','LeftHand','LeftFoot','RightFoot','Hips'],
            'canonical_qpos':q.tolist(),
            'local_rot_mats':motion['local_rot_mats'].detach().cpu().numpy().tolist(),
            'root_positions':motion['root_positions'].detach().cpu().numpy().tolist()}


def save_pose_constraint_record(path, record):
    if record.get('schema') != SCHEMA:
        raise ValueError('Unexpected constraint schema')
    with Path(path).open('x') as stream:
        json.dump(record, stream, allow_nan=False)
        stream.write('\n')


def load_pose_constraints(path, converter):
    """Validate matrices and canonical roundtrip before constructing model constraints."""
    import torch
    from kimodo.constraints import FullBodyConstraintSet, EndEffectorConstraintSet

    record = json.loads(Path(path).read_text())
    if (record.get('schema') != SCHEMA or record.get('direct_world_coordinate_copy') is not False
            or record.get('coordinate_contract') !=
            'canonical_raw_qpos_to_verified_model_local_matrices_to_model_FK'):
        raise ValueError('Unknown/unsafe coordinate contract')
    q = np.asarray(record['canonical_qpos'], dtype=np.float32)
    frames = np.asarray(record['frame_indices'])
    rows = np.asarray(record['fullbody_rows'])
    if (q.ndim != 2 or q.shape[1] != 36 or len(q) < 8 or not np.isfinite(q).all()
            or frames.shape != (len(q),) or frames.dtype.kind not in 'iu'
            or np.any(frames < 0) or np.any(np.diff(frames.astype(np.int64)) <= 0)
            or rows.ndim != 1 or len(rows) == 0 or rows.dtype.kind not in 'iu'
            or np.any(rows < 0) or np.any(rows >= len(q)) or len(np.unique(rows)) != len(rows)):
        raise ValueError('Invalid pose/frame/row arrays')
    if not np.isfinite(record['source_fps']) or record['source_fps'] <= 0:
        raise ValueError('FPS must be finite and positive')
    if record['joint_names'] != ['RightHand','LeftHand','LeftFoot','RightFoot','Hips']:
        raise ValueError('Unexpected end-effector selection')
    local = torch.tensor(record['local_rot_mats'], dtype=torch.float32)
    root = torch.tensor(record['root_positions'], dtype=torch.float32)
    if local.shape != (len(q),converter.skeleton.nbjoints,3,3) or root.shape != (len(q),3):
        raise ValueError('Invalid matrix/root dimensions')
    if not torch.isfinite(local).all() or not torch.isfinite(root).all():
        raise ValueError('Nonfinite pose matrices')
    check = local.detach().cpu().numpy()
    if (not np.allclose(check @ check.swapaxes(-1,-2), np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(check), 1., atol=1e-6, rtol=0)):
        raise ValueError('Local matrices must be proper rotations')
    restored = converter.to_qpos(local[None],root[None],root_quat_w_first=True,
                                 mujoco_rest_zero=False).detach().cpu().numpy()[0]
    restored = restored.copy()
    restored[:,3:7] *= np.where((restored[:,3:7]*q[:,3:7]).sum(axis=-1)<0,-1.,1.)[:,None]
    err = float(np.max(abs(restored-q)))
    if not np.isfinite(err) or err > 1e-6:
        raise ValueError(f'Saved canonical pose contract failed: {err:g}')
    sk = converter.skeleton
    global_rot, positions, _ = sk.fk(local, root)
    indices = torch.as_tensor(frames, dtype=torch.long)
    selection = torch.as_tensor(rows, dtype=torch.long)
    constraints = [FullBodyConstraintSet(sk,indices[selection],positions[selection],global_rot[selection]),
                   EndEffectorConstraintSet(sk,indices,positions,global_rot,None,
                                            joint_names=record['joint_names'])]
    return constraints, {'saved_matrix_roundtrip_max_error':err,
                         'world_xyz_identity_mapping':False,
                         'coordinate_contract':record['coordinate_contract']}
