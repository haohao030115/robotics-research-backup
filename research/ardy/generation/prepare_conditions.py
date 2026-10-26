"""Build ARDY conditions from the team's immutable tabletop setting (container)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

# This duplicate generation entry point lives at ``research/ardy/generation``;
# its workspace is three parents above the file.  Keeping the fallback aligned
# with the qualification/overlap entry points lets the validated container run
# without requiring an otherwise redundant RESEARCH_ROOT override.
WORKSPACE = Path(os.environ.get('RESEARCH_ROOT', Path(__file__).resolve().parents[3])).resolve()
PROJECT = Path(os.environ.get('DL_GROUP2_ROOT', WORKSPACE / 'dl-group2')).resolve()
SONIC = Path(os.environ.get('SONIC_REPO', WORKSPACE / 'GR00T-WholeBodyControl')).resolve()
ARDY = Path(os.environ.get('ARDY_REPO', WORKSPACE / 'workspace/fuyuhan/ardy')).resolve()
CONVERTER = ARDY.parent/'GR00T-WholeBodyControl/gear_sonic_deploy/reference/convert_ardy.py'
os.environ['SONIC_REPO'] = str(SONIC)
sys.path[:0] = [str(PROJECT / 'scripts'), str(SONIC), str(ARDY), str(Path(__file__).resolve().parents[1] / 'adaptation')]
import project_paths
project_paths.default_robot_xml=lambda:SONIC/'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml'
from scene import load_scene
from cases import reset_case, read_cases
from episode_schema import neutral_arms, NEUTRAL_POSE
from expert_trajectory import orders, JointMap, CRITERION
from planned_grasp_reference import GraspReference

ROBOT_REL = 'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml'
ROBOT = SONIC / ROBOT_REL

def initialize(case_id="center"):
    pin = 'canonical explicit robot XML; hash recorded'
    model, data = load_scene(supported=False, robot_path=ROBOT)
    case = next(c for c in read_cases()['cases'] if c['id'] == case_id)
    reset_case(model, data, case)
    for side in ('left', 'right'):
        for part in ('hip_pitch', 'knee', 'ankle_pitch'):
            data.qpos[model.joint(f'{side}_{part}_joint').qposadr[0]] = NEUTRAL_POSE[part]
    neutral_arms(model, data)
    mujoco.mj_forward(model, data)
    feet = [i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name in
            ('left_ankle_roll_link', 'right_ankle_roll_link') and model.geom_contype[i]]
    root = int(model.joint('floating_base_joint').qposadr[0])
    data.qpos[root + 2] -= min(data.geom_xpos[i, 2] - model.geom_size[i, 0] for i in feet) + .0001
    mujoco.mj_forward(model, data)
    assert model.neq == 0 and np.allclose(model.opt.gravity, [0, 0, -9.81])
    return model, data, case, pin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--case',default='center')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--prompt',required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from ardy.skeleton import G1Skeleton34
    from ardy.exports.mujoco import MujocoQposConverter
    from ardy.constraints import FullBodyConstraintSet, Root2DConstraintSet, EndEffectorConstraintSet, save_constraints_lst
    m, d, case, pin = initialize(args.case)
    skel = G1Skeleton34()
    converter = MujocoQposConverter(skel)
    names = [j.get('name') for j in ET.parse(converter.xml_path).findall('.//worldbody//joint') if j.get('type') != 'free']
    jq = np.array([m.joint(n).qposadr[0] for n in names])
    bn, hn, _ = orders()
    hand = JointMap(m, hn, 14)
    M = converter.mujoco_to_ardy_matrix.numpy()
    axes = converter._mujoco_joint_axis_values_ardy_space.numpy()
    ix = converter._mujoco_indices_to_ardy_indices.numpy()
    offsets = converter._rot_offsets_f2q.numpy()
    def dictionary(q):
        local = np.tile(np.eye(3), (34, 1, 1))
        local[ix] = np.einsum('nij,njk->nik', offsets[ix].transpose(0, 2, 1), Rotation.from_rotvec(axes*q[jq, None]).as_matrix())
        local[0] = M @ Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix() @ M.T
        return torch.tensor(local, dtype=torch.float32), torch.tensor(M@q[:3], dtype=torch.float32)
    plan = GraspReference(m, d, lift_height=.18)
    plan.natural_start = True
    poses, href, errors, targets = [], [], [], []
    dense_pos, dense_quat = [], []
    duration = 16.2
    # IK is used only for sparse ARDY conditioning and a separate finger schedule.
    for frame in range(int(round(duration * 50)) + 1):
        t = frame / 50
        if t >= 1:
            plan.control(t-1)
        mujoco.mj_forward(m, plan.plan)
        dense_pos.append(plan.plan.xpos[plan.wrist].copy())
        dense_quat.append(plan.plan.xquat[plan.wrist].copy())
        if frame % 10 == 0 or frame == duration*50:
            poses.append(plan.plan.qpos.copy())
            targets.append(plan.plan.xpos[plan.wrist].copy())
        if 8 <= t <= 10:
            errors.append(float(np.linalg.norm(plan.plan.xpos[plan.wrist]-plan.target)))
        href.append(plan.plan.qpos[hand.qa].copy())
    if max(errors) > .02:
        raise RuntimeError(f'IK target error exceeds 2 cm: {max(errors)}')
    frames = np.arange(0, 406, 5)
    local, roots = zip(*(dictionary(q) for q in poses))
    local, roots = torch.stack(local), torch.stack(roots)
    reconstructed = converter.dict_to_qpos({'local_rot_mats': local[None], 'root_positions': roots[None]}, device='cpu')[0]
    source = np.stack(poses)[:, np.r_[np.arange(7), jq]]
    roundtrip = float(np.max(np.abs(reconstructed-source)))
    if roundtrip > 1e-4:
        raise RuntimeError(f'Coordinate roundtrip failed: {roundtrip}')
    gr, gp, _ = skel.fk(local, roots)
    constraints = [FullBodyConstraintSet(skel, torch.tensor([0, 25]), gp[[0, 5]], gr[[0, 5]]),
                   Root2DConstraintSet(skel, torch.arange(406), torch.zeros((406, 2))),
                   EndEffectorConstraintSet(skel, torch.tensor(frames), gp, gr, None,
                       joint_names=['RightHand', 'LeftHand', 'LeftFoot', 'RightFoot', 'Hips'])]
    save_constraints_lst(str(args.out/'constraints.json'), constraints)
    np.savez(args.out/'pose_matrices.npz', local_rot_mats=local.numpy(), root_positions=roots.numpy(), canonical_qpos=source, frame_indices=frames)
    np.savez(args.out/'setting.npz', initial_qpos=d.qpos, initial_qvel=d.qvel,
             hand_ref_q=href[:810], conditioning_qpos=source, condition_frames=frames, wrist_targets=targets,
             planned_wrist_pos=dense_pos[:810], planned_wrist_quat=dense_quat[:810])
    spec = dict(scene=str(PROJECT/'scenes/tabletop.xml'), scene_sha256=hashlib.sha256((PROJECT/'scenes/tabletop.xml').read_bytes()).hexdigest(),
                robot=str(ROBOT), robot_authority=pin, robot_sha256=hashlib.sha256(ROBOT.read_bytes()).hexdigest(),
                neutral=NEUTRAL_POSE, initial_qpos=d.qpos.tolist(), case=case,
                cube_initial_position=d.xpos[m.body('task_red_cube').id].tolist(),
                cube_side_m=(m.geom_size[m.geom('task_cube_geom').id]*2).tolist(),
                root_position=d.qpos[:3].tolist(), body_joint_order=bn, hand_joint_order=hn,
                duration=duration, seed=args.seed, gpu=int(os.environ.get("ARDY_PHYSICAL_GPU", "7")), close=[8,10], lift=[10,12], hold=[12,16.2],
                ik_grasp_error_max_m=max(errors), coordinate_roundtrip_error=roundtrip, criterion=CRITERION,
                prompt=args.prompt)
    (args.out/'setting.json').write_text(json.dumps(spec, indent=2)+'\n')
    print(json.dumps(spec, indent=2))

if __name__ == '__main__':
    main()
