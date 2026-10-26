"""Additive Vision-Action episode metadata, read from the initialized scene."""
import json
from project_paths import CONFIGS

NEUTRAL_POSE = json.loads((CONFIGS / "neutral_standing.json").read_text())
SCHEMA_VERSION = 3
INSTRUCTION = "Pick up the red cube and lift it from the table."
CAMERA_ROLE = {"head_rgb": "policy_observation", "wrist_rgb": "policy_observation_optional", "task_overview": "evaluation_only"}

def dataset_fields(model, data, case, episode_id, random_seed):
    cube = model.body('task_red_cube').id
    geom = model.geom('task_cube_geom').id
    table = model.geom('task_table_top').id
    return dict(schema_version=SCHEMA_VERSION, instruction=INSTRUCTION,
        camera_role=dict(CAMERA_ROLE), task_config=dict(
            object_name=model.body(cube).name,
            cube_initial_position=data.xpos[cube].tolist(),
            cube_initial_quaternion_wxyz=data.xquat[cube].tolist(),
            cube_size=(2 * model.geom_size[geom]).tolist(),
            cube_size_convention='full side lengths in metres',
            cube_mass=float(model.body_mass[cube]),
            table_height=float(data.geom_xpos[table, 2] + model.geom_size[table, 2]),
            random_seed=random_seed, case_id=case['id'], episode_id=episode_id))

def neutral_arms(model, data):
    """Model-specific mirrored pose: wrists beside hips, fingers pointing down."""
    values = {}
    angles = NEUTRAL_POSE['arm_joints']
    for side, sign in [('left', 1), ('right', -1)]:
        for part, configured in angles.items():
            part = 'shoulder_roll' if part == 'shoulder_roll_outward' else part
            value = sign * configured if part == 'shoulder_roll' else configured
            name = f'{side}_{part}_joint'
            joint = model.joint(name)
            assert model.jnt_range[joint.id, 0] <= value <= model.jnt_range[joint.id, 1]
            data.qpos[joint.qposadr[0]] = value
            values[name] = value
    for side in ('left', 'right'):
        for finger in ('index_0','index_1','middle_0','middle_1','thumb_0','thumb_1','thumb_2'):
            data.qpos[model.joint(f'{side}_hand_{finger}_joint').qposadr[0]] = 0.
    return values


def validate_dataset_metadata(meta):
    """Reject inconsistent labels before publication or training-manifest inclusion."""
    import numpy as np
    if meta.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Expected episode schema_version=3')
    if meta.get('instruction') != INSTRUCTION or meta.get('camera_role') != CAMERA_ROLE:
        raise ValueError('Instruction/camera roles do not match the shared schema')
    task=meta['task_config']
    required={'object_name','cube_initial_position','cube_initial_quaternion_wxyz','cube_size',
              'cube_mass','table_height','random_seed','case_id','episode_id'}
    if not required <= task.keys():raise ValueError(f'Missing task configuration: {required-task.keys()}')
    for key,width in [('cube_initial_position',3),('cube_initial_quaternion_wxyz',4),('cube_size',3)]:
        value=np.asarray(task[key],float)
        if value.shape!=(width,) or not np.isfinite(value).all():raise ValueError(key)
    if not np.isclose(np.linalg.norm(task['cube_initial_quaternion_wxyz']),1):raise ValueError('Nonunit cube quaternion')
    if min(task['cube_size'])<=0 or task['cube_mass']<=0:raise ValueError('Invalid object geometry/mass')
    if task['case_id']!=meta['case']['id'] or not np.allclose(task['cube_initial_position'][:2],meta['case']['block_xy']):
        raise ValueError('Task configuration does not match the reset case')
    if len(meta['body_joint_order'])!=29 or len(meta['hand_joint_order'])!=14:
        raise ValueError('Expected full body29 and both hands14')
    seed=task['random_seed']
    if seed is not None and (not isinstance(seed,int) or isinstance(seed,bool)):raise ValueError('Invalid random seed')


def validate_images(folder, reference_timestamps):
    """RGB timestamps must index the same clock as reference and actual arrays."""
    import numpy as np
    path=folder/'vision/images.npz'
    if not path.exists():path=folder/'images.npz'
    if not path.exists():raise ValueError(f'Missing RGB images: {folder}')
    with np.load(path) as images:
        frames=images['reference_frame']
        if frames.ndim!=1 or not np.issubdtype(frames.dtype,np.integer):raise ValueError('Invalid RGB frame indexes')
        if len(frames)==0 or not np.all(np.diff(frames)>0):raise ValueError('Empty/nonmonotonic RGB indexes')
        if frames[0]<0 or frames[-1]>=len(reference_timestamps):raise ValueError('RGB index out of range')
        if not np.array_equal(images['timestamps'],reference_timestamps[frames]):raise ValueError('RGB/reference timestamps differ')
        for camera in CAMERA_ROLE:
            pixels=images[camera]
            if pixels.shape!=(len(frames),240,320,3) or pixels.dtype!=np.uint8:raise ValueError(f'Invalid RGB array: {camera}')
        return {'path':str(path),'frames':len(frames),'first_reference_frame':int(frames[0]),
                'last_reference_frame':int(frames[-1]),'timestamps_match_reference':True}
