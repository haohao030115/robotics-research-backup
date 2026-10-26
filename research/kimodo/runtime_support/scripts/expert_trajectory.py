"""Named joint mappings, expert episode IO and physical success evaluation."""
import ast
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np

from project_paths import SONIC_REPO

CRITERION = {
    "minimum_lift_m": .10, "minimum_continuous_hold_s": 2.,
    "contacts": "positive normal-force right thumb AND index/middle; no cube/table contact",
    "physical_only": True, "cube_attachment_allowed": False,
    "cube_teleport_after_reset_allowed": False,
    "require_complete_finite_synchronized_sonic_hand_rollout": True,
    "require_no_fall": True,
}


def literal_assignment(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Cannot find literal {name} in {path}")


def deployment_order(repo):
    names = literal_assignment(repo / "gear_sonic/envs/env_utils/joint_utils.py", "G1_ISAACLab_ORDER")
    source = (repo / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp").read_text()
    def permutation(name):
        match = re.search(r"\b" + name + r"\s*=\s*\{([^}]+)\}", source)
        if not match:
            raise ValueError(f"Missing deployment mapping: {name}")
        result = [int(x) for x in match[1].split(",")]
        if sorted(result) != list(range(29)):
            raise ValueError(f"Invalid permutation: {name}")
        return result
    to_mj = permutation("isaaclab_to_mujoco")
    to_il = permutation("mujoco_to_isaaclab")
    # The action-scale comments independently identify the 29 hardware/MuJoCo names.
    scale = source.split("g1_action_scale = {", 1)[1].split("};", 1)[0]
    mj_names = re.findall(r"//\s*(\w+_joint)\b", scale)
    if len(names) != 29 or len(set(names)) != 29 or len(mj_names) != 29:
        raise ValueError("Expected exactly 29 unique joint names")
    for i, name in enumerate(mj_names):
        if names[to_mj[i]] != name or to_il[to_mj[i]] != i:
            raise ValueError(f"Python/C++ disagreement at motor {i}: {name}")
    return names


def orders(repo=SONIC_REPO):
    body = deployment_order(repo)
    hand = literal_assignment(repo / "gear_sonic/envs/env_utils/joint_utils.py", "G1_HAND_JOINTS")
    src = (repo / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp").read_text()
    motor = re.findall(r"//\s*(\w+_joint)\b", src.split("g1_action_scale = {", 1)[1].split("};", 1)[0])
    assert len(body) == len(set(body)) == len(motor) == len(set(motor)) == 29
    assert len(hand) == len(set(hand)) == 14 and set(body).isdisjoint(hand)
    return body, hand, motor


class JointMap:
    def __init__(self, model, names, expected):
        if len(names) != expected or len(set(names)) != expected:
            raise ValueError(f"Expected {expected} distinct joint names")
        self.names = list(names)
        joints, actuators = [], []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Missing joint {name}")
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Joint {name} is not a scalar hinge")
            aids = [a for a in range(model.nu)
                    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == jid]
            if len(aids) != 1:
                raise ValueError(f"Expected one named-joint actuator for {name}, got {aids}")
            joints.append(jid)
            actuators.append(aids[0])
        self.jids = np.asarray(joints, dtype=int)
        self.qa = model.jnt_qposadr[self.jids]
        self.va = model.jnt_dofadr[self.jids]
        self.aids = np.asarray(actuators, dtype=int)
        if len(set(self.qa)) != expected or len(set(self.va)) != expected or len(set(self.aids)) != expected:
            raise ValueError("Mapping is not one-to-one")

    def describe(self):
        return [{"name": n, "joint_id": int(j), "qpos_address": int(q), "qvel_address": int(v), "actuator_id": int(a)}
                for n, j, q, v, a in zip(self.names, self.jids, self.qa, self.va, self.aids)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_arrays(folder, arrays):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(folder / f"{name}.npy", np.asarray(value))


def load_reference(folder):
    folder = Path(folder)
    keys = ("timestamps", "body_ref_q", "body_ref_dq", "hand_ref_q", "hand_ref_dq", "root_ref_pos", "root_ref_quat")
    arrays = {k: np.load(folder / f"{k}.npy", allow_pickle=False) for k in keys}
    t = arrays["timestamps"]
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not np.allclose(np.diff(t), .02, atol=1e-8):
        raise ValueError("Expected finite monotonic 50 Hz reference timestamps")
    for key, width in [("body_ref_q", 29), ("body_ref_dq", 29), ("hand_ref_q", 14), ("hand_ref_dq", 14),
                       ("root_ref_pos", 3), ("root_ref_quat", 4)]:
        if arrays[key].shape != (len(t), width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key}: {arrays[key].shape}")
    if not np.allclose(np.linalg.norm(arrays["root_ref_quat"], axis=1), 1, atol=1e-5):
        raise ValueError("Root quaternions must be unit wxyz")
    metadata = json.loads((folder / "metadata.json").read_text())
    body, hand, _ = orders()
    if metadata["body_joint_order"] != body or metadata["hand_joint_order"] != hand:
        raise ValueError("Episode joint order disagrees with deployment")
    return arrays, metadata


def robot_state(model, data, body, hand):
    pelvis = model.body("pelvis").id
    cube = model.body("task_red_cube").id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jp, jr, pelvis)
    return {
        "root_lin_vel": jp @ data.qvel, "root_ang_vel": jr @ data.qvel,
        "body_q": data.qpos[body.qa].copy(), "body_dq": data.qvel[body.va].copy(),
        "hand_q": data.qpos[hand.qa].copy(), "hand_dq": data.qvel[hand.va].copy(),
        "root_pos": data.xpos[pelvis].copy(), "root_quat": data.xquat[pelvis].copy(),
        "cube_pos": data.xpos[cube].copy(), "cube_quat": data.xquat[cube].copy(),
        "eef_pos": data.xpos[model.body("right_wrist_yaw_link").id].copy(),
        "eef_quat": data.xquat[model.body("right_wrist_yaw_link").id].copy(),
        "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy(),
    }


class PhysicalGraspEvaluator:
    def __init__(self, model, data, task_arm="right"):
        if task_arm not in ("left", "right"): raise ValueError("Invalid task_arm")
        self.task_arm = task_arm
        self.model = model
        self.cube = model.body("task_red_cube").id
        self.cube_geom = model.geom("task_cube_geom").id
        self.table = model.body("task_table").id
        self.initial_z = float(data.xpos[self.cube, 2])
        self.hold = self.max_hold = self.max_lift = 0.
        self.table_steps = self.fall_steps = 0
        self.table_bodies = set()
        self.last_contacts = []
        self.self_contact_steps = self.left_arm_environment_steps = 0
        self.inactive_arm_environment_steps = 0
        self.self_contact_pairs = set()
        self.body_names = [model.body(i).name or "" for i in range(model.nbody)]

    def step(self, data):
        force = np.zeros(6)
        contacts = []
        touching_table = False
        touching_self = touching_left_environment = touching_inactive_environment = False
        for i, contact in enumerate(data.contact):
            b1, b2 = self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]
            n1,n2 = self.body_names[b1],self.body_names[b2]
            if b1 > 0 and b2 > 0 and contact.dist <= 0:
                if not n1.startswith('task_') and not n2.startswith('task_'):
                    touching_self = True
                    self.self_contact_pairs.add(tuple(sorted((n1,n2))))
                for robot,environment in [(n1,n2),(n2,n1)]:
                    if robot.startswith('left_') and any(part in robot for part in ('shoulder','elbow','wrist','hand')) and environment.startswith('task_'):
                        touching_left_environment = True
                    inactive = 'left_' if self.task_arm == 'right' else 'right_'
                    if robot.startswith(inactive) and any(part in robot for part in ('shoulder','elbow','wrist','hand')) and environment.startswith('task_'):
                        touching_inactive_environment = True
            other = b2 if b1 == self.table else b1 if b2 == self.table else -1
            if other > 0 and other != self.cube:
                touching_table = True
                self.table_bodies.add(self.model.body(other).name)
            other_geom = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1 if contact.geom2 == self.cube_geom else -1
            if other_geom >= 0:
                mujoco.mj_contactForce(self.model, data, i, force)
                name = self.model.body(self.model.geom_bodyid[other_geom]).name
                contacts.append({"body": name, "normal_force_n": float(abs(force[0]))})
        self.table_steps += int(touching_table)
        self.self_contact_steps += int(touching_self)
        self.left_arm_environment_steps += int(touching_left_environment)
        self.inactive_arm_environment_steps += int(touching_inactive_environment)
        lift = float(data.xpos[self.cube, 2] - self.initial_z)
        self.max_lift = max(self.max_lift, lift)
        effective = [c["body"] for c in contacts if c["normal_force_n"] > 1e-4]
        thumb = any(f"{self.task_arm}_hand_thumb" in name for name in effective)
        finger = any(f"{self.task_arm}_hand_index" in name or f"{self.task_arm}_hand_middle" in name for name in effective)
        # Any detected cube/table contact invalidates the hold, even if force is tiny.
        cube_on_table = any(c["body"] == "task_table" for c in contacts)
        valid = lift >= .10 and thumb and finger and not cube_on_table
        self.hold = self.hold + self.model.opt.timestep if valid else 0.
        self.max_hold = max(self.max_hold, self.hold)
        self.fall_steps += int(data.xpos[self.model.body("pelvis").id, 2] < .2)
        self.last_contacts = contacts
        return {"lift_m": lift, "opposing_contact": thumb and finger, "cube_table_contact": cube_on_table,
                "valid_hold": valid, "robot_self_contact": touching_self, "left_arm_environment_contact": touching_left_environment, "robot_table_contact": touching_table, "hold_seconds": self.hold}

    def report(self, data):
        return {"physical_grasp_success": self.hold >= 2., "max_lift_m": self.max_lift,
                "final_lift_m": float(data.xpos[self.cube, 2] - self.initial_z),
                "max_continuous_hold_seconds": self.max_hold, "final_continuous_hold_seconds": self.hold,
                "robot_table_contact_steps": self.table_steps, "robot_table_contact_bodies": sorted(self.table_bodies),
                "robot_self_contact_steps": self.self_contact_steps, "robot_self_contact_pairs": sorted(self.self_contact_pairs),
                "left_arm_environment_contact_steps": self.left_arm_environment_steps,
                "inactive_arm_environment_contact_steps": self.inactive_arm_environment_steps,
                "fall_steps": self.fall_steps, "final_cube_contacts": self.last_contacts,
                "task_arm": self.task_arm,
                "criterion": dict(CRITERION, contacts=f"positive normal-force {self.task_arm} thumb AND index/middle; no cube/table contact"), "cube_attached": False, "cube_teleported_during_rollout": False}
