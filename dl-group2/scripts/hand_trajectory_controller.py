"""Independent, name-mapped 14-DoF hand player. No body joint writes."""
import mujoco
import numpy as np

from expert_trajectory import JointMap, orders


class HandTrajectoryController:
    def __init__(self, model, timestamps, hand_ref_q, hand_ref_dq=None, kp=12., kd=.3):
        _, names, _ = orders()
        self.mapping = JointMap(model, names, 14)
        self.model = model
        self.timestamps = np.asarray(timestamps, float)
        self.q = np.asarray(hand_ref_q, float)
        self.dq = np.zeros_like(self.q) if hand_ref_dq is None else np.asarray(hand_ref_dq, float)
        if self.timestamps.ndim != 1 or len(self.timestamps) < 2 or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Hand timestamps must be strictly increasing")
        if self.q.shape != (len(self.timestamps), 14) or self.dq.shape != self.q.shape:
            raise ValueError("Hand schema must be [T,14]")
        if not all(np.isfinite(a).all() for a in (self.timestamps, self.q, self.dq)):
            raise ValueError("Non-finite hand trajectory")
        self.kp, self.kd = kp, kd
        self.types = []
        for aid in self.mapping.aids:
            gain = model.actuator_gainprm[aid, 0]
            bias = model.actuator_biasprm[aid]
            if not np.allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]):
                raise ValueError("Only scalar unit-gear hand actuators are supported")
            if model.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE:
                raise ValueError("Unsupported dynamic hand actuator")
            if model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE and np.isclose(bias[1], -gain) and gain > 0:
                self.types.append("position")
            elif model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_NONE and np.isclose(gain, 1):
                self.types.append("torque_pd")
            else:
                raise ValueError("Unsupported hand actuator; inspect gain/bias before controlling")

    def target(self, reference_time):
        # Clamp endpoints and interpolate on the shared reference timestamp axis.
        q = np.array([np.interp(reference_time, self.timestamps, self.q[:, j]) for j in range(14)])
        dq = np.array([np.interp(reference_time, self.timestamps, self.dq[:, j]) for j in range(14)])
        return q, dq

    def apply(self, data, reference_time):
        qref, _ = self.target(reference_time)
        tau = self.kp * (qref - data.qpos[self.mapping.qa]) - self.kd * data.qvel[self.mapping.va]
        commands = np.array([qref[j] if kind == "position" else tau[j] for j, kind in enumerate(self.types)])
        for j, (jid, aid, kind) in enumerate(zip(self.mapping.jids, self.mapping.aids, self.types)):
            if kind == "torque_pd" and self.model.jnt_actfrclimited[jid]:
                commands[j] = np.clip(commands[j], *self.model.jnt_actfrcrange[jid])
            if self.model.actuator_ctrllimited[aid]:
                commands[j] = np.clip(commands[j], *self.model.actuator_ctrlrange[aid])
        data.ctrl[self.mapping.aids] = commands
        return qref, commands
