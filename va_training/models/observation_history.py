"""Rolling, named live observations matching V2's saved post-step samples."""
from collections import deque
import numpy as np
from datasets.tokens import STATE_WIDTHS, projected_gravity


class ObservationHistory:
    def __init__(self, length):
        if length < 1:
            raise ValueError('History length must be positive')
        self.length = length
        self.rows = deque(maxlen=length)
        self.last_index = None

    def append(self, index, state):
        if self.last_index is not None and index != self.last_index+1:
            raise ValueError('History requires contiguous control observations')
        state = {k: np.asarray(v, dtype=np.float32).copy() for k, v in state.items()}
        state['projected_gravity'] = projected_gravity(state['base_quat']).astype(np.float32)
        if set(state) != set(STATE_WIDTHS) or any(state[k].shape != (v,) or not np.isfinite(state[k]).all() for k, v in STATE_WIDTHS.items()):
            raise ValueError('Invalid named live state')
        self.rows.append(state); self.last_index = index

    def snapshot(self):
        if not self.rows:
            raise ValueError('Empty history')
        missing = self.length-len(self.rows)
        rows = [self.rows[0]]*missing+list(self.rows)
        return {k: np.stack([row[k] for row in rows]) for k in STATE_WIDTHS}, np.arange(self.length) >= missing


def simulator_state(runtime, previous_action):
    """Measured q/dq and free-base local angular velocity; no expert future access."""
    data = runtime.data
    quat = data.qpos[runtime.root_qa+3:runtime.root_qa+7].copy()
    return {'body_q': data.qpos[runtime.body.qa].copy(), 'body_dq': data.qvel[runtime.body.va].copy(),
            'hand_q': data.qpos[runtime.hand.qa].copy(), 'hand_dq': data.qvel[runtime.hand.va].copy(),
            'base_quat': quat, 'base_ang_vel': data.qvel[runtime.root_va+3:runtime.root_va+6].copy(),
            'previous_action': np.asarray(previous_action).copy()}
