"""Match consumed tokens and hand references on one native control timeline."""
import numpy as np


class TokenHandoff:
    def __init__(self):
        self.pending = None
        self.current = None
        self.applied = []

    def propose(self, prediction):
        if self.pending is not None:
            raise ValueError('Only one outstanding native token handoff is allowed')
        for key, width in [('token', 64), ('hand', 14)]:
            a = np.asarray(prediction[key])
            if a.shape != (width,) or not np.isfinite(a).all():
                raise ValueError('Invalid token/hand prediction')
        self.pending = prediction

    def consume(self, control_index, token):
        token = np.asarray(token)
        if token.shape != (64,) or not np.isfinite(token).all():
            raise ValueError('Invalid consumed token')
        if self.pending is not None and control_index >= self.pending['execution_index'] and np.max(abs(token-self.pending['token'])) <= 1e-6:
            self.current = self.pending
            self.current['applied_index'] = control_index
            self.applied.append(self.current)
            self.pending = None
        if self.current is None or np.max(abs(token-self.current['token'])) > 1e-6:
            raise RuntimeError('Native token has no matching synchronized hand reference')
        return self.current['hand']
