from enum import IntEnum
import numpy as np

class HandState(IntEnum): OPEN=0; CLOSING=1; HOLD=2

class HandContinuityLatch:
    """Execution-only latch. It never reads privileged object state."""
    def __init__(self, open_reference, confirm_count=2, close_threshold=.5,
                 velocity_limit=.08, closing_mask=None):
        self.open_reference=np.asarray(open_reference,float).copy()
        self.confirm_count=int(confirm_count); self.close_threshold=float(close_threshold)
        self.velocity_limit=float(velocity_limit)
        self.mask=np.ones(self.open_reference.shape[-1],bool) if closing_mask is None else np.asarray(closing_mask,bool)
        self.reset()
    def reset(self): self.state=HandState.OPEN; self.confirmations=0; self.previous=None
    def abort(self): self.reset()
    def release(self): self.reset()
    def step(self, trajectory, ready_logit, phase, current=None):
        x=np.asarray(trajectory,float); target=x[0].copy() if x.ndim==2 else x.copy()
        ready=float(ready_logit)>0
        if self.state==HandState.OPEN:
            self.confirmations=self.confirmations+1 if ready and int(phase)>=1 else 0
            if self.confirmations>=self.confirm_count: self.state=HandState.CLOSING
            else: target=self.open_reference.copy()
        if self.state==HandState.CLOSING:
            if self.previous is not None:
                delta=np.clip(target-self.previous,-self.velocity_limit,self.velocity_limit)
                # Closing joints never reverse toward opening while latched.
                delta[self.mask]=np.maximum(delta[self.mask],0.0)
                target=self.previous+delta
            if np.max(np.abs(target-self.open_reference))>.02 and (int(phase)>=2 or ready): self.state=HandState.HOLD
        elif self.state==HandState.HOLD:
            if self.previous is not None:
                delta=np.clip(target-self.previous,-self.velocity_limit,self.velocity_limit)
                delta[self.mask]=np.maximum(delta[self.mask],0.0)
                target=self.previous+delta
        self.previous=target.copy(); return target.copy(), self.state.name
