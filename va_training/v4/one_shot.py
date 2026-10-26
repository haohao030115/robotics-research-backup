"""Optional V4.0 execution-only one-shot hand constraint."""
import numpy as np

class OneShotHand:
    OPEN=0; CLOSING=1; HOLD=2
    def __init__(self, enabled=True, active_indices=None, confirm_replans=2,
                 velocity_limit=0.08, trigger_progress=0.05, hold_progress=0.95, fixed_scale=None, fixed_direction=None):
        self.enabled=bool(enabled);self.active=np.array(active_indices if active_indices is not None else range(7,14),dtype=int)
        self.confirm_replans=int(confirm_replans);self.velocity_limit=float(velocity_limit)
        self.trigger_progress=float(trigger_progress);self.hold_progress=float(hold_progress);self.fixed_scale=None if fixed_scale is None else np.asarray(fixed_scale,float);self.fixed_direction=None if fixed_direction is None else np.asarray(fixed_direction,float);self.reset()
    def reset(self):
        self.state=self.OPEN;self.confirmations=0;self.open_ref=None;self.direction=None;self.scale=None
        self.previous=None;self.trigger_tick=None;self.hold_tick=None;self.closure_stalled=False;self.events=[];self.reversal_count=0;self.boundary_jumps=[]
    def abort(self):self.reset()
    def release(self):self.reset()
    def prime(self, pred):
        """Prime from the complete V4 [50,14] plan; does not change targets."""
        if not self.enabled or self.open_ref is not None:return
        self._init(np.asarray(pred,float))
    def _init(self,pred):
        self.open_ref=pred[0].copy(); traj=pred[:,self.active]; delta=traj[-1]-traj[0]
        moving=np.abs(delta)>.02
        if not moving.any(): moving=np.ptp(traj,axis=0)>.02
        self.direction=np.sign(delta); self.direction[self.direction==0]=1
        self.scale=self.fixed_scale.copy() if self.fixed_scale is not None else np.maximum(np.max(np.abs(traj-traj[0]),axis=0),.05)
        if self.fixed_direction is not None:
            self.direction=self.fixed_direction.copy()
        self.moving=moving
    def _progress(self, target):
        d=(target[self.active]-self.open_ref[self.active])*self.direction
        return float(np.clip(np.mean(d/self.scale),0,1))
    def apply(self, absolute_tick, predicted_target, close_intent=None):
        x=np.asarray(predicted_target,float).copy()
        if x.shape!=(14,):raise ValueError('one-shot expects one absolute hand target [14]')
        if not self.enabled:return x
        if self.open_ref is None:
            self._init(x[None,:])
        raw_progress=self._progress(x)
        if self.state==self.OPEN:
            if close_intent is not None:
                self.confirmations = self.confirmations + 1 if float(close_intent)>0 else 0
            else:
                self.confirmations = self.confirmations + 1 if raw_progress>=self.trigger_progress else 0
            out=self.open_ref.copy()
            out[~np.isin(np.arange(14),self.active)]=x[~np.isin(np.arange(14),self.active)]
            if self.confirmations>=self.confirm_replans:
                self.state=self.CLOSING;self.trigger_tick=int(absolute_tick);self.events.append({'tick':int(absolute_tick),'event':'trigger','raw_progress':raw_progress})
            else:
                self.previous=out.copy();self.events.append({'tick':int(absolute_tick),'event':'open','raw_progress':raw_progress})
                return out
        if self.state==self.CLOSING:
            if self.previous is None:self.previous=self.open_ref.copy()
            delta=np.clip(x-self.previous,-self.velocity_limit,self.velocity_limit)
            signed=delta[self.active]*self.direction
            bad=signed<0;self.reversal_count+=int(bad.sum());delta[self.active[bad]]=0
            out=self.previous+delta
            out[~np.isin(np.arange(14),self.active)]=x[~np.isin(np.arange(14),self.active)]
            prog=self._progress(out)
            completion=((out[self.active]-self.open_ref[self.active])*self.direction/self.scale >= self.hold_progress)
            if bool(np.all(completion[self.moving])):
                self.state=self.HOLD;self.hold_tick=int(absolute_tick);self.events.append({'tick':int(absolute_tick),'event':'hold','progress':prog})
            elif np.max(np.abs(delta[self.active]))<1e-6:self.closure_stalled=True
            else:self.events.append({'tick':int(absolute_tick),'event':'closing','progress':prog})
            self.boundary_jumps.append(float(np.max(np.abs(out-self.previous))))
            self.previous=out.copy();return out
        # HOLD ignores every subsequent reopening prediction.
        self.events.append({'tick':int(absolute_tick),'event':'hold_keep','progress':self._progress(self.previous)})
        return self.previous.copy()
    def report(self):
        return dict(enabled=self.enabled,state=['OPEN','CLOSING','HOLD'][self.state],trigger_tick=self.trigger_tick,hold_tick=self.hold_tick,
            closure_stalled=self.closure_stalled,reversal_count=self.reversal_count,
            max_plan_boundary_jump=max(self.boundary_jumps) if self.boundary_jumps else 0.,events=self.events)
