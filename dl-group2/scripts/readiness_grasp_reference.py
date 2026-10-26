"""Arm-selectable offline rule planner; closure latches measured geometric readiness."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from grasp_reference import GraspReference

class ReadinessGraspReference(GraspReference):
    def __init__(
        self,
        model,
        data,
        task_arm,
        thresholds,
        approach_scale=1.0,
        lift=.18,
        wrist_target_offset_y=0.0,
    ):
        super().__init__(model,data,lift)
        if task_arm not in ('left','right'): raise ValueError(task_arm)
        self.task_arm=task_arm; self.thresholds=thresholds; self.approach_scale=approach_scale
        self.arm=[model.joint(task_arm+'_'+n+'_joint').id for n in
                  ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']]
        self.aq=model.jnt_qposadr[self.arm];self.av=model.jnt_dofadr[self.arm]
        self.wrist=model.body(task_arm+'_wrist_yaw_link').id
        self.hand={n:model.joint(task_arm+'_hand_'+n+'_joint').qposadr[0] for n in
                   ['thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1']}
        self.start=data.xpos[self.wrist].copy();self.start_rotation=data.xmat[self.wrist].reshape(3,3).copy()
        self.sign=1 if task_arm=='right' else -1
        self.final_rotation=np.array([[1,0,0],[0,0,self.sign],[0,-self.sign,0.]])
        self.offset=np.array([.135,.065*self.sign,0.])
        self.wrist_target_offset=np.array([0.,wrist_target_offset_y,0.])
        self.target=self.origin-self.final_rotation@self.offset+self.wrist_target_offset
        self.closed_at=None;self.ready_since=None;self.events=[];self.previous=None
        self.rotation=self.final_rotation.copy();self.last_metrics={}

    @staticmethod
    def smooth(a,b,u):
        u=np.clip(u,0,1);return np.asarray(a)+(np.asarray(b)-a)*(u*u*(3-2*u))

    def control(self,t):
        d=self.d; above=self.target+[0,0,.13];s=t/self.approach_scale
        if s<3:
            clearance=np.array([.06,self.start[1]-.03*self.sign,above[2]+.04])
            u=np.clip(s/3,0,1);u=u*u*(3-2*u)
            self.rotation=Slerp([0,1],Rotation.from_matrix([self.start_rotation,self.final_rotation]))([u]).as_matrix()[0]
            target=self.smooth(self.start,clearance,s/3);stage='raise_outside_table'
        elif s<5:
            clearance=np.array([.06,self.start[1]-.03*self.sign,above[2]+.04])
            self.rotation=self.final_rotation;target=self.smooth(clearance,above,(s-3)/2);stage='cross_table'
        elif s<7:
            self.rotation=self.final_rotation;target=self.smooth(above,self.target,(s-5)/2);stage='approach'
        else:
            self.rotation=self.final_rotation;target=self.target.copy();stage='wait_ready'
        wrist=d.xpos[self.wrist].copy();cube=d.xpos[self.cube].copy()
        desired=cube-self.final_rotation@self.offset+self.wrist_target_offset
        error=float(np.linalg.norm(wrist-desired))
        angle=float(np.arccos(np.clip((np.trace(self.final_rotation.T@d.xmat[self.wrist].reshape(3,3))-1)/2,-1,1)))
        speed=float(np.linalg.norm(wrist-self.previous[1])/(t-self.previous[0])) if self.previous is not None and t>self.previous[0] else 0.
        self.previous=(t,wrist);self.last_metrics=dict(time_s=float(t+1),wrist_error_m=error,orientation_error_rad=angle,wrist_speed_mps=speed)
        ready=s>=7 and error<=self.thresholds['position_m'] and angle<=self.thresholds['orientation_rad'] and speed<=self.thresholds['speed_mps']
        if self.closed_at is None:
            self.ready_since=t if ready and self.ready_since is None else self.ready_since if ready else None
            if self.ready_since is not None and t-self.ready_since>=.10-1e-8:
                self.closed_at=t;self.events.append(dict(self.last_metrics,event='closure_trigger',continuous_ready_s=.10))
        close=0.
        if self.closed_at is not None:
            elapsed=t-self.closed_at;close=float(self.smooth(0.,1.,elapsed/2.))
            target=self.target.copy();stage='close'
            if elapsed>=2:
                target=self.smooth(self.target,self.target+[0,0,self.lift_height],(elapsed-2)/2)
                stage='lift' if elapsed<4 else 'hold'
        self.ik(target)
        values=[0,-1*self.sign,-1.25*self.sign,1.2*self.sign,1.2*self.sign,1.2*self.sign,1.2*self.sign]
        for name,value in zip(self.hand,values):self.plan.qpos[self.hand[name]]=close*value
        self.stage=stage
