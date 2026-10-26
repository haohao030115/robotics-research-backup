"""Arm-only wrist IK and reach/grasp/lift reference planning.

This module only updates a separate MjData planning state; physical actuation
is owned by the whole-body controller and independent hand controller.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp


class GraspReference:
    def __init__(self, model, data, lift_height=.16):
        self.m, self.d = model, data
        self.lift_height = lift_height
        self.prelift_hold = 0.
        self.lift_duration = 2.
        self.plan = mujoco.MjData(model)
        self.plan.qpos[:] = data.qpos
        self.arm = [model.joint('right_'+n+'_joint').id for n in
                    ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']]
        self.aq = model.jnt_qposadr[self.arm]
        self.av = model.jnt_dofadr[self.arm]
        self.wrist = model.body('right_wrist_yaw_link').id
        self.cube = model.body('task_red_cube').id
        self.origin = data.xpos[self.cube].copy()
        self.hand = {n: model.joint('right_hand_'+n+'_joint').qposadr[0] for n in
                     ['thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1']}
        self.start = data.xpos[self.wrist].copy()
        self.rotation = np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
        self.target = self.origin - self.rotation @ np.array([.135,.065,0])
        self.thumb_pre=0.
        self.stage = ''
        self.natural_start = False
        self.start_rotation = data.xmat[self.wrist].reshape(3,3).copy()

    def ik(self, target):
        m,d = self.m,self.plan
        jp,jr = np.zeros((3,m.nv)),np.zeros((3,m.nv))
        for _ in range(60):
            mujoco.mj_forward(m,d)
            r = d.xmat[self.wrist].reshape(3,3)
            er = sum(np.cross(r[:,i],self.rotation[:,i]) for i in range(3))*.5
            error = np.r_[target-d.xpos[self.wrist], .3*er]
            if np.linalg.norm(error)<1e-5: break
            mujoco.mj_jacBody(m,d,jp,jr,self.wrist)
            jac = np.vstack([jp[:,self.av],.3*jr[:,self.av]])
            dq = jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-3,error)
            old=d.qpos[self.aq].copy()
            free=np.ones(len(self.arm),dtype=bool)
            for _ in range(len(self.arm)):
                blocked=((old<=m.jnt_range[self.arm,0]+1e-5)&(dq<0))|((old>=m.jnt_range[self.arm,1]-1e-5)&(dq>0))
                if not np.any(blocked & free):break
                free &= ~blocked
                jf=jac[:,free]
                dq[:]=0
                dq[free]=jf.T@np.linalg.solve(jf@jf.T+np.eye(6)*1e-3,error)
            old_cost=np.linalg.norm(target-d.xpos[self.wrist])**2+.045*np.linalg.norm(r-self.rotation)**2
            dq *= min(1., .1 / max(np.max(np.abs(dq)), 1e-12))
            for step in [1,.5,.25,.1]:
                d.qpos[self.aq] = np.clip(old+step*dq,m.jnt_range[self.arm,0],m.jnt_range[self.arm,1])
                mujoco.mj_forward(m,d)
                cost=np.linalg.norm(target-d.xpos[self.wrist])**2+.045*np.linalg.norm(d.xmat[self.wrist].reshape(3,3)-self.rotation)**2
                if cost<old_cost:break
            else:
                d.qpos[self.aq]=old
                break

    def control(self,t):
        def smooth(a,b,u):
            u=np.clip(u,0,1);return a+(b-a)*(u*u*(3-2*u))
        above = self.target + [0,0,.13]
        if self.natural_start:
            final_rotation=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
            if t < 5:
                # Lift outside the front edge before moving over the table.
                clearance=np.array([.06,self.start[1]-.03,above[2]+.04])
                if t < 3:
                    u=np.clip(t/3,0,1);u=u*u*(3-2*u)
                    self.rotation=Slerp([0,1],Rotation.from_matrix([self.start_rotation,final_rotation]))([u]).as_matrix()[0]
                    target=smooth(self.start,clearance,t/3)
                    self.stage='raise_outside_table'
                else:
                    self.rotation=final_rotation
                    target=smooth(clearance,above,(t-3)/2)
                    self.stage='cross_table_edge'
                self.ik(target)
                for address in self.hand.values():self.plan.qpos[address]=0
                return
            t-=3

        if t<2:
            stage='reach'; target=smooth(self.start,above,t/2); close=0
        elif t<4:
            stage='lower'; target=smooth(above,self.target,(t-2)/2); close=0
        elif t<6:
            stage='close'; target=self.target; close=smooth(0,1,(t-4)/2)
        elif t<6+self.prelift_hold:
            stage='closed_hold'; target=self.target; close=1
        elif t<6+self.prelift_hold+self.lift_duration:
            stage='lift'; target=smooth(self.target,self.target+[0,0,self.lift_height],
                                       (t-6-self.prelift_hold)/self.lift_duration); close=1
        else:
            stage='hold'; target=self.target+[0,0,self.lift_height]; close=1
        angle=smooth(0,-np.pi/2,t/2)
        c,s=np.cos(angle),np.sin(angle)
        self.rotation=np.array([[1,0,0],[0,c,-s],[0,s,c]])
        self.ik(target)
        for name, val in zip(self.hand,[0,-1,-1.25,1.2,1.2,1.2,1.2]):
            closure=self.thumb_pre+(1-self.thumb_pre)*close if name.startswith('thumb') else close
            self.plan.qpos[self.hand[name]]=closure*val
        self.stage=stage
