"""Snapshot/camera reader only. No SONIC inference, reset policy, PD or stepping."""
import numpy as np
from .common import ROOT, PROJECT, ROBOT, read, digest, require
from .runtime import Runtime, CONTRACT, camera_config, validate_camera, mujoco, load_scene, JointMap
from .schema import joint_orders


class SnapshotReader:
    integration_state=Runtime.integration_state
    snapshot=Runtime.snapshot
    render=Runtime.render

    def __init__(self,model=None,data=None,render=False):
        if model is None:
            model,data=load_scene(supported=False,robot_path=ROBOT,scene_path=PROJECT/'scenes/tabletop.xml')
            model.opt.timestep=.005
        self.model,self.data=model,data
        c=read(CONTRACT)
        require(mujoco.__version__==c['mujoco_version'],'MuJoCo version changed')
        self.camera=camera_config(model); validate_camera(self.camera,c['camera'])
        from .versioning import check_policy
        self.source_version=check_policy(emit=False)
        require((model.nq,model.nv,model.nu,model.neq)==(57,55,43,0),'Robot dimensions/weld mismatch')
        require(model.opt.timestep==.005,'Physics dt mismatch')
        require(model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST and np.array_equal(model.opt.gravity,[0,0,-9.81]),'Physics settings mismatch')
        bn,hn=joint_orders(); self.body=JointMap(model,bn,29); self.hand=JointMap(model,hn,14)
        root=model.joint('floating_base_joint'); self.rqa=int(root.qposadr[0]);self.rva=int(root.dofadr[0])
        self.state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.renderer=mujoco.Renderer(model,height=240,width=320) if render else None

    def restore(self,state):
        mujoco.mj_setState(self.model,self.data,state,self.state_spec)
        mujoco.mj_forward(self.model,self.data)

    def close(self):
        if self.renderer: self.renderer.close()
