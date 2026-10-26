"""Fixed canonical physical runtime and exact snapshots."""
import sys
import xml.etree.ElementTree as ET
import numpy as np
from .common import ROOT, PROJECT, ROBOT, CPP, BUNDLE, HASHES, dump, read, digest, require, snapshot_hash

sys.path.insert(0,str(PROJECT/'scripts'))
import mujoco
from scene import load_scene
from cases import reset_case
from expert_trajectory import JointMap, PhysicalGraspEvaluator, CRITERION
from hand_trajectory_controller import HandTrajectoryController
from .schema import joint_orders

CONTRACT = ROOT/'canonical_va/runtime_contract.json'


def source_paths():
    files=[ROBOT, PROJECT/'scenes/tabletop.xml', PROJECT/'configs/grasp_cases.json',
           CPP/'include/policy_parameters.hpp', CPP/'src/g1_deploy_onnx_ref.cpp']
    files += [PROJECT/'scripts'/n for n in ('scene.py','cases.py','expert_trajectory.py','hand_trajectory_controller.py','run_sonic_grasp.py')]
    files += [BUNDLE/n for n in HASHES]
    xml=ET.parse(ROBOT).getroot(); compiler=xml.find('compiler')
    for kind,attr in [('mesh','meshdir'),('texture','texturedir')]:
        for e in xml.findall('asset/'+kind):
            if e.get('file'):
                files.append((ROBOT.parent/compiler.get(attr,'')/e.get('file')).resolve())
    return sorted(set(files))


def camera_config(model):
    cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'head_rgb')
    require(cid>=0,'CAMERA_MISMATCH: missing head_rgb')
    return dict(name='head_rgb',parent_body=model.body(model.cam_bodyid[cid]).name,
                local_position=model.cam_pos[cid].tolist(),local_quaternion_wxyz=model.cam_quat[cid].tolist(),
                fov_y=float(model.cam_fovy[cid]),resolution=[320,240],image_hz=10,
                source_definition='dl-group2/scripts/scene.py',source_xml='dl-group2/scenes/tabletop.xml')


def validate_camera(actual, expected):
    require(actual==expected,'CAMERA_MISMATCH: actual camera differs from frozen runtime contract')


class Runtime:
    def __init__(self,a,m,p):
        contract=read(CONTRACT)
        require(mujoco.__version__==contract['mujoco_version'],'MuJoCo version mismatch')
        self.model,self.data=load_scene(supported=False,robot_path=ROBOT,scene_path=PROJECT/'scenes/tabletop.xml')
        self.model.opt.timestep=.005
        model,data=self.model,self.data
        self.camera=camera_config(model)
        validate_camera(self.camera,contract['camera'])
        for rel,h in contract['source_hashes'].items():
            require(digest(ROOT/rel)==h,'Canonical runtime source changed: '+rel)
        require((model.nq,model.nv,model.nu,model.neq)==(57,55,43,0),'Robot dimension/weld mismatch')
        require(model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST,'Integrator mismatch')
        require(np.array_equal(model.opt.gravity,[0,0,-9.81]),'Gravity mismatch')
        g=model.geom('task_cube_geom').id; b=model.body('task_red_cube').id; t=model.geom('task_table_top').id
        require(np.allclose(model.geom_size[g],[.035]*3) and model.body_mass[b]==.1,'Cube geometry/mass mismatch')
        require(np.allclose(model.geom_friction[g],[.95,.01,.001]) and model.geom_contype[g]==1 and model.geom_conaffinity[g]==1,'Cube contact mismatch')
        require(np.allclose(model.geom_size[t],[.4,.5,.03]) and np.allclose(model.geom_pos[t],[0,0,.72]),'Table mismatch')
        reset_case(model,data,m['case'])
        bn,hn=joint_orders(); self.body=JointMap(model,bn,29); self.hand=JointMap(model,hn,14); self.motor=JointMap(model,p['motor_names'],29)
        root=model.joint('floating_base_joint'); self.rqa=int(root.qposadr[0]); self.rva=int(root.dofadr[0])
        data.qpos[self.body.qa]=a['body_ref_q'][0]; data.qpos[self.hand.qa]=a['hand_ref_q'][0]
        data.qpos[self.rqa:self.rqa+3]=a['root_ref_pos'][0]; data.qpos[self.rqa+3:self.rqa+7]=a['root_ref_quat'][0]
        mujoco.mj_forward(model,data)
        self.hand_controller=HandTrajectoryController(model,a['time_s'],a['hand_ref_q'])
        require(all(x=='torque_pd' for x in self.hand_controller.types),'Hand must use independent torque PD')
        self.evaluator=PhysicalGraspEvaluator(model,data); self.p=p
        self.state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
        self.renderer=mujoco.Renderer(model,height=self.camera['resolution'][1],width=self.camera['resolution'][0])
        self.physics=[]
        self.config=dict(mujoco_version=mujoco.__version__,dimensions=[57,55,43,0],physics_hz=200,control_hz=50,
                         hand_pd={'kp':12.,'kd':.3},criterion=CRITERION,initial_qpos=data.qpos.tolist(),initial_qvel=data.qvel.tolist(),
                         initialization='reset_case then reference frame 0; zero velocities; no simulated warmup',
                         transport='synchronous in-process command mailbox; no DDS/native executable',
                         scheduling='OBSERVE ENCODE DECODE PUBLISH RECEIPT then four physics steps',
                         inference_provider='ONNX Runtime CPU float32 canonical release',
                         source_hashes=contract['source_hashes'],camera=self.camera)

    def integration_state(self):
        s=np.empty(mujoco.mj_stateSize(self.model,self.state_spec)); mujoco.mj_getState(self.model,self.data,s,self.state_spec)
        return s

    def snapshot(self, identity, step):
        m,d=self.model,self.data
        contacts=[]
        for i,c in enumerate(d.contact):
            f=np.empty(6); mujoco.mj_contactForce(m,d,i,f)
            contacts.append(dict(geom1=int(c.geom1),geom2=int(c.geom2),position=c.pos.tolist(),frame=c.frame.tolist(),distance=float(c.dist),force=f.tolist()))
        def pose(name):
            b=m.body(name).id; return np.r_[d.xpos[b],d.xquat[b]]
        return dict(snapshot_id=identity,sim_step_id=step,simulation_time=float(d.time),full_qpos=d.qpos.copy(),full_qvel=d.qvel.copy(),
                    integration_state=self.integration_state(),body_q=d.qpos[self.body.qa].copy(),body_dq=d.qvel[self.body.va].copy(),
                    hand_q=d.qpos[self.hand.qa].copy(),hand_dq=d.qvel[self.hand.va].copy(),root_pos=d.qpos[self.rqa:self.rqa+3].copy(),
                    root_quat=d.qpos[self.rqa+3:self.rqa+7].copy(),root_velocity=d.qvel[self.rva:self.rva+6].copy(),
                    cube_pose=pose('task_red_cube'),right_wrist_pose=pose('right_wrist_yaw_link'),left_wrist_pose=pose('left_wrist_yaw_link'),
                    contacts=__import__('json').dumps(contacts),current_applied_ctrl=d.ctrl.copy(),
                    snapshot_sha256=snapshot_hash(d.qpos,d.qvel,d.ctrl,d.time))

    def render(self):
        self.renderer.update_scene(self.data,camera='head_rgb')
        return self.renderer.render().copy()

    def apply(self,target,reference_time):
        m,d,p=self.model,self.data,self.p
        kp=p['kps'].astype(np.float32).astype(float); kd=p['kds'].astype(np.float32).astype(float)
        tau=kp*(target-d.qpos[self.motor.qa])-kd*d.qvel[self.motor.va]
        for i,j in enumerate(self.motor.jids):
            if m.jnt_actfrclimited[j]: tau[i]=np.clip(tau[i],*m.jnt_actfrcrange[j])
        d.ctrl[self.motor.aids]=tau
        self.hand_controller.apply(d,reference_time)
        require(not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied),'Unexpected external force')

    def step(self):
        mujoco.mj_step(self.model,self.data); mujoco.mj_forward(self.model,self.data)
        require(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all(),'Nonfinite physics')
        metrics=self.evaluator.step(self.data)
        self.physics.append(metrics)
        return metrics

    def physical_report(self,complete):
        r=self.evaluator.report(self.data)
        return dict(physical_success=bool(r['physical_grasp_success'] and not r['fall_steps'] and complete),
                    cube_max_lift=r['max_lift_m'],qualifying_hold_duration=r['max_continuous_hold_seconds'],
                    opposing_contact_valid=bool(self.physics and self.physics[-1]['opposing_contact']),
                    cube_table_contact_during_hold=any(x['cube_table_contact'] for x in self.physics if x['valid_hold']),
                    fall_detected=bool(r['fall_steps']),replay_complete=complete,canonical_evaluator=r)
