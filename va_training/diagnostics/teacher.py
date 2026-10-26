"""Frozen release G1 encoder prototype; exact mode-filtered native preprocessing."""
from pathlib import Path
import re
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation


class CorrectiveTeacher:
    def __init__(self,sonic,episode):
        import json,yaml
        metadata=json.loads((episode.path/'metadata.json').read_text())
        self.reference=episode.path/'candidates'/metadata['selected_candidate']
        self.q=np.loadtxt(self.reference/'joint_pos.csv',delimiter=',',skiprows=1,ndmin=2)
        self.dq=np.loadtxt(self.reference/'joint_vel.csv',delimiter=',',skiprows=1,ndmin=2)
        self.quat=np.loadtxt(self.reference/'body_quat.csv',delimiter=',',skiprows=1,ndmin=2)[:,:4]
        self.config=yaml.safe_load(Path(sonic['observation_config']).read_text())['encoder']
        names=[v['name'] for v in self.config['encoder_observations'] if v['enabled']]
        source=(Path(sonic['repository'])/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp').read_text()
        dimensions={name:int(dim) for name,dim in re.findall(r'\{"([^"]+)",\s*(\d+),',source)}
        self.offsets={}; offset=0
        for name in names: self.offsets[name]=(offset,dimensions[name]); offset+=dimensions[name]
        if offset!=1762: raise ValueError('Release encoder layout mismatch')
        required=next(m['required_observations'] for m in self.config['encoder_modes'] if m['mode_id']==0)
        if set(required)!={'encoder_mode_4','motion_joint_positions_10frame_step5','motion_joint_velocities_10frame_step5','motion_anchor_orientation_10frame_step5'}:
            raise ValueError('Unsupported corrective teacher mode')
        self.mode=0
        opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
        self.session=ort.InferenceSession(sonic['encoder'],opt,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name

    @staticmethod
    def rotation(q): return Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]])

    @staticmethod
    def heading(q):
        r=CorrectiveTeacher.rotation(q).as_matrix(); angle=np.arctan2(r[...,1,0],r[...,0,0])
        return Rotation.from_rotvec(np.stack([np.zeros_like(angle),np.zeros_like(angle),angle],-1))

    def token(self,frame,measured_base_quat,initial_base_quat,delta_heading=0.):
        frames=np.minimum(frame+np.arange(10)*5,len(self.q)-1)
        data=np.zeros(1762,dtype=np.float32)
        def put(name,value):
            offset,width=self.offsets[name];a=np.asarray(value).reshape(-1)
            if a.size!=width: raise ValueError('Encoder field shape mismatch')
            data[offset:offset+width]=a
        put('encoder_mode_4',[self.mode,0,0,0])
        put('motion_joint_positions_10frame_step5',self.q[frames])
        put('motion_joint_velocities_10frame_step5',self.dq[frames])
        apply=Rotation.from_rotvec([0,0,delta_heading])*self.heading(initial_base_quat)*self.heading(self.quat[0]).inv()
        relative=self.rotation(measured_base_quat).inv()*apply*self.rotation(self.quat[frames])
        put('motion_anchor_orientation_10frame_step5',relative.as_matrix()[...,:,:2].reshape(-1))
        return self.session.run(None,{self.input:data[None]})[0][0]
