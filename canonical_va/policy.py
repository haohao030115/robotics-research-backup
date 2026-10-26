"""Release ONNX mode-0 preprocessing and decoder, synchronous and explicitly logged.

Equivalent formulas are read from the current native deployment, not old CSV row shifts.
"""
import ast
import re
import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation
from .common import BUNDLE, CPP, HASHES, digest, require


def parameters():
    src = (CPP/'include/policy_parameters.hpp').read_text()
    clean = re.sub(r'//[^\n]*','',src)
    constants = {}
    def evaluate(s):
        t = ast.parse(s.strip(), mode='eval')
        allowed = (ast.Expression,ast.Constant,ast.Name,ast.Load,ast.BinOp,ast.UnaryOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.USub,ast.UAdd)
        require(all(isinstance(n,allowed) for n in ast.walk(t)), 'Unexpected policy expression')
        return eval(compile(t,'<canonical policy parameters>','eval'),{'__builtins__':{}},constants)
    for name,value in re.findall(r'const double\s+(\w+)\s*=\s*([^;]+);',clean):
        constants[name] = evaluate(value)
    p = {}
    for name in ('isaaclab_to_mujoco','mujoco_to_isaaclab','g1_action_scale','default_angles','kps','kds'):
        expr = re.search(r'\b'+name+r'\s*=\s*\{([^}]+)\}',clean).group(1)
        p[name] = np.asarray([evaluate(x) for x in expr.split(',') if x.strip()])
        require(p[name].shape==(29,), 'Policy parameter size '+name)
    for name in ('isaaclab_to_mujoco','mujoco_to_isaaclab'):
        p[name] = p[name].astype(int)
    p['motor_names'] = re.findall(r'//\s*(\w+_joint)\b',src.split('g1_action_scale = {',1)[1].split('};',1)[0])
    return p


def session(name):
    require(digest(BUNDLE/name)==HASHES[name], 'Canonical bundle hash mismatch: '+name)
    opt=ort.SessionOptions(); opt.intra_op_num_threads=1; opt.inter_op_num_threads=1
    return ort.InferenceSession(str(BUNDLE/name),opt,providers=['CPUExecutionProvider'])


def infer(s,x):
    return s.run(None,{s.get_inputs()[0].name:np.asarray(x,dtype=np.float32).reshape(1,-1)})[0][0]


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]])


def heading(q):
    r=rotation(q).as_matrix(); angle=np.arctan2(r[...,1,0],r[...,0,0])
    return Rotation.from_rotvec(np.stack([np.zeros_like(angle),np.zeros_like(angle),angle],-1))


class Policy:
    def __init__(self, arrays):
        for name,h in HASHES.items():
            require(digest(BUNDLE/name)==h,'Canonical bundle hash mismatch: '+name)
        self.encoder=session('model_encoder.onnx'); self.decoder=session('model_decoder.onnx')
        require(self.encoder.get_inputs()[0].shape==[1,1762] and self.encoder.get_outputs()[0].shape==[1,64], 'Encoder contract changed')
        require(self.decoder.get_inputs()[0].shape==[1,994] and self.decoder.get_outputs()[0].shape==[1,29], 'Decoder contract changed')
        c=yaml.safe_load((BUNDLE/'observation_config.yaml').read_text())
        names=[v['name'] for v in c['encoder']['encoder_observations'] if v['enabled']]
        dimensions={n:int(d) for n,d in re.findall(r'\{"([^"]+)",\s*(\d+),',(CPP/'src/g1_deploy_onnx_ref.cpp').read_text())}
        self.offsets={}; offset=0
        for name in names:
            self.offsets[name]=(offset,dimensions[name]); offset+=dimensions[name]
        require(offset==1762,'Encoder layout changed')
        self.p=parameters(); self.a=arrays; self.previous=np.zeros(29)
        self.history={k:np.zeros((10,w)) for k,w in [('ang',3),('q',29),('dq',29),('action',29),('quat',4)]}

    def encoder_input(self, frame, base_quat):
        frames=np.minimum(frame+np.arange(10)*5,len(self.a['time_s'])-1)
        x=np.zeros(1762,np.float32)
        apply=heading(self.a['root_ref_quat'][0])*heading(self.a['root_ref_quat'][0]).inv()
        relative=rotation(base_quat).inv()*apply*rotation(self.a['root_ref_quat'][frames])
        values={'encoder_mode_4':[0,0,0,0], 'motion_joint_positions_10frame_step5':self.a['body_ref_q'][frames],
                'motion_joint_velocities_10frame_step5':self.a['body_ref_dq'][frames],
                'motion_anchor_orientation_10frame_step5':relative.as_matrix()[...,:,:2]}
        for name,value in values.items():
            off,width=self.offsets[name]; x[off:off+width]=np.asarray(value).reshape(-1)
        return x

    def decoder_input(self, token, motor_q, motor_dq, base_quat, local_angular_velocity):
        p=self.p
        # Native LowState / motor values cross float32 fields before converting back to double.
        values={'ang':np.asarray(local_angular_velocity,dtype=np.float32).astype(float),
                'q':(np.asarray(motor_q,dtype=np.float32).astype(float)-p['default_angles'])[p['mujoco_to_isaaclab']],
                'dq':np.asarray(motor_dq,dtype=np.float32).astype(float)[p['mujoco_to_isaaclab']],
                'action':self.previous.copy(), 'quat':np.asarray(base_quat,dtype=np.float32).astype(float)}
        for k,v in values.items():
            self.history[k][:-1]=self.history[k][1:]; self.history[k][-1]=v
        h=self.history; w=h['quat'][:,0,None]; v=-h['quat'][:,1:]; z=np.array([0.,0.,-1.])
        gravity=z*(2*w*w-1)+2*w*np.cross(v,z)+2*v*(v*z).sum(-1,keepdims=True)
        proprio=np.concatenate([h[k].reshape(-1) for k in ('ang','q','dq','action')]+[gravity.reshape(-1)]).astype(np.float32)
        return np.concatenate([token,proprio]).astype(np.float32),proprio

    def target(self,action):
        p=self.p
        return (p['default_angles']+p['g1_action_scale']*action[p['isaaclab_to_mujoco']]).astype(np.float32).astype(float)
