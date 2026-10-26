"""Exact frozen decoder reconstruction from native pre-action logs."""
import ast
import re
from pathlib import Path
import numpy as np
import onnxruntime as ort


def cpp_parameters(repository):
    source = (Path(repository)/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp').read_text()
    clean = re.sub(r'//[^\n]*', '', source)
    constants = {}
    def evaluate(expression):
        tree = ast.parse(expression.strip(), mode='eval')
        allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
        if any(not isinstance(n, allowed) for n in ast.walk(tree)): raise ValueError('Unexpected C++ numeric expression')
        return eval(compile(tree, '<policy constants>', 'eval'), {'__builtins__': {}}, constants)
    for name, expression in re.findall(r'const double\s+(\w+)\s*=\s*([^;]+);', clean):
        constants[name] = evaluate(expression)
    result = {}
    for name in ('isaaclab_to_mujoco', 'mujoco_to_isaaclab', 'g1_action_scale', 'default_angles'):
        expression = re.search(r'\b'+name+r'\s*=\s*\{([^}]+)\}', clean).group(1)
        result[name] = np.array([evaluate(v) for v in expression.split(',') if v.strip()])
    result['isaaclab_to_mujoco'] = result['isaaclab_to_mujoco'].astype(int)
    result['mujoco_to_isaaclab'] = result['mujoco_to_isaaclab'].astype(int)
    result['motor_names'] = re.findall(r'//\s*(\w+_joint)\b', source.split('g1_action_scale = {',1)[1].split('};',1)[0])
    return result


def native_gravity(q):
    # Match quat_rotate_d(conjugate(q), [0,0,-1]), INCLUDING zero padded quaternion.
    q = np.asarray(q); w = q[...,0]; v = -q[...,1:]
    z = np.array([0.,0.,-1.])
    return z*(2*w[...,None]**2-1)+2*w[...,None]*np.cross(v,z)+2*v*(v*z).sum(-1,keepdims=True)


class ReleaseDecoder:
    def __init__(self, config):
        opt = ort.SessionOptions(); opt.intra_op_num_threads = 1; opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(config['decoder'], opt, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.parameters = cpp_parameters(config['repository']); self.cache = {}

    def logs(self, episode):
        key = str(episode.path)
        if key not in self.cache:
            p = Path(episode.provenance['token_label_source']).parent
            arrays = {name: np.loadtxt(p/(name+'.csv'), delimiter=',', skiprows=1, ndmin=2)[:,5:]
                      for name in ['q','dq','base_quat','base_ang_vel','action']}
            # CSV logging reverses decoder state conversion: measured hardware
            # order with standing offsets restored. Undo before padding history.
            permutation = self.parameters['mujoco_to_isaaclab']
            arrays['q'] = (arrays['q']-self.parameters['default_angles'])[:,permutation]
            arrays['dq'] = arrays['dq'][:,permutation]
            self.cache[key] = arrays
        return self.cache[key]

    def observation(self, episode, frame, token):
        arrays = self.logs(episode)
        index = int(episode.control_metadata[frame,0])
        indexes = np.arange(index-9,index+1); valid = indexes >= 0
        def history(name):
            a = arrays[name]; result = np.zeros((10,a.shape[1]), dtype=np.float64)
            result[valid] = a[indexes[valid]]
            return result
        values = [np.asarray(token).reshape(-1), history('base_ang_vel').reshape(-1),
                  history('q').reshape(-1), history('dq').reshape(-1), history('action').reshape(-1),
                  native_gravity(history('base_quat')).reshape(-1)]
        observation = np.concatenate(values).astype(np.float32)[None]
        if observation.shape != (1,994): raise ValueError('Release decoder shape mismatch')
        return observation

    def decode(self, episode, frame, token):
        return self.session.run(None, {self.input_name:self.observation(episode,frame,token)})[0][0]

    def target(self, raw):
        p = self.parameters
        return np.asarray(raw)[...,p['isaaclab_to_mujoco']]*p['g1_action_scale']+p['default_angles']
