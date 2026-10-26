"""Measure readiness envelope from physically successful historical rule source rollouts."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from scene import load_scene
from project_paths import SONIC_REPO,ROBOT_RELATIVE
root=Path(sys.argv[1]);out=Path(sys.argv[2]);m,d=load_scene(robot_path=SONIC_REPO/ROBOT_RELATIVE)
w=m.body('right_wrist_yaw_link').id;c=m.body('task_red_cube').id
R=np.array([[1,0,0],[0,0,1],[0,-1,0.]])
values=[];sources=[]
for ep in sorted((root/'episodes').iterdir())[:5]:
 p=ep/'source';report=json.loads((p/'report.json').read_text())
 if not report['physical_grasp_success']:continue
 q=np.load(p/'qpos.npy');t=np.load(p/'timestamps.npy');poses=[];errors=[];angles=[]
 for state in q:
  d.qpos[:]=state;mujoco.mj_forward(m,d);poses.append(d.xpos[w].copy());errors.append(np.linalg.norm(d.xpos[w]-(d.xpos[c]-R@np.array([.135,.065,0]))));angles.append(np.arccos(np.clip((np.trace(R.T@d.xmat[w].reshape(3,3))-1)/2,-1,1)))
 speeds=np.linalg.norm(np.gradient(np.asarray(poses),t,axis=0),axis=1);mask=(t>=7.8)&(t<=8.2)
 values.extend(np.c_[errors,angles,speeds][mask].tolist());sources.append(dict(path=str(p),qpos_sha256=hashlib.sha256((p/'qpos.npy').read_bytes()).hexdigest()))
a=np.asarray(values);limits=np.max(a,axis=0)*1.1
r=dict(method='Maximum observed in successful 7.8-8.2s approach/closure transition, plus explicit 10 percent measurement margin',sources=sources,samples=len(a),observed_max=np.max(a,axis=0).tolist(),thresholds=dict(zip(['position_m','orientation_rad','speed_mps'],limits.tolist())),geometry=dict(palm_offset_m=[.135,.065,0],origin='existing successful GraspReference geometry'),numeric_success_thresholds_changed=False)
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
