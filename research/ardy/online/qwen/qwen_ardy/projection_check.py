import argparse
from pathlib import Path
import mujoco, numpy as np
from .visual_trace import world_point_to_uv, trace_endpoint_to_world

p=argparse.ArgumentParser(); p.add_argument("--scene",type=Path,required=True); a=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(a.scene)); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
for name in ("wood_block","right_wrist_yaw_link"):
    point=d.xpos[m.body(name).id].copy(); uv=world_point_to_uv(m,d,"blocks_overview",point,(640,480))
    back=np.array(trace_endpoint_to_world(m,d,"blocks_overview",uv,(640,480),point[2]).endpoint_world_m)
    print(name,"point",point,"uv",uv,"back",back,"error",np.linalg.norm(back-point))
