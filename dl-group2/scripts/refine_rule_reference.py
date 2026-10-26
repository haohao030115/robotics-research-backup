"""Arm-aware offline IK tracking compensation; every result needs a fresh native replay."""
from pathlib import Path
import runpy
import scene,grasp_reference
from project_paths import SONIC_REPO,ROBOT_RELATIVE
from readiness_grasp_reference import ReadinessGraspReference
original=scene.load_scene
scene.load_scene=lambda **kw:original(**dict(kw,robot_path=SONIC_REPO/ROBOT_RELATIVE))
# Reuse the established correction algorithm; select joints from explicit source metadata.
source=Path(__file__).with_name('refine_freebase_reference.py').read_text()
source=source.replace("wrist=m.body('right_wrist_yaw_link').id", "wrist=m.body(meta.get('task_arm','right')+'_wrist_yaw_link').id")
source=source.replace('g=GraspReference(m,d)',"g=ReadinessGraspReference(m,d,meta.get('task_arm','right'),{})")
exec(compile(source,str(Path(__file__).with_name('refine_freebase_reference.py')),'exec'),dict(__name__='__main__',ReadinessGraspReference=ReadinessGraspReference))
