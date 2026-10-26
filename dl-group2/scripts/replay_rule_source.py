"""Use original SONIC/DDS runner with the explicit canonical robot XML."""
import runpy
from pathlib import Path
import scene
from project_paths import SONIC_REPO,ROBOT_RELATIVE
original=scene.load_scene
scene.load_scene=lambda **kw: original(**dict(kw,robot_path=SONIC_REPO/ROBOT_RELATIVE))
runpy.run_path(str(Path(__file__).with_name('run_sonic_grasp.py')),run_name='__main__')
