"""Actual EGL/MuJoCo render probe and explicit hardware policy; never hide driver warnings."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from .common import ROOT, dump, digest, require

PROFILE=ROOT/'canonical_va/graphics_profile.json'
SOFTWARE=('llvmpipe','softpipe','software rasterizer','swrast','swiftshader')


def software_renderer(name):
    return any(word in name.lower() for word in SOFTWARE)


def renderer_allowed(info,mode,allow_software=False):
    require(info.get('render_succeeded') is True,'MuJoCo Renderer probe failed')
    require(bool(info.get('GL_RENDERER')),'GL renderer identity unavailable')
    software=software_renderer(info['GL_RENDERER'])
    if software and mode=='production' and not allow_software:
        raise ValueError('SOFTWARE_RENDERER_REJECTED: '+info['GL_RENDERER']+'; select verified hardware EGL profile or explicitly --allow-software-rendering')
    return ['Software RGB rendering explicitly allowed' if allow_software else 'Audit uses software RGB rendering'] if software else []


def bootstrap(profile='auto'):
    """Library search paths must be set BEFORE interpreter/dynamic-loader initialization."""
    require(profile in ('auto','current','mthreads'),'Unknown rendering profile')
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    if profile=='current':return
    config=json.loads(PROFILE.read_text())
    files={key:ROOT/rel for key,rel in config['files'].items()}
    available=all(p.is_file() and p.stat().st_size>0 for p in files.values())
    if not available:
        require(profile!='mthreads','Verified MThreads graphics stage missing')
        return
    for key,p in files.items():require(digest(p)==config['sha256'][key],'Graphics driver identity changed: '+str(p))
    desired=dict(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',__EGL_VENDOR_LIBRARY_FILENAMES=str(files['vendor']),LIBGL_DRIVERS_PATH=str(files['dri'].parent))
    lib=str(files['compiler'].parent);old=os.environ.get('LD_LIBRARY_PATH','')
    desired['LD_LIBRARY_PATH']=lib+(':'+old if old and old.split(':')[0]!=lib else (':'+':'.join(old.split(':')[1:]) if ':' in old else ''))
    if all(os.environ.get(k)==v for k,v in desired.items()):return
    require('mujoco' not in sys.modules,'Configure graphics before importing MuJoCo')
    os.environ.update(desired)
    os.execvpe(sys.executable,[sys.executable,*sys.orig_argv[1:]],os.environ)


def probe(frames=3):
    import numpy as np
    import mujoco
    from OpenGL import GL,EGL
    from .runtime import load_scene,ROBOT,PROJECT,camera_config,validate_camera,CONTRACT
    m,d=load_scene(supported=False,robot_path=ROBOT,scene_path=PROJECT/'scenes/tabletop.xml')
    camera=camera_config(m);validate_camera(camera,json.loads(CONTRACT.read_text())['camera'])
    renderer=mujoco.Renderer(m,height=240,width=320)
    try:
        def gl(key):
            value=GL.glGetString(key);return value.decode() if value else ''
        info=dict(GL_VENDOR=gl(GL.GL_VENDOR),GL_RENDERER=gl(GL.GL_RENDERER),GL_VERSION=gl(GL.GL_VERSION),
            EGL_VENDOR=EGL.eglQueryString(EGL.eglGetCurrentDisplay(),EGL.EGL_VENDOR).decode(),
            environment={k:os.environ.get(k) for k in ('MUJOCO_GL','PYOPENGL_PLATFORM','__EGL_VENDOR_LIBRARY_FILENAMES','LIBGL_DRIVERS_PATH','LD_LIBRARY_PATH')},camera=camera)
        times=[]
        for i in range(frames+2):
            start=time.perf_counter();mujoco.mj_forward(m,d);renderer.update_scene(d,camera='head_rgb');rgb=renderer.render().copy()
            if i>=2:times.append(time.perf_counter()-start)
        require(rgb.shape==(240,320,3) and rgb.dtype==np.uint8 and float(rgb.std())>0,'Actual MuJoCo render invalid')
        info.update(render_succeeded=True,software=software_renderer(info['GL_RENDERER']),frames=frames,
            median_render_seconds=float(np.median(times)),mean_render_seconds=float(np.mean(times)),pixel_std=float(rgb.std()),
            image_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest())
        return info
    finally:renderer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['auto','current','mthreads'],default='auto')
    p.add_argument('--mode',choices=['audit','production'],default='audit');p.add_argument('--allow-software-rendering',action='store_true')
    p.add_argument('--frames',type=int,default=20);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    bootstrap(args.profile);info=probe(args.frames)
    try:info['warnings']=renderer_allowed(info,args.mode,args.allow_software_rendering);info['accepted']=True
    except ValueError as exc:info.update(accepted=False,error=str(exc))
    dump(args.output,info);print(json.dumps(info,indent=2));return 0 if info['accepted'] else 2

if __name__=='__main__':raise SystemExit(main())
