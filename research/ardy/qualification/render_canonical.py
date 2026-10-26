"""Render recorded canonical states; never run physics or certify grasp contacts.

Inside va-train: DISPLAY=:1 XAUTHORITY=/root/.Xauthority
LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw python render_canonical.py RUN_DIR
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def chart(title, times, values, unit):
    import numpy as np
    low, high = float(np.min(values)), float(np.max(values))
    pad = max((high-low)*.15, .001)
    low -= pad
    high += pad
    points = ' '.join(f'{55+700*t/max(times[-1],.001):.2f},{170-130*(v-low)/(high-low):.2f}'
                      for t, v in zip(times, values))
    return (f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 800 210" role="img">'
            f'<rect width="800" height="210" fill="#fff"/>'
            f'<path d="M55 35V170H760" stroke="#999" fill="none"/>'
            f'<text x="4" y="35">{high:.4f}</text><text x="4" y="170">{low:.4f}</text>'
            f'<text x="55" y="200">0 s</text><text x="685" y="200">{times[-1]:.2f} s</text>'
            f'<text x="65" y="25">{unit}</text>'
            f'<polyline points="{points}" stroke="#1779a6" stroke-width="2" fill="none"/></svg>')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    run = args.run_dir.resolve()
    out = (args.output_dir or run/'visualization').resolve()
    if out.exists():
        p.error('Output exists; choose a new --output-dir to preserve previous renders')
    provenance = json.loads((run/'provenance.json').read_text())
    result = json.loads((run/'result.json').read_text())
    for key in ('scene', 'robot'):
        if digest(provenance[key]) != provenance[key+'_sha256']:
            raise RuntimeError(f'{key} changed since recording; refusing misleading replay')
    recorded_builder_hash=provenance.get('audit_hashes',{}).get(provenance['builder'])
    if recorded_builder_hash and digest(provenance['builder'])!=recorded_builder_hash:
        raise RuntimeError('Scene builder changed since recording')
    os.environ.setdefault('SONIC_REPO', str(Path(os.environ.get('RESEARCH_ROOT', Path(__file__).resolve().parents[3])) / 'GR00T-WholeBodyControl'))
    sys.path.insert(0, str(Path(provenance['builder']).parent))
    import project_paths
    project_paths.default_robot_xml = lambda: Path(provenance['robot'])
    import mujoco
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from scene import load_scene
    model, data = load_scene(supported=False, robot_path=provenance['robot'], scene_path=provenance['scene'])
    model.opt.timestep = 1/provenance['physics_hz']
    a = np.load(run/'simulation.npz', allow_pickle=False)
    qpos, qvel = a['qpos'], a['qvel']
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or qvel.shape != (len(qpos),model.nv):
        raise ValueError('Recorded state/model shape mismatch')
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Nonfinite recorded state')
    if not np.array_equal(a['reference_frame'], np.arange(len(qpos))):
        raise ValueError('Missing reference frames; this simple replay requires contiguous frames')
    cameras = ['task_overview','head_rgb','wrist_rgb']
    for camera in cameras:
        model.camera(camera)
    out.mkdir(parents=True)
    times = a['sim_time']-a['sim_time'][0]
    # 50 Hz recorded states -> 10 Hz observation video, sampled without interpolation.
    indices = np.arange(0,len(qpos),5)
    grasp = bool(result.get('grasp_attempted'))
    preview_index = indices[-1] if grasp else indices[len(indices)//2]
    preview = Image.new('RGB',(960,270),(15,20,25))
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        option = mujoco.MjvOption()
        option.geomgroup[:] = 1
        option.geomgroup[0] = 0  # hide robot collision duplicates; retain visual group 1 and scene group 2
        for camera_index,camera in enumerate(cameras):
            with imageio.get_writer(out/f'{camera}.mp4',fps=10,codec='libx264',
                                    pixelformat='yuv420p',macro_block_size=16) as writer:
                for index in indices:
                    data.qpos[:] = qpos[index]
                    data.qvel[:] = qvel[index]
                    data.time = float(a['sim_time'][index])
                    mujoco.mj_forward(model,data)  # state reconstruction only, no mj_step
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0,0,640,38),fill=(12,18,24))
                    draw.text((8,4),f'RECORDED STATE REPLAY | {camera}',fill='white')
                    draw.text((8,21),f'frame {index} | t={times[index]:.3f}s | not live',fill='white')
                    writer.append_data(np.asarray(frame))
                    if index in (indices[0],indices[len(indices)//2],indices[3*len(indices)//4],indices[-1]):
                        frame.save(out/f'{camera}_frame{index:04d}.png')
                    if index == preview_index:
                        frame.save(out/f'{camera}_preview.png')
                        preview.paste(frame.resize((320,240)),(camera_index*320,30))
            print('VIDEO',out/f'{camera}.mp4',flush=True)
    label = 'Recorded grasp replay; see numerical criteria in report.' if grasp else 'Recorded standing replay; no grasp commanded.'
    ImageDraw.Draw(preview).text((8,8),label,fill='white')
    preview.save(out/'preview.png')
    q = a['root_quat']
    tilt = np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
    initial_z = float(a['cube_pos'][0,2])
    plots = chart('实际根部高度',times,a['root_pos'][:,2],'m')
    plots += chart('实际身体倾斜',times,tilt,'degrees')
    plots += chart('方块高度变化（相对回放首帧，仅供观察）',times,a['cube_pos'][:,2]-initial_z,'m')
    physics = run/'physics_steps.json'
    if physics.exists():
        active = [r for r in json.loads(physics.read_text()) if r['playing']]
        if active:
            pt = np.array([r['sim_time']-active[0]['sim_time'] for r in active])
            for key,title,unit in [('lift_m','正式判据：相对初始高度抬升，阈值0.10m','m'),
                    ('thumb_force_n','右拇指接触力，阈值1e-4N','N'),
                    ('opposing_finger_force_n','食指或中指接触力，阈值1e-4N','N'),
                    ('cube_table_contact','方块接触桌面：1表示仍接触','boolean'),
                    ('qualified','所有物理条件同时达标','boolean'),
                    ('hold_seconds','当前连续达标时间，阈值2秒','s')]:
                plots += chart(title,pt,np.array([r[key] for r in active]),unit)
    videos = ''.join(f'<section><h3>{cam}</h3><video controls preload="metadata" src="{cam}.mp4"></video>'
                     f'<p><a href="{cam}.mp4" download>下载视频</a></p></section>' for cam in cameras)
    page = ('<!doctype html><meta charset="utf-8"><title>Canonical replay inspection</title>'
            '<style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 20px;background:#f4f6f8;line-height:1.6}'
            'video{width:100%;max-width:640px}pre{white-space:pre-wrap;background:white;padding:16px}svg{width:100%}</style>'
            '<h1>Canonical 实际状态回放检查</h1>'
            f'<p>运行：{html.escape(run.name)}。这里回放的是实际物理状态，不是 ARDY 理想参考预览。</p>'
            f'<p><strong>{"抓取实验：成功与否请查看下方两个独立成功字段。" if grasp else "站立诊断没有执行抓取。"}</strong>'
            '视频只覆盖已记录的参考播放阶段；初始化仅有高度/倾斜采样。'
            '若显示接触力曲线，数据来自原运行的physics_steps.json，不是离线重算的力。视频本身不能认证抓取。</p>'
            f'{videos}<h2>实际状态曲线</h2>{plots}<h2>原始结果</h2>'
            f'<pre>{html.escape(json.dumps(result,ensure_ascii=False,indent=2))}</pre>'
            '<p>target_error_rad=0 表示参考目标流一致，不表示实际关节跟踪误差为零。'
            '对抓取实验，physical_success判断抬升/接触/连续保持，expert_valid还要求完整、同步、无辅助的执行检查通过。</p>')
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'render_manifest.json').write_text(json.dumps(dict(
        source=str(run/'simulation.npz'),source_sha256=digest(run/'simulation.npz'),
        scene_sha256=digest(provenance['scene']),robot_sha256=digest(provenance['robot']),
        builder_sha256_at_render=digest(provenance['builder']),
        builder_hash_at_recording_available=provenance['builder'] in provenance.get('audit_hashes',{}),
        physics_evidence_sha256=digest(physics) if physics.exists() else None,
        mujoco_version=mujoco.__version__,fps=10,source_fps=50,video_frames=len(indices),
        cameras=cameras,physics_reexecuted=False,grasp_validation_performed=False),indent=2)+'\n')
    print('REPORT',out/'index.html')

if __name__=='__main__':
    main()
