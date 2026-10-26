"""Synchronize and horizontally stack existing three-camera replay videos."""
from pathlib import Path
import hashlib
import json
import subprocess
import imageio_ffmpeg

root = Path(__file__).resolve().parents[1]
out = root / 'videos_three_views'
out.mkdir(exist_ok=True)
cameras = ['task_overview', 'head_rgb', 'wrist_rgb']
records = []
for name, run in [('success', 'center_seed1_matrix_calibrated'), ('fail', 'center_seed0_matrix')]:
    replay = root / 'runs' / run / 'replay'
    sources = [replay / 'visualization' / (camera + '.mp4') for camera in cameras]
    metadata = []
    for path in sources:
        reader = imageio_ffmpeg.read_frames(str(path))
        info = next(reader)
        reader.close()
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        metadata.append(dict(fps=info['fps'], size=info['size'], frames=count, duration_s=seconds))
    if not all(m == metadata[0] for m in metadata):
        raise ValueError('Camera frame counts/timing/resolution differ')
    destination = out / (name + '_three_views.mp4')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-n', '-filter_complex_threads', '2']
    for path in sources:
        command.extend(['-i', str(path)])
    command.extend(['-filter_complex', '[0:v][1:v][2:v]hstack=inputs=3[v]', '-map', '[v]',
                    '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                    '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)])
    subprocess.run(command, check=True)
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(destination))
    reader = imageio_ffmpeg.read_frames(str(destination))
    info = next(reader)
    # Decode all frames to ensure the finished file is readable.
    decoded = sum(1 for _ in reader)
    reader.close()
    width, height = metadata[0]['size']
    if (count, decoded, info['size'], info['fps']) != (metadata[0]['frames'], metadata[0]['frames'], (width*3, height), metadata[0]['fps']):
        raise ValueError('Composed video failed output verification')
    validation = json.loads((replay/'task_validation.json').read_text())
    record = dict(name=name, run=run, layout='left: overview; center: head; right: wrist',
                  output=str(destination), frames=count, duration_s=seconds, fps=info['fps'], resolution=info['size'],
                  physical_success=validation['physical_success'], strict_expert_valid=validation['strict_no_table_expert_valid'],
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                  output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), command=command)
    records.append(record)
    print(json.dumps(record), flush=True)
(out/'composition.json').write_text(json.dumps(records, indent=2)+'\n')
