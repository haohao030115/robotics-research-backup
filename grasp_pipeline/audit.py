import json
from pathlib import Path
import numpy as np
from .contracts import read,write,require,sha

def build(dataset, output):
 d=Path(dataset); out=Path(output);require((d/'manifest.json').is_file(),'Dataset manifest missing')
 m=read(d/'manifest.json'); split=read(d/'split.json')
 with np.load(d/'samples.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 episodes=m['episodes']; stats=dict(dataset_id=m['dataset_id'],sample_count=m['sample_count'],episode_count=len(episodes),
   train_episodes=split['train_episodes'],val_episodes=split['val_episodes'],producer_composition={},task_arm_composition={},
   cube_xy={'available':False},wrist={'available':False},rgb={'available':True,'shape':list(a['rgb'].shape),'mean':a['rgb'].mean((0,2,3)).tolist(),'std':a['rgb'].std((0,2,3)).tolist()},
   body_state={'mean':a['body'].mean(0).tolist(),'std':a['body'].std(0).tolist()},hand_state={'mean':a['hand'].mean(0).tolist(),'std':a['hand'].std(0).tolist()},
   token_target={'mean':a['body_tokens'].mean((0,1)).tolist(),'std':a['body_tokens'].std((0,1)).tolist()},hand_target={'mean':a['hand_ref'].mean((0,1)).tolist(),'std':a['hand_ref'].std((0,1)).tolist()},
   phase_distribution=np.bincount(a['phase'],minlength=3).tolist(),progress={'min':float(a['progress'].min()),'max':float(a['progress'].max()),'mean':float(a['progress'].mean())},
   unavailable=['cube XY distribution','wrist XYZ trajectories','wrist Z versus time','reach/grasp/lift timing'])
 for e in episodes:
  md=e.get('metadata',{}); producer=md.get('producer_type','unknown');arm=md.get('task_arm','unknown')
  stats['producer_composition'][producer]=stats['producer_composition'].get(producer,0)+1;stats['task_arm_composition'][arm]=stats['task_arm_composition'].get(arm,0)+1
  case=md.get('case',{});xy=case.get('block_xy')
  if xy is not None:stats['cube_xy']['available']=True;stats['cube_xy'].setdefault('values',[]).append(xy)
 out.mkdir(parents=True,exist_ok=False);write(out/'summary.json',stats)
 html='''<!doctype html><meta charset="utf-8"><title>Dataset Audit</title><style>body{font:14px sans-serif;max-width:1100px;margin:2em}pre{white-space:pre-wrap}img{max-width:180px;margin:4px}</style><h1>Dataset Audit</h1>'''+f'<p>Descriptive only; no filtering, scoring or duplicate detection. Dataset: {m["dataset_id"]}</p><pre>{json.dumps(stats,indent=2,ensure_ascii=False)}</pre><h2>RGB samples</h2>'
 (out/'figures').mkdir();
 for i in np.linspace(0,len(a['rgb'])-1,min(12,len(a['rgb'])),dtype=int):
  try:
   from PIL import Image
   name=f'frame_{i:04d}.png';Image.fromarray(np.transpose(a['rgb'][i]*255,(1,2,0)).astype('uint8')).save(out/'figures'/name);html+=f'<img src="figures/{name}">'
  except Exception: html+='<p>RGB gallery unavailable</p>';break
 (out/'report.html').write_text(html+'<h2>Raw summary</h2><p>All unavailable fields are marked in summary.json.</p>')
 return stats
