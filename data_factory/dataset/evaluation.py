import json,html
from pathlib import Path
import numpy as np
from .contracts import read,write,require,sha

def offline(checkpoint,dataset,output):
 import torch
 from va_research.models.v4.model import V4Policy
 from va_research.models.v4.training import metrics
 c=torch.load(checkpoint,map_location='cpu',weights_only=False);m=read(Path(dataset)/'manifest.json');s=read(Path(dataset)/'split.json')
 with np.load(Path(dataset)/'samples.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 model=V4Policy();model.load_state_dict(c['model']);model.eval();keys=['rgb','body','hand'];x=[torch.tensor(a[k],dtype=torch.float32) for k in keys]
 with torch.no_grad():p=model(*x)
 moving=torch.ones(14,dtype=torch.bool);target={k:torch.tensor(a[k]) for k in ('body_tokens','hand_ref','phase','progress')};met=metrics(p,target,moving)
 pred={k:v.detach().numpy() for k,v in p.items()};absb=np.abs(pred['body_tokens']-a['body_tokens']);absh=np.abs(pred['hand_ref']-a['hand_ref']);met.update(body_mae_by_horizon=absb.mean((0,2)).tolist(),hand_mae_by_horizon=absh.mean((0,2)).tolist(),
  confusion=np.histogram2d(a['phase'],pred['phase_logits'].argmax(-1),bins=[3,3],range=[[0,3],[0,3]])[0].astype(int).tolist(),overlap=dict(train_episodes=s['train_episodes'],val_episodes=s['val_episodes'],evaluation_episodes=sorted({r['episode_id'] for r in [json.loads(x) for x in (Path(dataset)/'index.jsonl').read_text().splitlines()]}),validation_overlap=True))
 out=Path(output);require(not out.exists(),'Evaluation output is immutable');out.mkdir(parents=True);np.savez_compressed(out/'predictions.npz',**pred);write(out/'metrics.json',dict(checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha(checkpoint),dataset=str(Path(dataset).resolve()),metrics=met))
 text='<html><meta charset="utf-8"><h1>Open-loop V4 evaluation</h1><p>Validation overlap is explicitly reported; this is not independent held-out generalization.</p><pre>'+html.escape(json.dumps(met,indent=2))+'</pre></html>'; (out/'report.html').write_text(text);return met
