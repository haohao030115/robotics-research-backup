"""Artifact adapter to the original v4.train_merged optimization loop."""
import json
from pathlib import Path
import numpy as np
from .contracts import read,write,require,sha,verify_files

def load_baseline_inputs(dataset):
 import torch
 from types import SimpleNamespace
 from .dataset import validate,baseline_class
 d=Path(dataset);m=validate(d);split=read(d/'split.json');CanonicalV4=baseline_class()
 with np.load(d/'samples.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
 datasets=[];provenance={}
 for e in m['episodes']:
  verify_files(e['source_hashes']);original=CanonicalV4(e['source_manifest'])
  data={k:torch.from_numpy(a[e['sample_start']:e['sample_end']]) for k,a in arrays.items()}
  require(all(torch.equal(v,original.data[k]) for k,v in data.items()),'Serialized tensors differ from original CanonicalV4')
  ds=SimpleNamespace(data=data,episode_id=e['episode_id'],moving=original.moving,keep=original.keep,manifest=original.manifest,provenance=original.provenance)
  datasets.append(ds);provenance[ds.episode_id]=ds.provenance
 require(all(np.array_equal(ds.moving,datasets[0].moving) for ds in datasets),'Moving joint masks differ')
 require(bool(split['val_episodes']),'Merged baseline requires nonempty validation split')
 return d/'manifest.json',d/'samples.npz',list(range(sum(ds.provenance.get('num_samples',len(ds.keep)) for ds in datasets))),datasets,provenance,set(split['train_episodes']),set(split['val_episodes'])

def run(dataset,output,config=None):
 from va_research.models.v4.train_merged import main
 d=Path(dataset).resolve();out=Path(output).resolve();require(not out.exists(),'Training output is immutable')
 from .dataset import validate
 m=validate(d);out.mkdir(parents=True)
 write(out/'manifest.json',dict(schema_version=1,artifact_type='training',dataset_id=m['dataset_id'],dataset=str(d),dataset_sha256=sha(d/'COMPLETED.json'),normalization='train-only mean/std clamp_min(.05)',model='v4',trainer='v4.train_merged.main'))
 write(out/'status.json',dict(state='RUNNING',finished=False))
 try:
  main(dataset=d,output=out,config=config)
  rows=[json.loads(x) for x in (out/'training.jsonl').read_text().splitlines()]
  cfg=read(out/'config.json')
  for row in rows:row['learning_rate']=cfg['learning_rate']*(.3**sum(row['epoch']>v for v in (1500,3000)))
  write(out/'history.json',rows)
  status=read(out/'training_status.json');status.update(state='SUCCEEDED',offline_gate=status['train_gate'],best_checkpoint=str(out/'checkpoint.pt'))
  write(out/'status.json',status)
 except BaseException as exc:
  write(out/'status.json',dict(state='FAILED',finished=False,error=repr(exc)));raise
 return dict(path=str(out),checkpoint=str(out/'checkpoint.pt'),evidence=[str(out/n) for n in ('status.json','history.json','manifest.json','checkpoint.pt')])
