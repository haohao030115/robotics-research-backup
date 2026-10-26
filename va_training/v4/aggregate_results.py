"""Verify all trial artifacts and write a single comparison manifest/table."""
import hashlib, json, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/va_v4_ablation'
CONFIGS=['AUX-0_baseline','AUX-1_no_phase','AUX-2_no_progress','AUX-3_no_phase_progress','ARCH-1_mlp_fusion','ARCH-2_proprio_only']

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def verify_video(path):
 import imageio.v2 as imageio
 reader=imageio.get_reader(path); n=0; first=None; last=None
 try:
  for frame in reader:
   if first is None:first=list(frame.shape)
   last=list(frame.shape); n+=1
 finally:reader.close()
 return dict(frame_count=n,first_shape=first,last_shape=last,playable=n>0)

def main():
 scenarios=json.loads((OUT/'evaluation_scenarios.json').read_text())['scenarios']; rows=[]; files=[]
 required=['checkpoint.pt','offline_metrics.json','trial_metadata.json','native/report.json','native/provenance.json','native/renderer_worker.json','native/physics.npz','native/physical_metrics.json','native/contacts.json','native/control_timeline.json','native/scheduler_events.json','native/predictions.npz','native/inference_timing.json','trial.mp4','trial.metadata.json','video_renderer.json']
 for cfg in CONFIGS:
  for s in scenarios:
   trial=OUT/cfg/'trials'/s['trial_id']; missing=[x for x in required if not (trial/x).is_file()]
   report=json.loads((trial/'native/report.json').read_text()) if not (trial/'native/report.json').exists() else json.loads((trial/'native/report.json').read_text())
   renderer=json.loads((trial/'native/renderer_worker.json').read_text()) if (trial/'native/renderer_worker.json').exists() else {}
   video_renderer=json.loads((trial/'video_renderer.json').read_text()) if (trial/'video_renderer.json').exists() else {}
   video=verify_video(trial/'trial.mp4') if (trial/'trial.mp4').exists() else dict(frame_count=0,playable=False)
   body_err=hand_err=None
   if (trial/'native/physics.npz').exists():
    with np.load(trial/'native/physics.npz',allow_pickle=False) as z:
     if 'body_target_q' in z and 'body_q' in z: body_err=float(np.abs(z['body_target_q']-z['body_q']).mean())
     if 'hand_target_q' in z and 'hand_q' in z: hand_err=float(np.abs(z['hand_target_q']-z['hand_q']).mean())
   pm=json.loads((trial/'native/physical_metrics.json').read_text()) if (trial/'native/physical_metrics.json').exists() else []
   if pm:
    body_vals=[np.abs(np.asarray(x['body_target_q'])-np.asarray(x['body_actual_q'])).mean() for x in pm if 'body_target_q' in x and 'body_actual_q' in x]
    hand_vals=[np.abs(np.asarray(x['hand_target_q'])-np.asarray(x['hand_actual_q'])).mean() for x in pm if 'hand_target_q' in x and 'hand_actual_q' in x]
    body_err=float(np.mean(body_vals)) if body_vals else body_err; hand_err=float(np.mean(hand_vals)) if hand_vals else hand_err
   # Native policy logs expose the effective 29-D body targets; the control
   # timeline records the applied 14-D hand targets. Compare each to the
   # measured physical state at the same control tick.
   try:
    with np.load(trial/'native/physics.npz',allow_pickle=False) as z: body_actual=z['body_q']; hand_actual=z['hand_q']
    import csv
    act=trial/'native/policy_logs/action.csv'
    if body_err is None and act.exists():
     vals=[]
     with act.open() as f:
      for row in csv.DictReader(f):
       tick=int(row['index'])
       if 0<=tick<len(body_actual): vals.append(np.abs(np.asarray([float(row[f'act_{j}']) for j in range(29)])-body_actual[tick]).mean())
     if vals: body_err=float(np.mean(vals[1:] if len(vals)>1 else vals))
    tl=json.loads((trial/'native/control_timeline.json').read_text())
    vals=[np.abs(np.asarray(x['hand_target_applied'])-hand_actual[int(x['actual_tick'])]).mean() for x in tl if 'hand_target_applied' in x and 0<=int(x['actual_tick'])<len(hand_actual)]
    if vals: hand_err=float(np.mean(vals))
   except Exception:
    pass
   max_lift=float(report.get('max_lift_m',0)); max_hold=float(report.get('max_continuous_hold_seconds',0))
   rows.append(dict(config=cfg,trial_id=s['trial_id'],seed=s['seed'],scenario_manifest=s['source_manifest'],
       missing=missing,video=video,renderer=dict(native_hardware=renderer.get('hardware_verified',False),video_hardware=video_renderer.get('hardware_verified',False),gl_vendor=renderer.get('gl_vendor'),gl_renderer=renderer.get('gl_renderer')),
       native_complete=bool(report.get('rollout_complete',False)),strict_runtime_pass=bool(report.get('strict_runtime_pass',False)),
       physical_success=bool(report.get('physical_grasp_success',False)),strict_physical_success=bool(report.get('physical_grasp_success',False) and report.get('strict_runtime_pass',False)),
       max_lift_m=max_lift,continuous_hold_s=max_hold,robot_table_contact_steps=report.get('robot_table_contact_steps'),robot_self_contact_steps=report.get('robot_self_contact_steps'),
       body_tracking_mae=body_err,hand_tracking_mae=hand_err,runtime_failure=report.get('failure'),scheduler=report.get('scheduler',{}),
       physical_metric_frames=len(pm),checkpoint_sha256=sha(trial/'checkpoint.pt') if (trial/'checkpoint.pt').exists() else None,video_sha256=sha(trial/'trial.mp4') if (trial/'trial.mp4').exists() else None))
   files.extend([dict(config=cfg,trial_id=s['trial_id'],kind='checkpoint',path=str(trial/'checkpoint.pt'),sha256=sha(trial/'checkpoint.pt')),
                 dict(config=cfg,trial_id=s['trial_id'],kind='video',path=str(trial/'trial.mp4'),sha256=sha(trial/'trial.mp4'))])
 table=[]
 for cfg in CONFIGS:
  rs=[r for r in rows if r['config']==cfg]
  table.append(dict(config=cfg,trials=len(rs),complete_native=sum(r['native_complete'] for r in rs),playable_videos=sum(r['video'].get('playable',False) and not r['missing'] for r in rs),
      physical_success=sum(r['physical_success'] for r in rs),strict_physical_success=sum(r['strict_physical_success'] for r in rs),
      mean_max_lift_m=float(np.mean([r['max_lift_m'] for r in rs])),individual_max_lift_m=[r['max_lift_m'] for r in rs],
      mean_continuous_hold_s=float(np.mean([r['continuous_hold_s'] for r in rs])),individual_continuous_hold_s=[r['continuous_hold_s'] for r in rs],
      robot_table_contact_steps=[r['robot_table_contact_steps'] for r in rs],robot_self_contact_steps=[r['robot_self_contact_steps'] for r in rs],
      body_tracking_mae=[r['body_tracking_mae'] for r in rs],hand_tracking_mae=[r['hand_tracking_mae'] for r in rs],
      scheduler_failures=sum(int(r['scheduler'].get('deadline_misses',0))+int(r['scheduler'].get('uncovered_ticks',0)) for r in rs),runtime_failures=sum(bool(r['runtime_failure']) for r in rs)))
 manifest=dict(schema_version=1,created='2026-10-18',source_dataset='expert_datasets/train_dataset_v0',scenarios=scenarios,configs=CONFIGS,
   counts=dict(configurations=len(CONFIGS),trials=len(rows),videos=sum(r['video'].get('playable',False) and not r['missing'] for r in rows)),trials=rows,files=files)
 (OUT/'trial_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n'); (OUT/'comparison_table.json').write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(counts=manifest['counts'],table=table),indent=2))

if __name__=='__main__':main()
