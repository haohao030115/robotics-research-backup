"""Read-only audit of delivered arrays and their certified replay hashes."""
import json,hashlib
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1]/'outputs/expert10_20261008'
rows=[]
for ep in sorted((B/'episodes').glob('*')):
 m=json.loads((ep/'metadata.json').read_text());v=json.loads((ep/'reference_validation.json').read_text());pr=json.loads((ep/'provenance.json').read_text())
 h=hashlib.sha256((ep/'reference.npz').read_bytes()).hexdigest()
 assert h==v['reference_sha256']==pr['final_reference_sha256']
 report=Path(v['replay_report_path']);r=json.loads(report.read_text())
 assert hashlib.sha256(report.read_bytes()).hexdigest()==pr['actual_replay_report_sha256']
 assert r['expert_valid'] and r['episode_completed'] and not r['failure_reason']
 with np.load(ep/'reference.npz') as a:
  n=len(a['time_s']);assert set(a.files)==set('time_s body_ref_q body_ref_dq root_ref_pos root_ref_quat hand_ref_q'.split())
  for k,d in [('body_ref_q',29),('body_ref_dq',29),('root_ref_pos',3),('root_ref_quat',4),('hand_ref_q',14)]:assert a[k].shape==(n,d) and np.isfinite(a[k]).all()
  assert np.allclose(np.diff(a['time_s']),.02) and np.isfinite(a['time_s']).all()
 assert m['task_arm']==r['task_arm'] and len(m['case']['block_xy'])==2 and 'block_yaw_deg' in m['case']
 rows.append(dict(episode_id=ep.name,producer=m['producer_type'],case=m['case'],task_arm=m['task_arm'],max_lift=v['max_lift'],hold_duration=v['hold_duration'],hash_verified=True))
print(json.dumps(dict(certified_count=len(rows),episodes=rows),indent=2))
