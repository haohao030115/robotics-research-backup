"""Replay failed model cases using the source validation's five standby ticks."""
import fcntl,json,traceback,time
import produce_expert10 as p
from export_canonical_rule_package import export
B=p.BASE
lock=(B/'production.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
rows=p.read(B/'results.json');p.write(B/'results_before_startup5.json',rows)
for r in rows:
 if r['physical_success'] or r['producer_type']=='rule_based':continue
 eid=r['episode_id'];ep=B/'attempts'/eid;logs=B/'logs'/eid
 previous=dict(r);candidate='actual'
 try:
  for i in range(3):
   name='startup5_diagnostic' if eid=='kimodo_0000' and i==0 else f'startup5_{i:03d}'
   p.write(B/'current.json',dict(phase='startup5_replay',episode_id=eid,attempt=i))
   folder=ep/'validation'/name
   if not (folder/'report.json').exists():
    rc=p.run('replay_rule_source.py',['--episode',ep,'--candidate',candidate,'--run-name',name,'--warmup-frames',5,'--no-images'],logs/(name+'.log'))
   report=p.read(folder/'report.json')
   if report['expert_valid']:
    new=export(ep,B/'episodes'/eid,eid,r['producer_type']);r.clear();r.update(new);r['previous_failed_result']=previous;break
   r.setdefault('startup5_attempts',[]).append(dict(run=name,failure_reason=report['failure_reason'],max_lift=report['max_lift_m'],hold_duration=report['max_continuous_hold_seconds']))
   if report['failure_reason']!=['cube_lift_opposing_contact_and_2s_hold_not_met']:break
   if i<2:
    candidate=f'startup5_corrected_{i:03d}'
    if p.run('refine_rule_reference.py',['--episode',ep,'--run',name,'--name',candidate,'--gain',.6],logs/(candidate+'.log')):break
 except Exception as e:r['startup5_exception']=str(e);(logs/'startup5_exception.txt').write_text(traceback.format_exc())
 p.summarize(rows);print(eid,r['physical_success'],flush=True)
p.write(B/'current.json',dict(phase='finished',successful=sum(r['physical_success'] for r in rows),completed_attempts=len(rows)))
