"""First batch: 4 Kimodo / 3 Rule / 3 ARDY, five shared cases."""
import os,sys,json,csv,time,subprocess,fcntl,traceback
from pathlib import Path
from export_canonical_rule_package import export
from project_paths import ROOT
BASE=ROOT/'outputs/expert10_20261008';SCRIPTS=ROOT/'scripts'
def read(p):return json.loads(p.read_text())
def write(p,x):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)
def run(script,args,log):
 with log.open('w') as f:return subprocess.run([sys.executable,'-B',str(SCRIPTS/script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode

def plan():
 cfg=read(ROOT/'configs/expert10_cases_20261008.json');rows=[]
 selections={'rule_based':[0,3,1],'kimodo':[0,1,3,4],'ardy':[2,3,4]}
 for producer in ['rule_based','kimodo','ardy']:
  for i,c in enumerate(selections[producer]):
   row=dict(episode_id=f'{producer}_{i:04d}',producer_type=producer,case=cfg['cases'][c],variant=i//5)
   if producer=='kimodo':row['source']=str(ROOT.parent/f'kimodo-t6/g1-integration/task8_expert_reference_batch_v1_20261007/episodes/kimodo_episode_{i+1:04d}')
   if producer=='ardy':row['source']=str(ROOT.parent/'workspace/tuchengyuan/task5/ardy_geometry_20261008/ardy_episode_0000')
   rows.append(row)
 return dict(configuration=cfg,episodes=rows,limits=dict(max_replays=4,max_corrections=2),generation_note='Model sources are cached genuine model references; each case gets independent kinematic adaptation/IK and actual replay. Rule trajectories generated from scratch. Historical rule_episode_0000 is never used.')

def summarize(rows):
 write(BASE/'results.json',rows)
 cols=['episode_id','producer_type','block_xy','task_arm','physical_success','max_lift','hold_duration','reference_path']
 with (BASE/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r.get(k) for k in cols} for r in rows)
 write(BASE/'status.json',dict(updated_unix=time.time(),completed_attempts=len(rows),successful=sum(r['physical_success'] for r in rows),target_successes=10,by_producer={p:sum(r['physical_success'] and r['producer_type']==p for r in rows) for p in ['kimodo','rule_based','ardy']},failures=[r for r in rows if not r['physical_success']]))

def main():
 BASE.mkdir(exist_ok=True);lock=(BASE/'production.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=plan()
 if (BASE/'plan.json').exists():
  if read(BASE/'plan.json')!=p:raise ValueError('Existing plan differs')
 else:write(BASE/'plan.json',p)
 rows=read(BASE/'results.json') if (BASE/'results.json').exists() else []
 for row in p['episodes']:
  eid=row['episode_id']
  if any(r['episode_id']==eid for r in rows):continue
  logs=BASE/'logs'/eid;logs.mkdir(parents=True,exist_ok=True);ep=BASE/'attempts'/eid;case=row['case'];attempts=[]
  result=dict(episode_id=eid,producer_type=row['producer_type'],block_xy=case['block_xy'],task_arm=case['task_arm'],physical_success=False,max_lift=None,hold_duration=None,reference_path=None)
  write(BASE/'current.json',dict(episode_id=eid,phase='generate_or_adapt',started_unix=time.time()));print('START',eid,flush=True)
  try:
   if not (ep/'metadata.json').exists():
    if row['producer_type']=='rule_based':
     args=['--episode',ep,'--case-id',case['id'],'--cube-x',case['block_xy'][0],'--cube-y',case['block_xy'][1],'--task-arm',case['task_arm'],'--calibration',ROOT/'outputs/rule_diversity_20261008/readiness_calibration.json','--approach-scale',1+.04*row['variant']]
     rc=run('generate_rule_reference.py',args,logs/'generation.log')
    else:
     rc=run('adapt_model_reference.py',['--source',row['source'],'--episode',ep,'--case-json',json.dumps(case),'--producer',row['producer_type'],'--entry-blend',6.+.2*row['variant']],logs/'adaptation.log')
    if rc:raise RuntimeError(f'Generation/adaptation failed: exit {rc}')
   meta=read(ep/'metadata.json')
   if not meta['source_success']:raise RuntimeError('Rule source grasp failed')
   candidate='actual';corrections=0
   for index in range(1,5):
    name=f'canonical_{index:03d}';folder=ep/'validation'/name
    write(BASE/'current.json',dict(episode_id=eid,phase='canonical_replay',attempt=index))
    if not (folder/'report.json').exists():
     rc=run('replay_rule_source.py',['--episode',ep,'--candidate',candidate,'--run-name',name,'--no-images'],logs/(name+'.log'))
    if not (folder/'report.json').exists():
     attempts.append(dict(run=name,failure='infrastructure_no_report'));continue
    report=read(folder/'report.json');attempts.append(dict(run=name,expert_valid=report['expert_valid'],failure_reason=report['failure_reason']))
    result.update(max_lift=report['max_lift_m'],hold_duration=report['max_continuous_hold_seconds'])
    if report['expert_valid']:
     result=export(ep,BASE/'episodes'/eid,eid,row['producer_type']);break
    if not report['frame_coverage_complete']:continue
    reasons=report['failure_reason']
    if reasons==['cube_lift_opposing_contact_and_2s_hold_not_met'] and corrections<2 and index<4:
     corrections+=1;candidate=f'corrected_{corrections:03d}'
     rc=run('refine_rule_reference.py',['--episode',ep,'--run',name,'--name',candidate,'--gain',.6],logs/(candidate+'.log'))
     if rc:raise RuntimeError('IK correction failed')
    elif any('contact' in r or 'fell' in r or 'supported' in r for r in reasons):break
   result['attempts']=attempts
   if not result['physical_success']:result['failure_reason']='Bounded replay attempts failed'
  except Exception as exc:result.update(failure_reason=str(exc),attempts=attempts);(logs/'exception.txt').write_text(traceback.format_exc())
  rows.append(result);summarize(rows);print('RESULT',eid,result['physical_success'],result.get('failure_reason'),flush=True)
 write(BASE/'current.json',dict(phase='finished',completed_attempts=len(rows),successful=sum(r['physical_success'] for r in rows)))
if __name__=='__main__':main()
