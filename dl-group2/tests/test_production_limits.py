"""Bounded retry/resume tests with simulated subprocess reports, no physics mocking claims."""
import json
from pathlib import Path
import sys,tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from produce_experts import produce


def test_limits_and_resume():
    for reason,expected_replays,expected_corrections in [('cube_lift_opposing_contact_and_2s_hold_not_met',3,2),('neutral_rollout_robot_self_contact',1,0)]:
        with tempfile.TemporaryDirectory() as tmp:
            ep=Path(tmp)/'episodes/case';ep.mkdir(parents=True)
            (ep/'metadata.json').write_text(json.dumps({'source_success':True,'expert_valid':False}))
            calls=[]
            def fake(script,args,log):
                calls.append(script);args=list(map(str,args))
                if script=='run_sonic_grasp.py':
                    run=args[args.index('--run-name')+1];candidate=args[args.index('--candidate')+1]
                    folder=ep/'validation'/run;folder.mkdir(parents=True)
                    (folder/'metadata.json').write_text(json.dumps({'candidate_type':candidate}))
                    (folder/'report.json').write_text(json.dumps(dict(expert_valid=False,failure_reason=[reason],frame_coverage_complete=True,recorded_frames=0,final_lift_m=0.,final_continuous_hold_seconds=0.,robot_table_contact_steps=0,robot_self_contact_steps=0)))
                    return 2
                if script=='refine_freebase_reference.py':
                    name=args[args.index('--name')+1];folder=ep/'candidates'/name;folder.mkdir(parents=True);(folder/'metadata.json').write_text('{}');return 0
                raise AssertionError(script)
            sample={'case_id':'case','xy':[.4,-.15],'sample_index':0};args=SimpleNamespace(seed=0,max_replays=3,max_corrections=2)
            with patch('produce_experts.command',fake):
                first=produce(ep,sample,args)
                assert first['replay_attempts']==expected_replays and first['correction_iterations']==expected_corrections
                assert first['generation_method']=='failed_sample' and not first['training_eligible']
                before=len(calls);second=produce(ep,sample,args)
                assert len(calls)==before and second['replay_attempts']==expected_replays
    print('PASS finite correction/replay budgets, stop-on-safety-failure, and resume without repeated subprocesses')

if __name__=='__main__':test_limits_and_resume()
