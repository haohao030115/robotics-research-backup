import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from grasp_pipeline.cli import main,parser
from grasp_pipeline.contracts import *
from grasp_pipeline.state import Experiment
from grasp_pipeline.dataset import make_split,source_info

class StateTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def exp(self):return Experiment.create(self.root/'exp','exp',{})
    def test_success_is_immutable_and_validated(self):
        e=self.exp();calls=[]
        def action(folder,log,child):
            calls.append(1);write(folder/'out.json',{'okay':True});return dict(exit_code=0,evidence=[str(folder/'out.json')],path=str(folder))
        with e:
            r=e.execute('dataset_build',{}, {},'mock',action,lambda r:require(read(Path(r['path'])/'out.json')['okay'],'bad'))
            self.assertEqual(r['state'],'SUCCEEDED');first=read(e.manifest)
            e.execute('dataset_build',{}, {},'mock',action,lambda r:None)
            self.assertEqual(len(calls),1);self.assertEqual(first,read(e.manifest))
            write(Path(r['result']['path'])/'out.json',{'okay':False})
            with self.assertRaises(PipelineError):e.execute('dataset_build',{}, {},'mock',action,lambda r:None)
            self.assertEqual(e.load()['stages']['dataset_build']['state'],'BLOCKED')
    def test_bad_artifact_does_not_succeed_and_retry_is_new(self):
        e=self.exp()
        with e:
            first=e.execute('compile',{}, {},'mock',lambda *a:dict(exit_code=0,evidence=[]),lambda r:require(False,'invalid output'))
            self.assertEqual(first['error']['kind'],'invalid_artifact');old=Path(e.load()['stages']['compile']['attempts'][0]).read_bytes()
            second=e.execute('compile',{}, {},'mock',lambda *a:dict(exit_code=0,evidence=[]),lambda r:None)
            self.assertEqual(second['attempt'],2);self.assertEqual(Path(e.load()['stages']['compile']['attempts'][0]).read_bytes(),old)
    def test_interrupt_recovery_and_lock(self):
        e=self.exp()
        with e:
            with self.assertRaises(PipelineError):
                with Experiment(e.path):pass
            p=e.path/'stages/compile/attempt_0001/attempt.json';write(p,dict(state='RUNNING',exit_code=None))
            m=e.load();m['stages']['compile']=dict(state='RUNNING',attempts=[str(p)]);e.save(m);e.recover()
            self.assertEqual(e.load()['stages']['compile']['state'],'FAILED');self.assertEqual(read(p)['error']['kind'],'interrupted')
    def test_keyboard_interrupt_retains_log(self):
        e=self.exp()
        def interrupted(*args):raise KeyboardInterrupt()
        with e:r=e.execute('compile',{}, {},'mock',interrupted,lambda r:None)
        self.assertEqual(r['exit_code'],130);self.assertTrue(Path(r['log']).exists())
    def test_no_overwrite_experiment(self):
        self.exp()
        with self.assertRaises(FileExistsError):self.exp()
    def test_manifest_training_guard(self):
        e=self.exp();m=e.load();m['stages']['training']['state']='RUNNING'
        with self.assertRaises(PipelineError):e.save(m)

class CLITests(unittest.TestCase):
    def test_commands_and_no_training(self):
        self.assertEqual(parser().parse_args(['dataset','build','--all-train']).dataset_command,'build')
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parser().parse_args(['train'])
    def test_dry_run_no_writes_or_process(self):
        with tempfile.TemporaryDirectory() as tmp,patch('subprocess.Popen') as spawn,contextlib.redirect_stdout(io.StringIO()) as out:
            code=main(['run','--references-root','/not-needed-for-dry-run','--workspace',tmp,'--experiment','e','--all-train','--dry-run'])
            self.assertEqual(code,0);self.assertFalse(list(Path(tmp).iterdir()));spawn.assert_not_called();self.assertFalse(json.loads(out.getvalue())['training'])
    def test_explicit_split_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['dataset','build','--canonical-root','x','--output-root','y','--dry-run']),2)
            self.assertEqual(main(['dataset','build','--canonical-root','x','--output-root','y','--val-count','2','--dry-run']),2)
    def test_invalid_input_records_failed_stage(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            rc=main(['dataset','build','--canonical-root',tmp+'/missing','--output-root',tmp+'/data','--workspace',tmp,'--experiment','bad','--all-train'])
            self.assertEqual(rc,2);m=read(Path(tmp)/'experiments/bad/experiment_manifest.json')
            self.assertEqual(m['stages']['dataset_build']['state'],'FAILED');self.assertEqual(m['stages']['training']['state'],'NOT_STARTED')
    def test_split_deterministic_and_explicit(self):
        cfg=dict(mode='count',val_count=2,seed=9)
        self.assertEqual(make_split(['c','a','b','d'],cfg),make_split(['d','b','a','c'],cfg))
        with self.assertRaises(PipelineError):make_split(['a','b'],dict(mode='explicit',train_episodes=['a'],val_episodes=['a']))

if __name__=='__main__':unittest.main()
