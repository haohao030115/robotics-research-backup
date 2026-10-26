import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from grasp_pipeline.cli import main
from grasp_pipeline.contracts import read,write
from grasp_pipeline.adapters import compiler_command

class CompilerAdapterTests(unittest.TestCase):
    def test_native_audit_command(self):
        cmd=compiler_command('/ref','/out',{})
        self.assertEqual(cmd[cmd.index('--mode')+1],'audit');self.assertEqual(cmd[cmd.index('--backend')+1],'native')
    def run_policy(self,policy,partial=True):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/'ref';ref.mkdir()
            def fake_compile(ref,out,cfg,log,child,fd):
                folder=out/'accepted';folder.mkdir(parents=True)
                write(folder/'training_manifest.json',{'episodes':[{'episode_id':'e','expert_valid':True,'path':'va_dataset'}]})
                (folder/'va_dataset').mkdir();(folder/'va_dataset/samples.npz').write_bytes(b'mock')
                rows=[dict(status='PASS',episode_id='e',output=str(folder))]
                if partial:rows.append(dict(status='FAIL',episode_id='bad',reason='fixture native failure',output=str(out/'bad')))
                batch=dict(PASS=1,FAIL=int(partial),SKIP=0,enumerated=len(rows),episodes=rows)
                write(out/'batch_summary.json',batch)
                return dict(path=str(out),exit_code=2 if partial else 0,evidence=[str(out/'batch_summary.json')],error_kind='partial_batch',error='fixture partial batch')
            def accepted(folder):return read(folder/'training_manifest.json')['episodes'][0],folder/'va_dataset/samples.npz'
            def fake_build(source,out,cfg):
                out.mkdir(parents=True)
                for n in ['COMPLETED.json','manifest.json','samples.npz','split.json','index.jsonl']:(out/n).write_text('{}')
                return {'dataset_id':out.name}
            with patch('grasp_pipeline.adapters.run_compiler',side_effect=fake_compile) as compiler,patch('canonical_va.production.accepted_episode',side_effect=accepted),patch('grasp_pipeline.dataset.build',side_effect=fake_build) as builder,patch('grasp_pipeline.dataset.validate'),contextlib.redirect_stdout(io.StringIO()):
                rc=main(['run','--references-root',str(ref),'--workspace',tmp,'--experiment','e','--all-train','--partial-policy',policy])
                m=read(root/'experiments/e/experiment_manifest.json')
                self.assertEqual(m['stages']['training']['state'],'NOT_STARTED');self.assertFalse(m['training_authorized'])
                if partial:
                    self.assertEqual(rc,2);self.assertEqual(m['stages']['compile']['state'],'FAILED')
                    self.assertEqual(m['stages']['dataset_build']['state'],'SUCCEEDED' if policy=='accepted' else 'BLOCKED')
                    self.assertEqual(builder.call_count,int(policy=='accepted'))
                else:self.assertEqual(rc,0);self.assertEqual(m['milestone'],'READY_FOR_TRAINING')
                if not partial or policy=='accepted':
                    rc2=main(['resume','--workspace',tmp,'--experiment','e'])
                    self.assertEqual(compiler.call_count,1);self.assertEqual(builder.call_count,1);self.assertEqual(rc2,rc)
    def test_partial_default_blocks(self):self.run_policy('error')
    def test_partial_explicit_consumes_without_faking_success(self):self.run_policy('accepted')
    def test_complete_batch_and_resume(self):self.run_policy('error',False)


class ProcessCleanupTests(unittest.TestCase):
    def test_descendant_with_new_session_is_terminated(self):
        import os,subprocess,sys,time
        from grasp_pipeline.adapters import terminate_tree
        with tempfile.TemporaryDirectory() as tmp:
            pidfile=Path(tmp)/'pid'
            source='import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)'
            p=subprocess.Popen([sys.executable,'-c',source,str(pidfile)],start_new_session=True)
            try:
                end=time.monotonic()+3
                while not pidfile.exists() and time.monotonic()<end:time.sleep(.01)
                self.assertTrue(pidfile.exists());pid=int(pidfile.read_text());terminate_tree(p)
                end=time.monotonic()+3
                while time.monotonic()<end:
                    stat=Path(f'/proc/{pid}/stat')
                    if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z':break
                    time.sleep(.01)
                else:self.fail('Native-style separate-session descendant survived interruption')
            finally:
                if p.poll() is None:terminate_tree(p)

if __name__=='__main__':unittest.main()
