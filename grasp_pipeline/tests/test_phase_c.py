import json,tempfile,unittest,contextlib,io
from pathlib import Path
from grasp_pipeline.cli import main
class PhaseCTests(unittest.TestCase):
 def test_audit_dry_run(self):
  with contextlib.redirect_stdout(io.StringIO()) as o:self.assertEqual(main(['dataset','audit','--dataset','/x','--dry-run']),0)
  self.assertTrue(json.loads(o.getvalue())['dry_run'])
 def test_train_requires_explicit_command(self):
  with contextlib.redirect_stdout(io.StringIO()) as o:self.assertEqual(main(['train','--dataset','/x','--output','/y','--dry-run']),0)
  self.assertTrue(json.loads(o.getvalue())['training'])
if __name__=='__main__':unittest.main()
