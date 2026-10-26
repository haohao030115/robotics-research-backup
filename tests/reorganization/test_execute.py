import json
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.execute import execute_stages


class ExecutionTests(unittest.TestCase):
    def test_stages_execute_with_logs_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            code = "from pathlib import Path; Path('stage_artifact.txt').write_text('ok')"
            commands = {"stage": [sys.executable, "-c", code]}
            self.assertEqual(execute_stages(run_id="smoke", family="test", commands=commands, output=output, root=Path.cwd()), 0)
            self.assertEqual(json.loads((output / "run_manifest.json").read_text())["status"], "SUCCEEDED")
            self.assertTrue((output / "stage.log").is_file())
            self.assertTrue((Path.cwd() / "stage_artifact.txt").is_file())
            (Path.cwd() / "stage_artifact.txt").unlink()

    def test_failed_stage_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            commands = {"stage": [sys.executable, "-c", "raise SystemExit(7)"]}
            self.assertEqual(execute_stages(run_id="failed", family="test", commands=commands, output=output, root=Path.cwd()), 7)
            self.assertEqual(json.loads((output / "run_manifest.json").read_text())["status"], "FAILED")
