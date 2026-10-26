"""Opt-in CPU integration tests on existing artifacts. Never launch physics or inference."""
import json
import os
from pathlib import Path
import tempfile
import shutil
import unittest
import numpy as np
from grasp_pipeline.dataset import build,baseline_class,source_info,validate
from grasp_pipeline.contracts import read,write,PipelineError

FULL=os.environ.get('GRASP_FULL_FIXTURE')
COMPACT=os.environ.get('GRASP_COMPACT_FIXTURE')

@unittest.skipUnless(FULL,'Set GRASP_FULL_FIXTURE to an existing accepted full canonical episode')
class SavedV4Tests(unittest.TestCase):
    def test_exact_all_tensors_and_order(self):
        baseline=baseline_class()(Path(FULL)/'training_manifest.json')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'dataset';m=build(FULL,out,dict(model='v4',split={'mode':'all_train'}))
            with np.load(out/'samples.npz',allow_pickle=False) as z:
                self.assertEqual(set(z.files),set(baseline.data))
                for k,t in baseline.data.items():
                    a=t.numpy();self.assertEqual(z[k].dtype,a.dtype);np.testing.assert_array_equal(z[k],a,err_msg=k)
            index=[json.loads(x) for x in (out/'index.jsonl').read_text().splitlines()]
            self.assertEqual([r['canonical_sample_index'] for r in index],baseline.keep.tolist())
            self.assertEqual([r['hand_start_row'] for r in index],baseline.starts.tolist())
            self.assertEqual([r['mapping'] for r in index],[baseline.mapping[i] for i in baseline.keep])
            self.assertEqual(m['sample_count'],len(baseline.keep));validate(out)
            with self.assertRaises(PipelineError):build(FULL,out,dict(split={'mode':'all_train'}))
    def test_left_reject_and_explicit_exclusion(self):
        full=Path(FULL)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);left=root/'left';left.mkdir()
            # Synthetic metadata-only incompatibility fixture; canonical source bytes remain untouched.
            for p in full.iterdir():
                if p.name=='va_dataset':shutil.copytree(p,left/p.name)
                elif p.name!='training_manifest.json':(left/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
            m=read(full/'training_manifest.json');m['episodes'][0]['task_arm']='left';write(left/'training_manifest.json',m)
            with self.assertRaisesRegex(PipelineError,'UNSUPPORTED_V4_TASK_ARM'):source_info(left)
            (root/'right').symlink_to(full.resolve(),target_is_directory=True)
            out=root/'result'
            m=build(root,out,dict(split={'mode':'all_train'},incompatible_policy='exclude'))
            self.assertEqual(len(m['episodes']),1);self.assertEqual(len(m['excluded']),1)
            self.assertIn('UNSUPPORTED_V4_TASK_ARM',m['excluded'][0]['reason'])

@unittest.skipUnless(COMPACT,'Set GRASP_COMPACT_FIXTURE to compact accepted production output')
class CompactTests(unittest.TestCase):
    def test_compact_rejected(self):
        with self.assertRaisesRegex(PipelineError,'INSUFFICIENT_CANONICAL_RETENTION'):source_info(Path(COMPACT))

if __name__=='__main__':unittest.main()
