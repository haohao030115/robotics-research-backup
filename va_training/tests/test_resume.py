import copy
from pathlib import Path
import tempfile
import unittest
import torch
from scripts.make_synthetic import create_fixture
from tests.common import config
from trainers.training import run_training


class ResumeTests(unittest.TestCase):
    def test_mid_epoch_resume_matches_uninterrupted_baseline_and_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = create_fixture(directory/'data')
            for kind in ('baseline', 'flow'):
                with self.subTest(kind=kind):
                    cfg = config()
                    cfg['model_type'] = kind
                    cfg['data']['manifest'] = str(manifest)
                    cfg['training'].update(device='cpu', num_workers=0, epochs=2, max_steps=4,
                                           batch_size_per_rank=4, log_every=4, lr=0.001,
                                           validation_sampling_batches=1, output_dir=str(directory/(kind+'_full')))
                    full = run_training(copy.deepcopy(cfg))
                    cfg['training']['output_dir'] = str(directory/(kind+'_resume'))
                    stopped = run_training(copy.deepcopy(cfg), stop_after_steps=1)
                    snapshot = torch.load(stopped['checkpoint'], weights_only=True)
                    self.assertEqual(snapshot['resume_progress']['batch'], 1)
                    self.assertEqual(snapshot['resume_progress']['epoch'], 0)
                    self.assertEqual(len(snapshot['rng_states']), 1)
                    resumed = run_training(copy.deepcopy(cfg), resume=stopped['checkpoint'])
                    expected = torch.load(full['checkpoint'], weights_only=True)
                    actual = torch.load(resumed['checkpoint'], weights_only=True)
                    self.assertEqual(actual['global_step'], 4)
                    self.assertEqual(actual['scheduler'], expected['scheduler'])
                    for key in expected['model']:
                        torch.testing.assert_close(actual['model'][key], expected['model'][key], atol=0, rtol=0)
