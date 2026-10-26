import tempfile
from pathlib import Path
import unittest
import torch
from models import build_policy
from models.flow_scheduler import FlowScheduler
from utils.normalization import Normalization
from utils.checkpoint import save_checkpoint, load_policy
from tests.common import config, stats, batch


class SamplingTests(unittest.TestCase):
    def test_euler_constant_velocity_endpoint(self):
        initial = torch.zeros(2, 20, 43)
        output = FlowScheduler().integrate(lambda x, t: torch.ones_like(x)*2,
                                          initial.shape, initial.device, initial.dtype, 10, initial)
        torch.testing.assert_close(output, torch.full_like(initial, 2))

    def test_sampling_inverse_normalization(self):
        cfg = config()
        cfg['model_type'] = 'flow'
        statistics = stats()
        statistics['action'] = {'mean': [2.0]*43, 'std': [3.0]*43}
        model = build_policy(cfg, statistics).eval()
        image, q, target = batch(cfg)
        noise = torch.zeros_like(target)
        normalized = model.sample_normalized(image, q, initial_noise=noise)
        output = model.predict(image, q[:, :29], q[:, 29:], initial_noise=noise)
        self.assertEqual(output.shape, (2, 20, 43))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, normalized*3+2)

    def test_checkpoint_roundtrip(self):
        cfg = config()
        model = build_policy(cfg, stats()).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        image, q, _ = batch(cfg)
        expected = model(image, q)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.pt'
            save_checkpoint(path, model, optimizer, scheduler, 3, 20, cfg, stats())
            loaded, checkpoint = load_policy(path, torch.device('cpu'))
            torch.testing.assert_close(loaded(image, q), expected)
            self.assertEqual(checkpoint['global_step'], 20)
            self.assertEqual(checkpoint['epoch'], 3)
            self.assertTrue(all(k in checkpoint for k in ['model', 'optimizer', 'scheduler',
                                                         'config', 'normalization_stats']))
