import unittest
import torch
from models import build_policy
from tests.common import config, stats, batch


class FlowTests(unittest.TestCase):
    def test_forward_backward_all_parameters(self):
        cfg = config()
        cfg['model_type'] = 'flow'
        model = build_policy(cfg, stats())
        image, q, target = batch(cfg)
        xt, t, velocity = model.scheduler.training_pair(target)
        prediction = model(image, q, xt, t)
        self.assertEqual(prediction.shape, (2, 20, 43))
        loss = (prediction-velocity).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_language_context_token(self):
        cfg = config()
        cfg['model_type'] = 'flow'
        cfg['model']['language_enabled'] = True
        model = build_policy(cfg, stats())
        image, q, target = batch(cfg)
        xt, t, velocity = model.scheduler.training_pair(target)
        model(image, q, xt, t, ['red cube', 'blue cube']).square().mean().backward()
        self.assertGreater(model.observation.language.embedding.weight.grad.abs().sum().item(), 0)
