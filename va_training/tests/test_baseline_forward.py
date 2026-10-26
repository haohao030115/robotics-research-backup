import unittest
import torch
from models import build_policy
from tests.common import config, stats, batch


class BaselineTests(unittest.TestCase):
    def test_forward_backward_spatial_tokens_and_predict(self):
        cfg = config()
        model = build_policy(cfg, stats())
        image, q, target = batch(cfg)
        tokens = model.observation.vision(image)
        self.assertGreater(tokens.shape[1], 1)
        output = model(image, q)
        self.assertEqual(output.shape, (2, 20, 43))
        (output-target).abs().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        prediction = model.predict(image, q[:, :29], q[:, 29:])
        self.assertEqual(prediction.shape, output.shape)

    def test_optional_language_has_gradients(self):
        cfg = config()
        cfg['model']['language_enabled'] = True
        model = build_policy(cfg, stats())
        image, q, _ = batch(cfg)
        model(image, q, ['red cube', 'blue cube']).square().mean().backward()
        self.assertGreater(model.observation.language.embedding.weight.grad.abs().sum().item(), 0)
