import unittest
import numpy as np
import torch
from models import build_policy
from models.inference import ModelOnlyPolicy, split_action_chunk
from tests.common import config, stats, batch


class InferenceTests(unittest.TestCase):
    def test_current_state_normalization_and_inverse_action(self):
        cfg, statistics = config(), stats()
        statistics['state'] = {'mean': [2.0]*43, 'std': [4.0]*43}
        statistics['image'] = {'mean': [0.2]*3, 'std': [0.5]*3}
        statistics['action'] = {'mean': [5.0]*43, 'std': [3.0]*43}
        model = build_policy(cfg, statistics).eval()
        image, q, _ = batch(cfg)
        seen = {}
        hooks = [model.observation.state.register_forward_pre_hook(lambda m, args: seen.update(q=args[0].clone())),
                 model.observation.vision.register_forward_pre_hook(lambda m, args: seen.update(image=args[0].clone()))]
        normalized = model(image, q)
        torch.testing.assert_close(seen['q'], (q-2)/4)
        torch.testing.assert_close(seen['image'], (image-0.2)/0.5)
        for h in hooks:
            h.remove()
        interface = ModelOnlyPolicy(model)
        predicted = interface.predict(image[0], q[0, :29], q[0, 29:])
        torch.testing.assert_close(predicted, normalized[0]*3+5)
        body, hand = split_action_chunk(predicted)
        self.assertEqual(predicted.shape, (20, 43))
        self.assertEqual(body.shape, (20, 29))
        self.assertEqual(hand.shape, (20, 14))

    def test_numpy_rgb_resize_and_flow_single_sample(self):
        cfg = config()
        cfg['model_type'] = 'flow'
        interface = ModelOnlyPolicy(build_policy(cfg, stats()))
        rgb = np.full((24, 32, 3), 128, dtype=np.uint8)
        image, _, _, single = interface.prepare(rgb, np.zeros(29), np.zeros(14))
        self.assertEqual(image.shape, (1, 3, 16, 16))
        self.assertAlmostEqual(image.mean().item(), 128/255, places=6)
        output = interface.predict(rgb, np.zeros(29), np.zeros(14), steps=4)
        self.assertTrue(single)
        self.assertEqual(output.shape, (20, 43))
        self.assertTrue(torch.isfinite(output).all())

    def test_invalid_rgb_and_nonfinite_state_rejected(self):
        interface = ModelOnlyPolicy(build_policy(config(), stats()))
        with self.assertRaisesRegex(ValueError, 'RGB'):
            interface.predict(torch.full((3, 16, 16), 2.0), torch.zeros(29), torch.zeros(14))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            interface.predict(torch.zeros(3, 16, 16), torch.full((29,), float('nan')), torch.zeros(14))
