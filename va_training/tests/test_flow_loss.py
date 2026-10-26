import unittest
from unittest.mock import patch
import torch
from models.flow_scheduler import FlowScheduler
from losses.action import action_loss


class FlowLossTests(unittest.TestCase):
    def test_straight_interpolation_and_target_velocity(self):
        x1 = torch.full((2, 20, 43), 3.0)
        x0 = torch.ones_like(x1)
        t = torch.tensor([0.0, 0.5])
        with patch('torch.randn_like', return_value=x0), patch('torch.rand', return_value=t):
            xt, result_t, velocity = FlowScheduler().training_pair(x1)
        self.assertTrue(torch.equal(xt[0], x0[0]))
        self.assertTrue(torch.equal(xt[1], torch.full_like(x1[1], 2)))
        self.assertTrue(torch.equal(velocity, x1-x0))
        loss, parts = action_loss(velocity, x1-x0, 'mse')
        self.assertEqual(loss.item(), 0)

    def test_body_hand_dimension_weighting(self):
        prediction = torch.ones(2, 20, 43)
        prediction[..., 29:] = 2
        loss, parts = action_loss(prediction, torch.zeros_like(prediction), 'mse')
        self.assertAlmostEqual(loss.item(), (29+14*4)/43, places=6)
        self.assertEqual(parts['body'].item(), 1)
        self.assertEqual(parts['hand'].item(), 4)
