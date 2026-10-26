import unittest
import numpy as np
import torch
from diagnostics.corrective import masked_corrective_loss,phase_match


class CorrectiveTests(unittest.TestCase):
    def test_unselected_learner_head_has_no_target_gradient(self):
        for branch,body,hand in [('body',True,False),('hand',False,True),('combined',True,True),('none',False,False)]:
            token=torch.ones(4,5,64,requires_grad=True);h=torch.ones(4,5,14,requires_grad=True)
            config={'huber_delta':.0625,'body_loss_weight':1,'hand_loss_weight':1}
            loss=masked_corrective_loss({'sonic_token':token,'hand_ref':h},torch.zeros_like(token),torch.zeros_like(h),2,branch,config)
            loss.backward()
            self.assertEqual(bool(token.grad[2:].abs().sum()>0),body)
            self.assertEqual(bool(h.grad[2:].abs().sum()>0),hand)
            self.assertGreater(float(token.grad[:2].abs().sum()),0)
            self.assertGreater(float(h.grad[:2].abs().sum()),0)

    def test_phase_match_uses_velocity_wrist_and_hand_as_well_as_body(self):
        expert={k:np.array([[0.,0.],[1.,1.],[2.,2.]]) for k in ['body_q','body_dq','eef_pos','hand_q']}
        actual={k:v[[1,2]] for k,v in expert.items()}
        np.testing.assert_array_equal(phase_match(actual,expert,{k:1 for k in expert}),[1,2])
        # q alone is ambiguous; other fields select the later phase.
        expert['body_q'][:]=0;actual['body_q'][:]=0
        np.testing.assert_array_equal(phase_match(actual,expert,{k:1 for k in expert}),[1,2])


if __name__=='__main__':unittest.main()
