import unittest
import numpy as np
import torch
from v4.model import V4Policy
from v4.losses import objectives
from v4.runtime import PlanBuffer, UncoveredTick
from v4.dataset import gravity, progress_labels


class V4Tests(unittest.TestCase):
    def test_shapes_and_no_hand_leakage(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        m = V4Policy().eval()
        rgb, b, h = torch.rand(2,3,128,128), torch.rand(2,68), torch.rand(2,28)
        p, changed = m(rgb,b,h), m(rgb,b,h+2)
        self.assertEqual(p['body_tokens'].shape, (2,10,64))
        self.assertEqual(p['hand_ref'].shape, (2,50,14))
        for k in ('body_tokens','phase_logits','closure_progress'):
            self.assertTrue(torch.equal(p[k],changed[k]), k)
        self.assertFalse(torch.equal(p['hand_ref'],changed['hand_ref']))
        loss, terms = objectives(p, dict(body_tokens=torch.zeros(2,10,64), hand_ref=torch.zeros(2,50,14),phase=torch.zeros(2,dtype=torch.long),progress=torch.zeros(2)))
        loss.backward()
        for name in ('visual','body_query','grasp_query','hand_state'):
            self.assertGreater(sum(float(x.grad.abs().sum()) for x in getattr(m,name).parameters()),0)

    def test_overlap_and_gap(self):
        b,h = np.zeros((10,64)),np.zeros((50,14))
        s=PlanBuffer();s.add(0,b,h,0)
        for tick in range(7):
            r=s.take(tick,(tick//5)*5)
            self.assertEqual(r['intended_tick'],tick)
            self.assertEqual(r['overlap_fallback'],tick>=5)
        s.add(5,b+1,h+1,7)
        r=s.take(7,5)
        self.assertEqual(r['prediction_slot'],2)
        self.assertEqual(r['plan_generation_id'],1)
        for tick in range(8,15):s.take(tick,(tick//5)*5)
        with self.assertRaises(UncoveredTick):s.take(15,15)
        self.assertEqual(s.counts['uncovered_ticks'],1)
        self.assertEqual(s.counts['deadline_misses'],3)

    def test_no_startup_fabrication(self):
        s=PlanBuffer();s.add(2,np.zeros((10,64)),np.zeros((50,14)),0)
        with self.assertRaises(UncoveredTick):s.take(0,0)

    def test_duplicate_and_nonfinite(self):
        s=PlanBuffer();b=np.zeros((10,64));h=np.zeros((50,14))
        s.add(0,b,h,0);s.take(0,0)
        with self.assertRaises(RuntimeError):s.take(0,0)
        b[0,0]=np.nan
        with self.assertRaises(ValueError):s.add(5,b,h,5)

    def test_labels_and_gravity(self):
        np.testing.assert_allclose(gravity(np.array([1.,0,0,0])),[0,0,-1])
        names=['left_x']*7+['right_x']*7
        refs=np.zeros((3,14));refs[1,7:]=.5;refs[2,7:]=1
        p,phase,moving,_,_=progress_labels(refs,names)
        np.testing.assert_allclose(p,[0,.5,1]);np.testing.assert_equal(phase,[0,1,2])
        self.assertEqual(moving.sum(),7)


if __name__ == '__main__':unittest.main()
