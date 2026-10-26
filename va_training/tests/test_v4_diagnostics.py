import unittest
import numpy as np
from v4.diagnostics.plan_consistency import combine, consistency


class PlanDiagnosticTests(unittest.TestCase):
    def plans(self):
        start=np.array([0,5,10])
        return dict(start_tick=start,hand_ref=np.repeat((start[:,None]+np.arange(50))[...,None],14,axis=2).astype(float))

    def test_same_absolute_tick_overlap(self):
        p=self.plans();_,over,_=consistency(p,np.ones(14,bool))
        np.testing.assert_array_equal(over,0)
        self.assertEqual(over.shape,(2,45,14))

    def test_ensemble_never_mixes_ticks(self):
        p=self.plans()
        for decay in (None,2.,.7):np.testing.assert_allclose(combine(p,12,decay,np.array([0,5,10])),12)

    def test_unavailable_and_future_plans_excluded(self):
        p=self.plans();p['hand_ref'][1:]+=100
        np.testing.assert_allclose(combine(p,5,2.,np.array([0,6,10])),5)
        np.testing.assert_allclose(combine(p,5,None,np.array([0,5,10])),105)
        np.testing.assert_allclose(combine(p,5,2.,np.array([0,5,10]),exclude=1),5)
        with self.assertRaises(ValueError):combine(p,60,.7,np.array([0,5,10]))


if __name__=='__main__':unittest.main()
