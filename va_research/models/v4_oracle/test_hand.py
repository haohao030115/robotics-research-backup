import unittest,tempfile,json
from pathlib import Path
import numpy as np
from .hand import OracleHand,relative_pose,angle

class OracleTests(unittest.TestCase):
    def test_geometry_and_one_shot(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);q=[1.,0,0,0];w=np.r_[[1,0,0],q];c=np.r_[[0,0,0],q]
            (p/'oracle_calibration.json').write_text(json.dumps(dict(relative_xyz_min=[.9,-.1,-.1],relative_xyz_max=[1.1,.1,.1],relative_quaternion_center_wxyz=q,orientation_limit_rad=.1)))
            traj=np.array([[0,0],[.2,-.3],[.5,-.7]])
            np.savez(p/'expert_hand.npz',trajectory=traj)
            h=OracleHand(p);far=w.copy();far[0]=2
            for t in range(3):np.testing.assert_array_equal(h.step(t,far,c),traj[0])
            rotated=w.copy();rotated[3:]=[0,1,0,0]
            np.testing.assert_array_equal(h.step(3,rotated,c),traj[0]);self.assertIsNone(h.trigger_tick)
            np.testing.assert_array_equal(h.step(4,w,c),traj[0]);self.assertEqual(h.trigger_tick,4)
            np.testing.assert_array_equal(h.step(5,far,c),traj[1]);np.testing.assert_array_equal(h.step(6,far,c),traj[2])
            np.testing.assert_array_equal(h.step(7,w,c),traj[2]);self.assertTrue(h.report()['closure_completed'])
            self.assertIsNone(OracleHand(p).trigger_tick)
    def test_relative_rotation(self):
        q=np.array([np.sqrt(.5),0,0,np.sqrt(.5)]);cube=np.r_[[3,4,0],q];wrist=np.r_[[3,5,0],q]
        xyz,rel=relative_pose(wrist,cube);np.testing.assert_allclose(xyz,[1,0,0],atol=1e-12)
        self.assertLess(float(angle(rel,[1,0,0,0])),1e-7)
if __name__=='__main__':unittest.main()
