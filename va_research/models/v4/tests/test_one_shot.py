import unittest,numpy as np
from va_research.models.v4.one_shot import OneShotHand
class TestOneShot(unittest.TestCase):
 def test_disabled_equivalence(self):
  w=OneShotHand(False);x=np.arange(14.);np.testing.assert_array_equal(w.apply(0,x),x)
 def test_latch_direction_reset(self):
  w=OneShotHand(True,confirm_replans=2,velocity_limit=.2)
  x=np.zeros(14);assert w.apply(0,x)[7]==0
  x[7:]=.4;w.prime(np.stack([np.zeros(14),x]));w.apply(5,np.zeros(14));w.apply(6,x);y=w.apply(10,x);assert w.trigger_tick==10
  z=x.copy();z[7:]=-.4;out=w.apply(15,z);assert np.all(out[7:]>=y[7:]);w.reset();assert w.state==w.OPEN
 def test_hold(self):
  w=OneShotHand(True,confirm_replans=1,velocity_limit=1)
  x=np.zeros(14);x[7:]=1;w.prime(np.stack([np.zeros(14),x]));w.apply(0,np.zeros(14));y=w.apply(1,x);assert w.state==w.HOLD;z=x.copy();z[7:]=0;np.testing.assert_array_equal(w.apply(2,z),y)
if __name__=='__main__':unittest.main()
