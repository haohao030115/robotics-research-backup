import numpy as np
from va_training.v4_1.continuity import HandContinuityLatch,HandState
def test_latch_confirmation_and_no_reopen():
    l=HandContinuityLatch(np.zeros(2),confirm_count=2,velocity_limit=.2)
    assert l.step(np.ones((3,2)), -1, 1)[1]=='OPEN'
    assert l.step(np.ones((3,2)), 1, 1)[1]=='OPEN'
    y,s=l.step(np.ones((3,2)), 1, 1); assert s in ('CLOSING','HOLD')
    y2,s=l.step(np.zeros((3,2)), -1, 0); assert np.all(y2>=y-1e-9) and s in ('CLOSING','HOLD')
def test_reset():
    l=HandContinuityLatch(np.zeros(2)); l.step(np.ones((2,2)),1,1); l.reset(); assert l.state==HandState.OPEN
