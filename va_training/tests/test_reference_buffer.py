import unittest
import numpy as np
from scripts.reference_buffer import FutureReferenceBuffer


class ReferenceBufferTests(unittest.TestCase):
    def chunk(self,anchor):
        return np.arange(anchor*43,(anchor+50)*43,dtype=np.float32).reshape(50,43)/1000

    def test_slow_prediction_preserves_consumed_history_and_extends_lookahead(self):
        buffer=FutureReferenceBuffer()
        start,packet=buffer.replenish(0,self.chunk(0))
        self.assertEqual(start,0)
        self.assertGreater(len(packet)-46,50)
        np.testing.assert_array_equal(packet[:50],self.chunk(0))
        # Simulate inference taking 12 native ticks, longer than the former
        # four-tick window. The cursor continues on already buffered values.
        for frame in range(12):
            advanced,error=buffer.consume(self.chunk(0)[frame,:29])
            self.assertTrue(advanced)
            np.testing.assert_array_equal(buffer.hand(),self.chunk(0)[frame,29:])
        protected=buffer.q[:buffer.cursor+buffer.COMMIT_LEAD].copy()
        old_end=buffer.end
        start,packet=buffer.replenish(5,self.chunk(5)+.01)
        self.assertGreater(start,0)
        self.assertGreater(buffer.end,old_end)
        self.assertEqual(buffer.cursor,11)
        np.testing.assert_array_equal(buffer.q[:len(protected)],protected)
        self.assertGreater(buffer.end-buffer.cursor,46)

    def test_hand_does_not_advance_when_native_body_stalls(self):
        buffer=FutureReferenceBuffer()
        buffer.replenish(0,self.chunk(0))
        buffer.consume(self.chunk(0)[0,:29])
        target=buffer.hand().copy()
        advanced,_=buffer.consume(self.chunk(0)[0,:29])
        self.assertFalse(advanced)
        self.assertEqual(buffer.cursor,0)
        np.testing.assert_array_equal(buffer.hand(),target)
        with self.assertRaisesRegex(RuntimeError,'discontinuity'):
            buffer.consume(self.chunk(0)[4,:29])

    def test_expired_chunks_do_not_reset_cursor_or_overwrite_history(self):
        buffer=FutureReferenceBuffer()
        buffer.replenish(0,self.chunk(0))
        for frame in range(48):buffer.consume(self.chunk(0)[frame,:29])
        before=buffer.q.copy()
        self.assertIsNone(buffer.replenish(0,self.chunk(0)+1))
        self.assertEqual(buffer.cursor,47)
        np.testing.assert_array_equal(buffer.q,before)
        self.assertEqual(buffer.stale_chunks,1)


if __name__=='__main__':unittest.main()
