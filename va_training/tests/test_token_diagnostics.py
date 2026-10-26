import os
import unittest
import numpy as np
from diagnostics.chunk_buffer import TokenChunkBuffer
from diagnostics.control import native_gravity


class DiagnosticTests(unittest.TestCase):
    def test_chunk_preserves_timestamped_hand_pair_and_drops_expired_entries(self):
        b = TokenChunkBuffer()
        prediction = dict(token=np.arange(5)[:,None]*np.ones((5,64)),
                          hand=np.arange(5)[:,None]*np.ones((5,14)),
                          execution_index=2, anchor=0)
        b.add(prediction, 3)
        self.assertEqual(b.expired, 2)
        self.assertIsNone(b.take(3))
        entry = b.take(4)
        self.assertEqual(entry['execution_index'], 4)
        self.assertEqual(entry['chunk_index'], 2)
        np.testing.assert_array_equal(entry['hand'], np.full(14,2))
        np.testing.assert_array_equal(entry['token'], np.full(64,2))
        replacement = dict(prediction, execution_index=5)
        b.add(replacement,4)
        self.assertEqual(b.overwrites,2)
        np.testing.assert_array_equal(b.take(5)['hand'],np.zeros(14))

    def test_native_zero_history_gravity_differs_from_valid_identity(self):
        np.testing.assert_allclose(native_gravity([[0,0,0,0],[1,0,0,0]]),[[0,0,1],[0,0,-1]])

    @unittest.skipUnless(os.environ.get('VA_V2_REAL_CONFIG'),'Actual release decoder integration')
    def test_decoder_reconstruction_matches_logged_next_previous_action(self):
        from utils.v2_config import load_v2_config
        from datasets.tokens import load_token_episodes
        from diagnostics.control import ReleaseDecoder
        c=load_v2_config(os.environ['VA_V2_REAL_CONFIG'])
        episodes=load_token_episodes(c['data']); e=episodes[0]
        decoder=ReleaseDecoder(c['sonic'])
        for frame in [0,1,10,400]:
            action=decoder.decode(e,frame,e.tokens[frame])
            index=int(e.control_metadata[frame,0])
            np.testing.assert_allclose(action,decoder.logs(e)['action'][index+1],atol=1e-5,rtol=0)


if __name__=='__main__': unittest.main()
