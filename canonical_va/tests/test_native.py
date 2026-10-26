import unittest
from canonical_va.native_forensic import verify_receipt_edge
from canonical_va.native_replay import DECISION_DTYPE
from canonical_va.common import read
from canonical_va.native_build import BUILD_ROOT


class NativeContracts(unittest.TestCase):
    def test_standby_velocity_is_zero_in_original_native(self):
        import numpy as np
        from canonical_va.native_forensic import expected_native_reference
        arrays={'body_ref_q':np.ones((5,29)), 'body_ref_dq':np.full((5,29),.003)}
        frames=np.zeros(10,dtype=int)
        self.assertTrue(np.all(expected_native_reference(arrays,'body_ref_dq',frames,False)==0))
        self.assertTrue(np.all(expected_native_reference(arrays,'body_ref_dq',frames,True)!=0))
        self.assertTrue(np.all(expected_native_reference(arrays,'body_ref_q',frames,False)==1))

    def test_binary_record_layout(self):
        self.assertEqual(DECISION_DTYPE.itemsize,10*8+(1762+994+64+29+29+5*29)*4)
    def test_real_receipt_identity_required(self):
        pubs={21:dict(command_id='command:7',monotonic_ns=100)}
        receipts={21:dict(command_number=7,receipt_time=105)}
        verify_receipt_edge('command:7',7,21,pubs,receipts,110)
        for command,number,pub,rs,end in [('command:8',7,21,receipts,110),('command:7',8,21,receipts,110),
              ('command:7',7,22,receipts,110),('command:7',7,21,{},110),('command:7',7,21,receipts,104)]:
            with self.subTest(number=number,pub=pub,end=end),self.assertRaises(ValueError):
                verify_receipt_edge(command,number,pub,pubs,rs,end)
    def test_production_entry_has_no_sync_fallback(self):
        from canonical_va.replay import replay
        self.assertEqual(replay.__module__,'canonical_va.native_replay')
    def test_instrumentation_does_not_change_control_formulas(self):
        patch=(BUILD_ROOT/'instrumentation.patch').read_text()
        removed=[x for x in patch.splitlines() if x.startswith('-') and not x.startswith('---')]
        # Existing lines replaced only by themselves plus hook insertions.
        added=[x[1:] for x in patch.splitlines() if x.startswith('+') and not x.startswith('+++')]
        for line in removed:
            self.assertIn(line[1:],added)

if __name__=='__main__':unittest.main()
