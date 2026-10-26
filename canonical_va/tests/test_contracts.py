import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from canonical_va.schema import validate, joint_orders, UNITS
from canonical_va.common import dump, table, jsonl
from canonical_va.compiler import plan_samples, compile_samples


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        bn,hn=joint_orders(); self.a=dict(time_s=np.arange(20)/50,body_ref_q=np.zeros((20,29)),body_ref_dq=np.zeros((20,29)),
            hand_ref_q=np.zeros((20,14)),root_ref_pos=np.zeros((20,3)),root_ref_quat=np.tile([1.,0,0,0],(20,1)))
        self.m=dict(episode_id='test',producer_type='kimodo',reference_complete=True,reference_hz=50,body_joint_names=bn,
            hand_joint_names=hn,quaternion_convention='wxyz',root_pose_frame='world',units=UNITS,case={'id':'test','block_xy':[.4,-.15]})
        dump(self.root/'provenance.json',{}); dump(self.root/'reference_validation.json',{'reference_schema_valid':True})
    def tearDown(self): self.tmp.cleanup()
    def save(self):
        np.savez(self.root/'reference.npz',**self.a); dump(self.root/'metadata.json',self.m)
    def test_three_producers(self):
        for producer in ('kimodo','rule_based','ardy'):
            self.m['producer_type']=producer; self.save(); self.assertTrue(validate(self.root)[2]['reference_schema_valid'])
    def test_schema_rejections(self):
        original_a=copy.deepcopy(self.a); original_m=copy.deepcopy(self.m)
        def bad_nan(): self.a['body_ref_q'][5,2]=np.nan
        def bad_time(): self.a['time_s'][5]=self.a['time_s'][4]
        def bad_names(): self.m['body_joint_names']=self.m['body_joint_names'][::-1]
        def bad_shape(): self.a['hand_ref_q']=np.zeros((20,13))
        def bad_complete(): self.m['reference_complete']=False
        def bad_units(): self.m['units']={}
        def bad_quat(): self.a['root_ref_quat'][4]=0
        def bad_hz(): self.m['reference_hz']=49
        def bad_missing(): del self.a['body_ref_dq']
        def bad_dtype(): self.a['body_ref_dq']=np.full((20,29),'0')
        for mutate in (bad_nan,bad_time,bad_names,bad_shape,bad_complete,bad_units,bad_quat,bad_hz,bad_missing,bad_dtype):
            self.a=copy.deepcopy(original_a); self.m=copy.deepcopy(original_m); mutate(); self.save()
            with self.subTest(mutate=mutate.__name__),self.assertRaisesRegex(ValueError,'REJECT_REFERENCE'): validate(self.root)
    def truth(self):
        (self.root/'rgb').mkdir(); dump(self.root/'metadata.json',{'episode_id':'test'})
        obs=[]; refs=[]; enc=[]
        for t in range(20):
            phase='standby' if t<2 else 'inactive' if t==12 else 'active'
            ids=dict(observation_id=f'o{t*19}',control_tick_id=f't{t*7}',decision_id=f'd{t*3}',reference_frame_id=f'r{t*11}')
            obs.append(dict(**ids,phase=phase))
            refs.append(dict(**ids,phase=phase,reference_index=t,hand_ref_id=f'h{t*13}',hand_ref_q=np.full(14,t)))
            enc.append(dict(**ids,token_id=f'z{t*17}',current_post_fsq_64d_token=np.full(64,t)))
        # Deliberately unrelated physical row ordering among all tables.
        table(self.root/'observations.npz',obs[::-1]); table(self.root/'references.npz',refs[3:]+refs[:3]); table(self.root/'sonic_encoder.npz',enc[::-1])
        jsonl(self.root/'rgb/frames.jsonl',[dict(rgb_frame_id=f'i{t}',observation_id=f'o{t*19}',path=f'rgb/{t}.png') for t in range(20)])
    def test_mapping_not_row_join(self):
        self.truth(); maps=plan_samples(self.root)
        self.assertEqual([s['anchor_execution_index'] for s in maps],[2,3,4,5,13])
        s=maps[0]; self.assertEqual(s['future_execution_indices'],[4,5,6,7,8])
        self.assertEqual(s['state_history_observation_ids'],['o38']*3)
        self.assertEqual(s['history_valid_mask'],[False,False,True])
        self.assertEqual(s['body_token_ids'],[f'z{x*17}' for x in range(4,9)])
        self.assertEqual(s['hand_ref_ids'],[f'h{x*13}' for x in range(4,9)])
    def test_shifted_future_rejected(self):
        self.truth(); p=self.root/'sonic_encoder.npz'
        with np.load(p) as z: a={k:z[k] for k in z.files}
        a['decision_id']=np.roll(a['decision_id'],1); np.savez(p,**a)
        with self.assertRaisesRegex(ValueError,'future decision mismatch'): plan_samples(self.root)
    def test_failed_episode_cannot_export(self):
        with self.assertRaisesRegex(ValueError,'expert_valid'): compile_samples(self.root,self.root/'out',{'expert_valid':False})
    def test_camera_mismatch(self):
        from canonical_va.runtime import validate_camera
        with self.assertRaisesRegex(ValueError,'CAMERA_MISMATCH'): validate_camera({'fov':74},{'fov':75})
    def test_schema_failure_never_replays(self):
        from canonical_va.pipeline import run
        self.m['reference_complete']=False; self.save()
        with patch('canonical_va.replay.replay') as replay:
            self.assertEqual(run(self.root,self.root/'rejected'),2); replay.assert_not_called()

if __name__=='__main__': unittest.main()
