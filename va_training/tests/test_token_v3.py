import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from v3.data import V3Dataset, FIELDS, WIDTHS, targets, preprocess_uint8
from v3.model import V3Policy
from v3.runtime import TwoTickBuffer
from v3.training import losses


class V3Tests(unittest.TestCase):
    def test_current_only_alignment_and_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'images.npz'
            np.savez(p,head_rgb=np.zeros((4,16,16,3),dtype=np.uint8))
            state={k:np.tile(np.arange(6)[:,None],(1,w)).astype(np.float32) for k,w in zip(FIELDS,WIDTHS)}
            state['previous_action']=np.full((6,29),np.nan)
            e=SimpleNamespace(id='a',image_path=p,anchors=np.array([0,1,3,4]),times=np.arange(6)/50,
                rgb_times=np.array([0,1,3,4])/50,state=state,
                tokens=np.tile(np.arange(6)[:,None],(1,64)).astype(np.float32),
                hand_ref=np.tile(np.arange(6)[:,None],(1,14)).astype(np.float32))
            d=V3Dataset([e]);self.assertEqual(len(d),3)
            s=d[1];self.assertEqual(s['proprio'].shape,(96,));self.assertTrue((s['proprio']==1).all())
            self.assertEqual(s['execution_indices'].tolist(),[2,3]);self.assertEqual(s['target_sonic_token'][:,0].tolist(),[2,3])
            self.assertEqual(s['target_hand_ref'][:,0].tolist(),[2,3]);self.assertNotIn('history_mask',s)
            self.assertEqual(s['rgb_timestamp'],s['proprio_timestamp'])

    def test_no_cross_sample_memory_and_both_heads_train(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        model=V3Policy();image=torch.randn(2,3,128,128);state=torch.randn(2,96)
        first=model(image,state);model(torch.randn_like(image),torch.randn_like(state));second=model(image,state)
        torch.testing.assert_close(first['sonic_token'],second['sonic_token'])
        self.assertEqual(first['sonic_token'].shape,(2,2,64));self.assertEqual(first['hand_ref'].shape,(2,2,14))
        b,h=losses(first,torch.ones(2,2,64),torch.ones(2,2,14),dict(body_loss='huber',huber_delta=.0625))
        (b+h).backward()
        for module in (model.visual[0],model.state[0],model.cross.attention.out_proj,model.body[-1],model.hand[-1]):
            self.assertGreater(module.weight.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):model(image,torch.randn(2,3,96))
        self.assertFalse(any(isinstance(m,(torch.nn.GRU,torch.nn.LSTM)) for m in model.modules()))

    def test_exact_slots_reset_and_late_chunks(self):
        b=TwoTickBuffer()
        def prediction(r):return dict(anchor=r,execution_index=r+1,token=np.stack([np.full(64,r+1),np.full(64,r+2)]),hand=np.stack([np.full(14,r+1),np.full(14,r+2)]))
        b.add(prediction(-1),-1)
        for tick in (0,1):
            s=b.take(tick);self.assertEqual(s['execution_index'],tick);self.assertTrue((s['token']==tick).all());self.assertTrue((s['hand']==tick).all())
        b.add(prediction(1),2);self.assertEqual(b.expired,1);self.assertIsNone(b.take(2));self.assertEqual(b.take(3)['chunk_index'],1)
        with self.assertRaises(ValueError):b.add(prediction(5),3)
        np.testing.assert_array_equal(targets(-1),[0,1])

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            m=V3Policy().eval();p=Path(folder)/'model.pt';torch.save(m.state_dict(),p)
            n=V3Policy().eval();n.load_state_dict(torch.load(p,weights_only=True))
            rgb=torch.zeros(1,3,128,128);q=torch.zeros(1,96)
            torch.testing.assert_close(m(rgb,q)['hand_ref'],n(rgb,q)['hand_ref'])


class V3CommandTimingTests(unittest.TestCase):
    def test_dds_matching_uses_values_and_never_nearest_time_only(self):
        from unittest.mock import patch
        from v3.latency import command_alignment
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'policy_logs').mkdir()
            action=np.zeros((3,34));action[:,0]=np.arange(3);action[1,5:]=.2
            np.savetxt(path/'policy_logs/action.csv',action,delimiter=',',header=','.join(['x']*34),comments='')
            h=dict(applied_index=0,consumed_monotonic=10.,requested=9.99,anchor=1,chunk_index=0,
                   execution_index=2,rgb_sim_timestamp=1.,proprio_sim_timestamp=1.,hand_applied_monotonic=10.004)
            events=[(10.001,np.zeros(29)),(10.003,np.full(29,.2)),(10.025,np.full(29,.2))]
            parameters=dict(isaaclab_to_mujoco=np.arange(29),g1_action_scale=np.ones(29),default_angles=np.zeros(29))
            with patch('v3.latency.cpp_parameters',return_value=parameters):
                rows,report=command_alignment(path,[h],events,{'sonic':{'repository':'unused'}})
            self.assertEqual(report['matched'],1)
            self.assertEqual(rows[0]['body_command_received_monotonic'],10.003)
            self.assertAlmostEqual(rows[0]['observation_to_command_ms'],14.)

if __name__=='__main__':unittest.main()
