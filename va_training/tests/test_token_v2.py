import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from datasets.tokens import TokenEpisode, TokenDataset, STATE_WIDTHS, read_log, projected_gravity, METADATA_COLUMNS
from models import build_policy
from models.token_inference import TokenModelPolicy
from losses.token import TokenMetrics, token_loss
from sonic.release import snap_token, ExternalTokenAdapter, require_pass


class TokenV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def episode(self, ident='a', bias=0):
        times = np.arange(6)*.02; anchors = np.array([0, 2, 4, 5])
        image = self.path/(ident+'.npz')
        np.savez(image, head_rgb=np.zeros((4, 8, 8, 3), dtype=np.uint8))
        state = {k: np.zeros((6, v), dtype=np.float32) for k, v in STATE_WIDTHS.items()}
        state['base_quat'][:, 0] = 1
        state['projected_gravity'][:, 2] = -1
        state['body_q'][:, 0] = np.arange(6)+bias
        token = np.repeat((np.arange(6)/16+bias)[:, None], 64, axis=1).astype(np.float32)
        hand = np.repeat((10+np.arange(6)+bias)[:, None], 14, axis=1).astype(np.float32)
        return TokenEpisode(ident, ident, None, self.path, image, times, times[anchors], anchors,
            state, token, hand, np.zeros((6, 5)), [], [], {})

    def test_post_step_targets_next_tick_and_preserves_shared_hand_timestamp(self):
        data = TokenDataset([self.episode()], image_size=16, history_length=3, target_offset=1)
        first, middle, last = data[0], data[1], data[2]
        self.assertEqual(len(data), 3)  # final RGB at index 5 has no next tick
        self.assertEqual(first['state_index'], 0)
        self.assertEqual(first['execution_index'], 1)
        self.assertEqual(first['execution_timestamp'], .02)
        torch.testing.assert_close(first['target_sonic_token'], torch.full((64,), 1/16))
        torch.testing.assert_close(first['target_hand_ref'], torch.full((14,), 11.))
        torch.testing.assert_close(first['history_mask'], torch.tensor([False, False, True]))
        self.assertEqual(middle['execution_index'], 3)
        self.assertEqual(last['execution_index'], 5)
        torch.testing.assert_close(middle['state']['body_q'][:, 0], torch.tensor([0., 1., 2.]))

    def test_history_and_chunk_cannot_cross_episode_boundary(self):
        data = TokenDataset([self.episode('a'), self.episode('b', 10)], 16, 3, 2, 1)
        self.assertEqual(len(data), 4)
        first_b = data[2]
        self.assertEqual(first_b['episode_id'], 'b')
        torch.testing.assert_close(first_b['state']['body_q'][:, 0], torch.full((3,), 10.))
        self.assertEqual(first_b['target_sonic_token'].shape, (2, 64))
        torch.testing.assert_close(first_b['target_sonic_token'][:, 0], torch.tensor([10+1/16, 10+2/16]))
        with self.assertRaises(ValueError):
            TokenDataset([self.episode()], target_offset=0)

    def test_csv_metadata_is_never_a_token_coordinate(self):
        path = self.path/'token.csv'
        with path.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(METADATA_COLUMNS+[f'token_{i}' for i in range(64)])
            writer.writerow([0, 0, 1e12, 1e8, 0]+[i/16 for i in range(64)])
            writer.writerow([1, 20, 1e12+20, 1e8+20, 0]+[0]*64)
        meta, token = read_log(path, [f'token_{i}' for i in range(64)])
        self.assertEqual(meta.shape, (2, 5)); self.assertEqual(token.shape, (2, 64))
        np.testing.assert_array_equal(token[0], np.arange(64)/16)
        path.write_text(path.read_text().replace('token_63', 'wrong_name'))
        with self.assertRaises(ValueError):
            read_log(path, [f'token_{i}' for i in range(64)])

    def test_gravity_and_fsq_projection(self):
        np.testing.assert_allclose(projected_gravity(np.array([[1., 0, 0, 0], [0., 1, 0, 0]])), [[0, 0, -1], [0, 0, 1]])
        np.testing.assert_array_equal(snap_token(np.array([-2., .02, .07, 2.])), [-1, 0, 1/16, 15/16])
        metrics = TokenMetrics()
        target = torch.zeros(1, 64); predicted = target+.02
        metrics.add({'sonic_token': predicted, 'hand_ref': torch.zeros(1, 14)}, target, torch.zeros(1, 14))
        report = metrics.report()
        self.assertEqual(report['fsq_level_accuracy'], 1)
        self.assertAlmostEqual(report['token_mae'], .02, places=6)
        self.assertEqual(len(report['token_mae_per_dimension']), 64)

    def test_v2_roundtrip_and_invalid_history_does_not_change_prediction(self):
        config = {'model_type': 'token_v2', 'model': {'image_size': 16, 'vision_channels': [4, 8],
                  'd_model': 16, 'fusion_hidden': 32, 'field_hidden': 4, 'history_length': 3, 'token_chunk_size': 1}}
        stats = {'image': {'mean': [0]*3, 'std': [1]*3}, 'hand': {'mean': [0]*14, 'std': [1]*14},
                 'state': {k: {'mean': [0]*v, 'std': [1]*v} for k, v in STATE_WIDTHS.items()}}
        policy = build_policy(config, stats)
        sample = TokenDataset([self.episode()], 16, 3)[0]
        state = {k: v[None] for k, v in sample['state'].items()}; mask = sample['history_mask'][None]
        image = sample['image'][None]
        output = policy(image, state, mask)
        self.assertEqual(output['sonic_token'].shape, (1, 64)); self.assertEqual(output['hand_ref'].shape, (1, 14))
        changed = {k: v.clone() for k, v in state.items()}
        for value in changed.values():
            value[:, :2] += 100
        other = policy(image, changed, mask)
        torch.testing.assert_close(other['sonic_token'], output['sonic_token'])
        loss, _ = token_loss(output, torch.zeros(1, 64), torch.zeros(1, 14),
                            {'token_loss': 'mse', 'body_loss_weight': 1., 'hand_loss_weight': 2.})
        loss.backward()
        self.assertGreater(float(policy.body_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(policy.hand_head.weight.grad.abs().sum()), 0)
        path = self.path/'v2.pt'
        torch.save({'config': config, 'normalization_stats': stats, 'model': policy.state_dict()}, path)
        interface, _ = TokenModelPolicy.from_checkpoint(path, 'cpu')
        result = interface.predict(sample['image'], sample['state'], sample['history_mask'])
        torch.testing.assert_close(result['sonic_token'], output['sonic_token'][0])

    def test_oracle_gate_rejects_failure_and_different_bundle(self):
        path = self.path/'gate.json'
        path.write_text(json.dumps({'oracle_pass': False}))
        with self.assertRaises(ValueError):
            require_pass(path, 'oracle_pass')
        path.write_text(json.dumps({'oracle_pass': True, 'provenance': {'sonic': {'x': 1}}}))
        with self.assertRaises(ValueError):
            require_pass(path, 'oracle_pass', {'x': 2})

    def test_live_history_and_acknowledged_hand_pair(self):
        from models.observation_history import ObservationHistory
        from sonic.token_handoff import TokenHandoff
        history = ObservationHistory(3)
        state = {k: np.zeros(v) for k, v in STATE_WIDTHS.items()}
        state['base_quat'][0] = 1
        history.append(0, state)
        rows, mask = history.snapshot()
        np.testing.assert_array_equal(mask, [False, False, True])
        np.testing.assert_allclose(rows['projected_gravity'][-1], [0, 0, -1])
        with self.assertRaises(ValueError):
            history.append(2, state)
        handoff = TokenHandoff()
        first = dict(token=np.zeros(64), hand=np.zeros(14), execution_index=0, anchor=-1)
        handoff.propose(first)
        np.testing.assert_array_equal(handoff.consume(0, np.zeros(64)), first['hand'])
        next_pair = dict(token=np.ones(64), hand=np.ones(14), execution_index=1, anchor=0)
        handoff.propose(next_pair)
        np.testing.assert_array_equal(handoff.consume(1, np.zeros(64)), first['hand'])
        np.testing.assert_array_equal(handoff.consume(2, np.ones(64)), next_pair['hand'])
        self.assertEqual(handoff.applied[-1]['applied_index'], 2)
        with self.assertRaises(RuntimeError):
            handoff.consume(3, np.full(64, .5))

    @unittest.skipUnless(os.environ.get('VA_V2_REAL_CONFIG'), 'Set VA_V2_REAL_CONFIG for actual-episode alignment checks')
    def test_actual_native_labels_and_protocol_v4(self):
        from utils.v2_config import load_v2_config
        from datasets.tokens import load_token_episodes
        config = load_v2_config(os.environ['VA_V2_REAL_CONFIG'])
        episodes = load_token_episodes(config['data'])
        data = TokenDataset(episodes, config['model']['image_size'], config['model']['history_length'], 1, 1)
        # Inspect the first and last eligible RGB anchors of EVERY real episode.
        for i, episode in enumerate(episodes):
            indexes = [j for j, v in enumerate(data.index) if v[0] == i]
            for index in (indexes[0], indexes[-1]):
                sample = data[index]; t = sample['state_index']; target = t+1
                torch.testing.assert_close(sample['target_sonic_token'], torch.from_numpy(episode.tokens[target]))
                torch.testing.assert_close(sample['target_hand_ref'], torch.tensor(episode.hand_ref[target], dtype=torch.float32))
                self.assertLess(target, len(episode.times))
                self.assertEqual(sample['execution_timestamp'], float(episode.times[target]))
        packer = ExternalTokenAdapter(config['sonic']['repository'])
        message = packer.message(episodes[0].tokens[0], 0)
        header = json.loads(message[4:1284].rstrip(b'\0'))
        self.assertEqual(header['v'], 4)
        self.assertEqual(header['fields'][0]['shape'], [1, 64])
        np.testing.assert_array_equal(np.frombuffer(message[1284:1284+256], dtype='<f4'), episodes[0].tokens[0])


if __name__ == '__main__':
    unittest.main()
