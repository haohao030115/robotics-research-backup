import json
from pathlib import Path
import tempfile
import shutil
import unittest
import numpy as np
import torch
from datasets import load_episodes, split_episodes, EpisodeDataset, compute_statistics
from scripts.make_synthetic import create_fixture
from tests.common import config


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cfg = config()
        self.cfg['data']['manifest'] = str(create_fixture(self.root))

    def tearDown(self):
        self.directory.cleanup()

    def rewrite_images(self, mutate):
        path = self.root/'images_0.npz'
        with np.load(path) as data:
            values = {k: data[k].copy() for k in data.files}
        mutate(values)
        np.savez(path, **values)

    def test_actual_state_and_same_row_reference_chunk(self):
        episodes = load_episodes(self.cfg['data'])
        # Use nonconstant reference to catch wrong horizon start/end indices.
        episodes[0].ref = np.arange(80*43, dtype=np.float32).reshape(80, 43)
        data = EpisodeDataset([episodes[0]], 20, 16)
        sample = data[1]
        self.assertEqual(sample['image'].shape, (3, 16, 16))
        self.assertEqual(sample['q'].shape, (43,))
        self.assertEqual(sample['target'].shape, (20, 43))
        np.testing.assert_array_equal(sample['q'].numpy(), episodes[0].q[5])
        np.testing.assert_array_equal(sample['target'].numpy(), episodes[0].ref[5:25])
        self.assertEqual(len(data), 13)

    def test_first_target_values_and_last_valid_anchor(self):
        episode = load_episodes(self.cfg['data'])[0]
        episode.ref = np.arange(80*43, dtype=np.float32).reshape(80, 43)
        dataset = EpisodeDataset([episode], 20, 16)
        for index in [0, 1, len(dataset)-1]:
            sample = dataset[index]
            anchor = sample['state_index']
            torch.testing.assert_close(sample['target'][0, :29], torch.from_numpy(episode.ref[anchor, :29]))
            torch.testing.assert_close(sample['target'][0, 29:], torch.from_numpy(episode.ref[anchor, 29:]))
            self.assertFalse(torch.equal(sample['target'][0], torch.from_numpy(episode.ref[anchor+1])))
            np.testing.assert_array_equal(sample['target'].numpy(), episode.ref[anchor:anchor+20])
        self.assertEqual(dataset[-1]['state_index'], 80-20)
        np.testing.assert_array_equal(dataset[-1]['target'][-1].numpy(), episode.ref[-1])
        self.assertFalse(any(anchor > 80-20 for _, _, anchor in dataset.index))

    def test_split_and_train_only_statistics(self):
        episodes = load_episodes(self.cfg['data'])
        train, val = split_episodes(episodes, self.cfg['data'])
        self.assertFalse({e.id for e in train} & {e.id for e in val})
        val[0].q[:] = 1e6
        data = EpisodeDataset(train, 20, 16)
        statistics = compute_statistics(data)
        expected = torch.stack([data[i]['q'] for i in range(len(data))]).mean(0)
        np.testing.assert_allclose(statistics['state']['mean'], expected.numpy(), atol=1e-6)
        self.assertEqual(statistics['train_episode_ids'], ['SYNTHETIC_0', 'SYNTHETIC_1'])

    def test_nearest_timestamp_alignment(self):
        self.cfg['data']['alignment'] = 'nearest_timestamp'
        episodes = load_episodes(self.cfg['data'])
        np.testing.assert_array_equal(episodes[0].anchors, np.arange(0, 80, 5))

    def test_bad_mapping_rejected(self):
        self.rewrite_images(lambda d: d['rgb_reference_frame_indices'].__setitem__(0, 999))
        with self.assertRaisesRegex(ValueError, 'out of range'):
            load_episodes(self.cfg['data'])

    def test_timestamp_mismatch_rejected(self):
        self.rewrite_images(lambda d: d.__setitem__('timestamps', d['timestamps']+0.05))
        with self.assertRaisesRegex(ValueError, 'timestamp mismatch'):
            load_episodes(self.cfg['data'])

    def test_reused_episode_files_rejected(self):
        path = self.root/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['episodes'][1]['states'] = manifest['episodes'][0]['states']
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'reused'):
            load_episodes(self.cfg['data'])

    def test_ambiguous_mapping_rejected(self):
        self.cfg['data']['mapping_semantics'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'semantics'):
            load_episodes(self.cfg['data'])

    def test_training_manifest_canonical_paths_and_inline_mapping(self):
        manifest = json.loads((self.root/'manifest.json').read_text())
        for entry in manifest['episodes']:
            entry['episode_path'] = entry.pop('states')
            entry['rgb_path'] = entry.pop('images')
            entry['rgb_reference_frame_indices'] = list(range(0, 80, 5))
        path = self.root/'training_manifest.json'
        path.write_text(json.dumps(manifest))
        self.cfg['data']['manifest'] = str(path)
        episodes = load_episodes(self.cfg['data'])
        self.assertEqual(episodes[0].anchors.tolist(), list(range(0, 80, 5)))
        episode = episodes[0]
        dataset = EpisodeDataset([episode], 20, 16)
        before = dataset[0]
        episode.ref += 100
        after = dataset[0]
        torch.testing.assert_close(before['q'], after['q'])
        torch.testing.assert_close(before['target']+100, after['target'])

    def test_directory_episode_path_requires_explicit_filename(self):
        manifest = json.loads((self.root/'manifest.json').read_text())
        for i, entry in enumerate(manifest['episodes']):
            entry['episode_path'] = '.'
            entry.pop('states')
        path = self.root/'training_manifest.json'
        path.write_text(json.dumps(manifest))
        self.cfg['data']['manifest'] = str(path)
        with self.assertRaisesRegex(ValueError, 'episode_state_filename'):
            load_episodes(self.cfg['data'])

    def test_conflicting_manifest_and_npz_mapping_rejected(self):
        manifest = json.loads((self.root/'manifest.json').read_text())
        manifest['episodes'][0]['rgb_reference_frame_indices'] = list(range(1, 81, 5))
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'mappings disagree'):
            load_episodes(self.cfg['data'])

    def test_directory_episode_path_with_configured_state_filename(self):
        manifest = json.loads((self.root/'manifest.json').read_text())
        for i, entry in enumerate(manifest['episodes']):
            directory = self.root/f'episode_{i}'
            directory.mkdir()
            shutil.copyfile(self.root/entry.pop('states'), directory/'actual_state.npz')
            entry['episode_path'] = directory.name
            entry['rgb_path'] = entry.pop('images')
        path = self.root/'training_manifest.json'
        path.write_text(json.dumps(manifest))
        self.cfg['data'].update(manifest=str(path), episode_state_filename='actual_state.npz')
        episodes = load_episodes(self.cfg['data'])
        self.assertEqual(len(episodes), 3)
        sample = EpisodeDataset(episodes, 20, 16)[0]
        self.assertEqual(sample['q'].shape, (43,))
        self.assertEqual(sample['target'].shape, (20, 43))

    def test_source_episode_group_prevents_chunk_leakage(self):
        manifest = json.loads((self.root/'manifest.json').read_text())
        manifest['episodes'][0]['source_episode_id'] = 'same_robot_episode'
        manifest['episodes'][2]['source_episode_id'] = 'same_robot_episode'
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        episodes = load_episodes(self.cfg['data'])
        with self.assertRaisesRegex(ValueError, 'frame leakage'):
            split_episodes(episodes, self.cfg['data'])
