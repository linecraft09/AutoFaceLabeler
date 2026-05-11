#!/usr/bin/env python3
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from validators.validator import V2ContentFilter


class FakeFaceEmbedder:
    def __init__(self):
        self.added_embeddings = []

    def is_duplicate(self, embedding, threshold):
        return False

    def add_embedding(self, embedding):
        self.added_embeddings.append(embedding)


class TestValidatorFineFilter(unittest.TestCase):
    def _make_filter(self, embedder, fine_overrides=None):
        fine_config = {
            'dedup_threshold': 0.5,
            'speech_required': False,
            'min_duration': 0,
        }
        if fine_overrides:
            fine_config.update(fine_overrides)
        with patch('validators.validator.YOLODetector'), \
             patch('validators.validator.ArcFaceEmbedder', return_value=embedder):
            return V2ContentFilter(
                {
                    'device': 'cpu',
                    'fine': fine_config,
                },
                video_store=object(),
            )

    def test_keeps_temporal_order_and_persists_all_selected_embeddings(self):
        embedder = FakeFaceEmbedder()
        filt = self._make_filter(embedder)
        clips = ['clip_1.mp4', 'clip_2.mp4', 'clip_3.mp4']
        embeddings = {
            'clip_1.mp4': np.ones(512, dtype=np.float32),
            'clip_2.mp4': np.full(512, 2.0, dtype=np.float32),
            'clip_3.mp4': np.full(512, 3.0, dtype=np.float32),
        }
        durations = {'clip_1.mp4': 1.0, 'clip_2.mp4': 10.0, 'clip_3.mp4': 2.0}

        def process_single(path):
            # clip_2 has the highest quality/duration. The new logic must still
            # keep the original temporal order instead of ranking it first.
            clarity = 0.99 if path == 'clip_2.mp4' else 0.5
            return path, True, (0.5, clarity, embeddings[path]), None

        with patch.object(filt, '_process_single_video', side_effect=process_single), \
             patch(
                 'validators.validator.VU.get_video_secs',
                 side_effect=lambda path: durations[path],
             ), \
             patch('validators.validator.VU.concat_videos', return_value='merged.mp4') as concat:
            result, fail_reason = filt._fine_filter(clips)

        self.assertEqual(result, 'merged.mp4')
        self.assertIsNone(fail_reason)
        concat.assert_called_once_with(clips)
        self.assertEqual(len(embedder.added_embeddings), 3)
        for added, expected_path in zip(embedder.added_embeddings, clips):
            np.testing.assert_array_equal(added, embeddings[expected_path])

    def test_local_dedup_keeps_first_temporal_occurrence(self):
        embedder = FakeFaceEmbedder()
        filt = self._make_filter(embedder)
        clips = ['clip_1.mp4', 'clip_2.mp4', 'clip_3.mp4']
        first_embedding = np.ones(512, dtype=np.float32)
        embeddings = {
            'clip_1.mp4': first_embedding,
            'clip_2.mp4': first_embedding.copy(),
            'clip_3.mp4': np.full(512, 3.0, dtype=np.float32),
        }

        def process_single(path):
            return path, True, (0.8, 0.8, embeddings[path]), None

        with patch.object(filt, '_process_single_video', side_effect=process_single), \
             patch('validators.validator.VU.get_video_secs', return_value=1.0), \
             patch('validators.validator.VU.concat_videos', return_value='merged.mp4') as concat:
            result, fail_reason = filt._fine_filter(clips)

        self.assertEqual(result, 'merged.mp4')
        self.assertIsNone(fail_reason)
        concat.assert_called_once_with(['clip_1.mp4', 'clip_3.mp4'])
        self.assertEqual(len(embedder.added_embeddings), 2)
        np.testing.assert_array_equal(
            embedder.added_embeddings[0],
            embeddings['clip_1.mp4'],
        )
        np.testing.assert_array_equal(
            embedder.added_embeddings[1],
            embeddings['clip_3.mp4'],
        )

    def test_rejects_when_selected_total_duration_is_shorter_than_minimum(self):
        embedder = FakeFaceEmbedder()
        filt = self._make_filter(embedder, {'min_duration': 30})
        clips = ['clip_1.mp4', 'clip_2.mp4']
        embeddings = {
            'clip_1.mp4': np.ones(512, dtype=np.float32),
            'clip_2.mp4': np.full(512, 2.0, dtype=np.float32),
        }

        def process_single(path):
            return path, True, (0.8, 0.8, embeddings[path]), None

        with patch.object(filt, '_process_single_video', side_effect=process_single), \
             patch('validators.validator.VU.get_video_secs', return_value=14.0), \
             patch('validators.validator.VU.concat_videos') as concat:
            result, fail_reason = filt._fine_filter(clips)

        self.assertIsNone(result)
        self.assertEqual(fail_reason, 'merged_duration_too_short')
        concat.assert_not_called()
        self.assertEqual(embedder.added_embeddings, [])

    def test_rejects_when_merged_video_duration_is_shorter_than_minimum(self):
        embedder = FakeFaceEmbedder()
        filt = self._make_filter(embedder, {'min_duration': 30})
        clips = ['clip_1.mp4', 'clip_2.mp4']
        embeddings = {
            'clip_1.mp4': np.ones(512, dtype=np.float32),
            'clip_2.mp4': np.full(512, 2.0, dtype=np.float32),
        }

        def process_single(path):
            return path, True, (0.8, 0.8, embeddings[path]), None

        durations = {
            'clip_1.mp4': 20.0,
            'clip_2.mp4': 20.0,
            'merged.mp4': 29.0,
        }

        with patch.object(filt, '_process_single_video', side_effect=process_single), \
             patch(
                 'validators.validator.VU.get_video_secs',
                 side_effect=lambda path: durations[path],
             ), \
             patch('validators.validator.VU.concat_videos', return_value='merged.mp4'), \
             patch('validators.validator.os.unlink') as unlink:
            result, fail_reason = filt._fine_filter(clips)

        self.assertIsNone(result)
        self.assertEqual(fail_reason, 'merged_duration_too_short')
        unlink.assert_called_once_with('merged.mp4')
        self.assertEqual(embedder.added_embeddings, [])


if __name__ == '__main__':
    unittest.main()
