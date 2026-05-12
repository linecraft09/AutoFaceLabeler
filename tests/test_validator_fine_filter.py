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
from validators.v2_models.scrfd_batch_detector import Detection


class FakeFaceEmbedder:
    def __init__(self):
        self.added_embeddings = []

    def is_duplicate(self, embedding, threshold):
        return False

    def add_embedding(self, embedding):
        self.added_embeddings.append(embedding)


class FakeEmbeddingStore:
    def __init__(self, embeddings_by_source=None):
        self.embeddings_by_source = embeddings_by_source or {}
        self.saved_embeddings = []

    def load_embeddings_excluding(self, video_id, platform):
        labels = []
        embeddings = []
        for (source_video_id, source_platform), source_embeddings in self.embeddings_by_source.items():
            if source_video_id == video_id and source_platform == platform:
                continue
            for index, embedding in enumerate(source_embeddings):
                labels.append(f"{source_platform}:{source_video_id}:face:{index}")
                embeddings.append(embedding)
        return labels, embeddings

    def save_embeddings(self, video_id, platform, embeddings):
        self.saved_embeddings.append((video_id, platform, list(embeddings)))
        return len(embeddings)


class TestValidatorFineFilter(unittest.TestCase):
    def _make_filter(self, embedder, fine_overrides=None, video_store=None):
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
                video_store=video_store or object(),
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

    def test_same_source_duplicate_embeddings_are_kept(self):
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
        concat.assert_called_once_with(clips)
        self.assertEqual(len(embedder.added_embeddings), 3)
        for added, expected_path in zip(embedder.added_embeddings, clips):
            np.testing.assert_array_equal(added, embeddings[expected_path])

    def test_rejects_duplicate_embedding_from_other_source_video(self):
        embedder = FakeFaceEmbedder()
        duplicate_embedding = np.ones(512, dtype=np.float32)
        store = FakeEmbeddingStore({
            ('other_video', 'youtube'): [duplicate_embedding.copy()],
        })
        filt = self._make_filter(embedder, video_store=store)

        def process_single(path):
            return path, True, (0.8, 0.8, duplicate_embedding), None

        with patch.object(filt, '_process_single_video', side_effect=process_single), \
             patch('validators.validator.VU.get_video_secs', return_value=1.0):
            result, fail_reason = filt._fine_filter(
                ['clip_1.mp4'],
                video_id='current_video',
                platform='youtube',
            )

        self.assertIsNone(result)
        self.assertEqual(fail_reason, 'duplicate')
        self.assertEqual(embedder.added_embeddings, [])

    def test_ignores_duplicate_embedding_from_current_source_video(self):
        embedder = FakeFaceEmbedder()
        duplicate_embedding = np.ones(512, dtype=np.float32)
        store = FakeEmbeddingStore({
            ('current_video', 'youtube'): [duplicate_embedding.copy()],
        })
        filt = self._make_filter(embedder, video_store=store)

        def process_single(path):
            return path, True, (0.8, 0.8, duplicate_embedding), None

        with patch.object(filt, '_process_single_video', side_effect=process_single), \
             patch('validators.validator.VU.get_video_secs', return_value=1.0):
            result, fail_reason = filt._fine_filter(
                ['clip_1.mp4'],
                video_id='current_video',
                platform='youtube',
            )

        self.assertEqual(result, 'clip_1.mp4')
        self.assertIsNone(fail_reason)
        self.assertEqual(len(embedder.added_embeddings), 1)

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

    def test_primary_only_selects_largest_face_per_frame(self):
        filt = V2ContentFilter.__new__(V2ContentFilter)
        filt.face_primary_only = True
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        small = Detection(
            bbox=np.array([10, 10, 30, 30], dtype=np.float32),
            score=0.99,
        )
        large = Detection(
            bbox=np.array([20, 20, 80, 70], dtype=np.float32),
            score=0.75,
        )

        selected = filt._select_frame_detections(frame, [small, large])

        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0][0], large)

    def test_primary_only_disabled_keeps_all_valid_faces(self):
        filt = V2ContentFilter.__new__(V2ContentFilter)
        filt.face_primary_only = False
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            Detection(
                bbox=np.array([10, 10, 30, 30], dtype=np.float32),
                score=0.99,
            ),
            Detection(
                bbox=np.array([20, 20, 80, 70], dtype=np.float32),
                score=0.75,
            ),
        ]

        selected = filt._select_frame_detections(frame, detections)

        self.assertEqual([item[0] for item in selected], detections)

    def test_format_pose_stats_reports_axis_and_max_abs_distribution(self):
        poses = [
            np.array([1.0, -2.0, 3.0], dtype=np.float32),
            np.array([-4.0, 5.0, -6.0], dtype=np.float32),
        ]

        stats = V2ContentFilter._format_pose_stats(poses)

        self.assertIn("n=2", stats)
        self.assertIn("pitch_abs=", stats)
        self.assertIn("yaw_abs=", stats)
        self.assertIn("roll_abs=", stats)
        self.assertIn("max_abs=", stats)


if __name__ == '__main__':
    unittest.main()
