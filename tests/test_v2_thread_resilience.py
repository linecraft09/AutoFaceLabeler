"""Test that V2 filter thread survives per-video exceptions and continues processing."""
import os
import sys
import sqlite3
import tempfile
import time
from pathlib import Path

# Ensure src is on path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pytest


class TestV2FilterThreadResilience:
    """Verify V2Filter daemon thread doesn't die on per-video exceptions."""

    def test_thread_survives_exception_in_process_video(self, monkeypatch: "pytest.MonkeyPatch"):
        """After one video raises in _process_video, thread must keep processing subsequent videos."""
        from validators.validator import V2ContentFilter
        from core.storage.video_store import VideoStore

        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        store = VideoStore(db_path=db_path)

        # Insert two downloaded videos
        from core.models.video_meta import VideoMeta
        store.insert_or_update(
            VideoMeta(video_id="v1", platform="youtube", title="test1",
             url="http://x", duration_seconds=60, resolution="720p",
             channel="ch", publish_date="2024-01-01", view_count=1000,
             tags=[], search_term="test"),
            file_path="/nonexistent/v1.mp4")
        store.insert_or_update(
            VideoMeta(video_id="v2", platform="youtube", title="test2",
             url="http://x", duration_seconds=60, resolution="720p",
             channel="ch", publish_date="2024-01-01", view_count=1000,
             tags=[], search_term="test"),
            file_path="/nonexistent/v2.mp4")

        # Verify both are in 'downloaded' status
        pending = store.get_pending_videos(limit=10)
        assert len(pending) == 2

        # Create filter and monkey-patch _process_video to:
        # - Raise on first call, succeed on second
        call_count = [0]
        original_process = V2ContentFilter._process_video

        def _process_video_patch(self, video_row):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated processing crash")
            # Mark as v2_coarse_failed so video leaves 'downloaded' state
            self.store.update_status(video_row['video_id'], video_row['platform'],
                                     'v2_coarse_failed', 'test_reason')

        monkeypatch.setattr(V2ContentFilter, '_process_video', _process_video_patch)

        config = {
            'coarse_filter': {'model_path': 'yolo11n.pt', 'device': 'cpu',
                              'single_person_threshold': 0.8, 'audio_required': False},
            'fine_filter': {'device': 'cpu', 'face_db_path': '/tmp/test_faiss.faiss'},
            'qualified_dir': '/tmp/qual_test',
        }

        v2 = V2ContentFilter(config, store)
        v2.start()
        time.sleep(3)  # Wait for thread to process both videos
        v2.stop()

        # Both videos should have been processed (first raised, second succeeded)
        assert call_count[0] >= 2, (
            f"Expected >=2 calls, got {call_count[0]}. "
            f"Thread likely died after first exception."
        )

    def test_thread_continues_after_file_missing(self):
        """Thread should mark status and continue when file is missing."""
        from validators.validator import V2ContentFilter
        from core.storage.video_store import VideoStore

        db_path = os.path.join(tempfile.mkdtemp(), "test.db")
        store = VideoStore(db_path=db_path)

        from core.models.video_meta import VideoMeta
        for i in range(3):
            store.insert_or_update(
                VideoMeta(video_id=f"v{i}", platform="youtube", title=f"t{i}",
                 url="http://x", duration_seconds=60, resolution="720p",
                 channel="ch", publish_date="2024-01-01", view_count=1000,
                 tags=[], search_term="t"),
                file_path=f"/nonexistent/v{i}.mp4")

        config = {
            'coarse_filter': {'model_path': 'yolo11n.pt', 'device': 'cpu',
                              'single_person_threshold': 0.8, 'audio_required': False},
            'fine_filter': {'device': 'cpu', 'face_db_path': '/tmp/test_faiss2.faiss'},
            'qualified_dir': '/tmp/qual_test2',
        }

        v2 = V2ContentFilter(config, store)
        v2.start()
        time.sleep(4)
        v2.stop()

        # All 3 should be marked as 'v2_failed' with reason 'file_missing'
        with store._connect() as conn:
            cursor = conn.execute("SELECT status, v2_fail_reason FROM videos")
            rows = cursor.fetchall()
            assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
            for row in rows:
                assert row[0] == 'v2_failed', f"Expected v2_failed, got {row}"
                assert 'file_missing' in (row[1] or ''), f"Unexpected reason: {row[1]}"
