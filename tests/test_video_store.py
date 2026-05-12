#!/usr/bin/env python3
"""
测试 VideoStore 模块：
1. 初始化数据库
2. insert_or_update 插入/更新视频
3. get_pending_videos
4. update_status
5. get_statistics
6. get_existing_video_ids
7. get_file_paths_by_excluded_statuses
"""

import os
import shutil
import sqlite3
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

# 让 Python 能直接导入 src 下的模块（core/aflutils）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from core.models.video_meta import VideoMeta
from core.storage.video_store import VideoStore


OUTPUT_DIR = Path(PROJECT_ROOT) / "tests" / "test_output"


def make_video(video_id: str, platform: str = "youtube", title: str = "title") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://example.com/watch/{video_id}",
        platform=platform,
        title=title,
        duration_seconds=120,
        resolution="1080p",
        channel="channel_a",
        publish_date="2026-01-01",
        view_count=1000,
        tags=["tag1", "tag2"],
        search_term="keyword",
        extra={},
    )


class TestVideoStore(unittest.TestCase):
    def setUp(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = OUTPUT_DIR / f"video_store_{uuid.uuid4().hex}.db"
        self.store = VideoStore(str(self.db_path))
        self.temp_dirs = []

    def tearDown(self):
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        for suffix in ("", "-wal", "-shm"):
            p = Path(f"{self.db_path}{suffix}")
            if p.exists():
                p.unlink()

    def test_init_db(self):
        self.assertTrue(self.db_path.exists(), "数据库文件应在 tests/test_output 下创建")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='videos'"
            )
            self.assertIsNotNone(cursor.fetchone(), "videos 表应被初始化")

    def test_insert_or_update(self):
        v1 = make_video("vid_1", title="first")
        ok = self.store.insert_or_update(v1, file_path="/tmp/a.mp4")
        self.assertTrue(ok)

        inserted = self.store.get_video_by_id("vid_1", "youtube")
        self.assertIsNotNone(inserted)
        self.assertEqual(inserted["title"], "first")
        self.assertEqual(inserted["file_path"], "/tmp/a.mp4")
        self.assertEqual(inserted["status"], "downloaded")

        # 再次插入同一主键，验证更新逻辑（file_path=None 时保留原路径）
        v1_updated = make_video("vid_1", title="updated")
        v1_updated.duration_seconds = 360
        v1_updated.tags = ["new_tag"]
        ok = self.store.insert_or_update(v1_updated, file_path=None)
        self.assertTrue(ok)

        updated = self.store.get_video_by_id("vid_1", "youtube")
        self.assertEqual(updated["title"], "updated")
        self.assertEqual(updated["duration_seconds"], 360)
        self.assertEqual(updated["file_path"], "/tmp/a.mp4")

    def test_get_pending_videos(self):
        self.store.insert_or_update(make_video("p1"), file_path="/tmp/p1.mp4")
        self.store.insert_or_update(make_video("p2"), file_path="/tmp/p2.mp4")
        self.store.insert_or_update(make_video("done"), file_path="/tmp/done.mp4")
        self.store.update_status("done", "youtube", "v2_passed")

        pending = self.store.get_pending_videos(limit=10)
        pending_ids = {row["video_id"] for row in pending}
        self.assertEqual(pending_ids, {"p1", "p2"})

    def test_download_queue_lifecycle(self):
        queued = make_video("q1")
        self.assertTrue(self.store.enqueue_download_candidate(queued))
        self.assertEqual(self.store.get_download_queue_count(), 1)

        candidates = self.store.list_download_candidates(limit=10)
        self.assertEqual([candidate.video_id for candidate in candidates], ["q1"])

        self.store.insert_or_update(queued, file_path="/tmp/q1.mp4")
        self.assertTrue(self.store.delete_download_candidate("q1", "youtube"))
        self.assertEqual(self.store.get_download_queue_count(), 0)

        self.assertFalse(self.store.enqueue_download_candidate(queued))

    def test_update_status(self):
        self.store.insert_or_update(make_video("u1"), file_path="/tmp/u1.mp4")
        self.store.update_status("u1", "youtube", "v2_failed", "face_not_found")

        row = self.store.get_video_by_id("u1", "youtube")
        self.assertEqual(row["status"], "v2_failed")
        self.assertEqual(row["v2_fail_reason"], "face_not_found")

    def test_get_statistics(self):
        self.store.insert_or_update(make_video("s1"), file_path="/tmp/s1.mp4")
        self.store.insert_or_update(make_video("s2"), file_path="/tmp/s2.mp4")
        self.store.insert_or_update(make_video("s3"), file_path="/tmp/s3.mp4")

        self.store.update_status("s2", "youtube", "v2_passed")
        self.store.update_status("s3", "youtube", "v2_failed", "bad_quality")

        stats = self.store.get_statistics()
        self.assertEqual(stats.get("downloaded", 0), 1)
        self.assertEqual(stats.get("v2_passed", 0), 1)
        self.assertEqual(stats.get("v2_failed", 0), 1)

    def test_get_existing_video_ids(self):
        self.store.insert_or_update(make_video("e1", platform="youtube"), file_path="/tmp/e1.mp4")
        self.store.insert_or_update(make_video("e2", platform="youtube"), file_path="/tmp/e2.mp4")
        self.store.insert_or_update(make_video("e1", platform="bilibili"), file_path="/tmp/e1b.mp4")

        existing = self.store.get_existing_video_ids(
            "youtube", ["e1", "e2", "e3", "", None]  # type: ignore[list-item]
        )
        self.assertEqual(existing, {"e1", "e2"})

    def test_get_file_paths_by_excluded_statuses(self):
        self.store.insert_or_update(make_video("f1"), file_path="/tmp/a.mp4")
        self.store.insert_or_update(make_video("f2"), file_path="/tmp/b.mp4")
        self.store.insert_or_update(make_video("f3"), file_path="/tmp/c.mp4")
        self.store.insert_or_update(make_video("f4"), file_path="/tmp/c.mp4")
        self.store.insert_or_update(make_video("f5"), file_path="")
        self.store.insert_or_update(make_video("f6"), file_path=None)

        self.store.update_status("f1", "youtube", "v2_passed")
        self.store.update_status("f2", "youtube", "v2_failed", "reason_b")
        self.store.update_status("f3", "youtube", "v2_coarse_failed", "reason_c")
        self.store.update_status("f4", "youtube", "v2_fine_failed", "reason_d")

        paths = self.store.get_file_paths_by_excluded_statuses()
        self.assertEqual(set(paths), {"/tmp/b.mp4", "/tmp/c.mp4"})

        self.assertEqual(self.store.get_file_paths_by_excluded_statuses(excluded_statuses=()), [])

    def test_reap_stale_leases(self):
        self.store.insert_or_update(make_video("lease_expired"), file_path="/tmp/expired.mp4")
        self.store.insert_or_update(make_video("lease_future"), file_path="/tmp/future.mp4")
        self.store.insert_or_update(make_video("lease_other"), file_path="/tmp/other.mp4")

        with self.store._connect() as conn:
            conn.execute("""
                UPDATE videos
                SET status = 'v2_in_progress',
                    processing_owner = 'worker-old',
                    lease_expires_at = datetime('now', '-1 minute')
                WHERE video_id = 'lease_expired'
            """)
            conn.execute("""
                UPDATE videos
                SET status = 'v2_in_progress',
                    processing_owner = 'worker-current',
                    lease_expires_at = datetime('now', '+1 minute')
                WHERE video_id = 'lease_future'
            """)
            conn.execute("""
                UPDATE videos
                SET status = 'v2_failed',
                    processing_owner = 'worker-failed',
                    lease_expires_at = datetime('now', '-1 minute')
                WHERE video_id = 'lease_other'
            """)

        reclaimed = self.store.reap_stale_leases()

        self.assertEqual(reclaimed, 1)
        expired = self.store.get_video_by_id("lease_expired", "youtube")
        future = self.store.get_video_by_id("lease_future", "youtube")
        other = self.store.get_video_by_id("lease_other", "youtube")
        self.assertEqual(expired["status"], "downloaded")
        self.assertIsNone(expired["processing_owner"])
        self.assertIsNone(expired["lease_expires_at"])
        self.assertEqual(future["status"], "v2_in_progress")
        self.assertEqual(future["processing_owner"], "worker-current")
        self.assertEqual(other["status"], "v2_failed")
        self.assertEqual(other["processing_owner"], "worker-failed")

    def test_get_pipeline_metrics(self):
        statuses = {
            "m_downloaded": "downloaded",
            "m_v1_passed": "v1_passed",
            "m_v2_in_progress": "v2_in_progress",
            "m_v2_passed": "v2_passed",
            "m_v2_failed": "v2_failed",
        }
        for video_id, status in statuses.items():
            self.store.insert_or_update(make_video(video_id), file_path=f"/tmp/{video_id}.mp4")
            self.store.update_status(video_id, "youtube", status)

        self.store.save_embeddings("m_v2_passed", "youtube", [np.zeros(512, dtype=np.float32)])
        self.store.save_checkpoint("m_downloaded", "youtube", "coarse", {"clips": []})
        self.store.save_checkpoint("m_v2_in_progress", "youtube", "fine", {"processed": [1]})

        self.store.create_pipeline_run("metrics-completed")
        self.store.complete_pipeline_run("metrics-completed")
        self.store.create_pipeline_run("metrics-failed")
        self.store.fail_pipeline_run("metrics-failed")
        self.store.create_pipeline_run("metrics-active")

        metrics = self.store.get_pipeline_metrics()

        self.assertEqual(metrics["total_videos"], 5)
        for status in statuses.values():
            self.assertEqual(metrics["by_status"].get(status, 0), 1)
        self.assertEqual(metrics["pipeline_runs"]["total"], 3)
        self.assertEqual(metrics["pipeline_runs"]["completed"], 1)
        self.assertEqual(metrics["pipeline_runs"]["failed"], 1)
        self.assertEqual(metrics["pipeline_runs"]["in_progress"], 1)
        self.assertEqual(metrics["embedding_count"], 1)
        self.assertEqual(metrics["checkpoint_count"], 2)

    def test_cleanup_orphaned_checkpoints(self):
        checkpoint_dir = OUTPUT_DIR / f"checkpoints_{uuid.uuid4().hex}"
        self.temp_dirs.append(checkpoint_dir)
        (checkpoint_dir / "deleted_video").mkdir(parents=True)
        (checkpoint_dir / "valid_video").mkdir(parents=True)
        (checkpoint_dir / "terminal_video").mkdir(parents=True)

        self.store.insert_or_update(make_video("deleted_video"), file_path="/tmp/deleted.mp4")
        self.store.save_checkpoint("deleted_video", "youtube", "coarse", {"clips": ["a.mp4"]})
        with self.store._connect() as conn:
            conn.execute("DELETE FROM videos WHERE video_id = 'deleted_video' AND platform = 'youtube'")

        self.store.insert_or_update(make_video("valid_video"), file_path="/tmp/valid.mp4")
        self.store.save_checkpoint("valid_video", "youtube", "coarse", {"clips": ["b.mp4"]})

        self.store.insert_or_update(make_video("terminal_video"), file_path="/tmp/terminal.mp4")
        self.store.save_checkpoint("terminal_video", "youtube", "coarse", {"clips": ["c.mp4"]})
        self.store.update_status("terminal_video", "youtube", "v2_passed")

        result = self.store.cleanup_orphaned_checkpoints(checkpoint_dir=str(checkpoint_dir))

        self.assertEqual(result["db_rows_deleted"], 2)
        self.assertEqual(result["dirs_removed"], 2)
        self.assertIsNone(self.store.load_checkpoint("deleted_video", "youtube", "coarse"))
        self.assertEqual(
            self.store.load_checkpoint("valid_video", "youtube", "coarse"),
            {"clips": ["b.mp4"]},
        )
        self.assertIsNone(self.store.load_checkpoint("terminal_video", "youtube", "coarse"))
        self.assertFalse((checkpoint_dir / "deleted_video").exists())
        self.assertTrue((checkpoint_dir / "valid_video").exists())
        self.assertFalse((checkpoint_dir / "terminal_video").exists())

    def test_prune_old_pipeline_runs(self):
        for index in range(5):
            run_id = f"completed-{index}"
            self.store.create_pipeline_run(run_id)
            self.store.complete_pipeline_run(run_id)
            with self.store._connect() as conn:
                conn.execute("""
                    UPDATE pipeline_run
                    SET created_at = ?,
                        updated_at = ?,
                        completed_at = ?
                    WHERE run_id = ?
                """, (
                    f"2026-01-01 00:00:0{index}",
                    f"2026-01-01 00:00:0{index}",
                    f"2026-01-01 00:00:0{index}",
                    run_id,
                ))
        self.store.create_pipeline_run("active-run")

        deleted = self.store.prune_old_pipeline_runs(keep_last=2)

        self.assertEqual(deleted, 3)
        remaining = self.store.list_pipeline_runs(limit=10)
        remaining_ids = {run["run_id"] for run in remaining}
        self.assertEqual(len(remaining), 3)
        self.assertEqual(remaining_ids, {"completed-3", "completed-4", "active-run"})

    def test_housekeeping_idempotent(self):
        checkpoint_dir = OUTPUT_DIR / f"checkpoints_{uuid.uuid4().hex}"
        self.temp_dirs.append(checkpoint_dir)
        (checkpoint_dir / "done").mkdir(parents=True)

        self.store.insert_or_update(make_video("stale"), file_path="/tmp/stale.mp4")
        self.store.insert_or_update(make_video("done"), file_path="/tmp/done.mp4")
        self.store.save_checkpoint("done", "youtube", "coarse", {"clips": []})
        self.store.update_status("done", "youtube", "v2_passed")
        with self.store._connect() as conn:
            conn.execute("""
                UPDATE videos
                SET status = 'v2_in_progress',
                    processing_owner = 'worker-old',
                    lease_expires_at = datetime('now', '-1 minute')
                WHERE video_id = 'stale'
            """)

        for index in range(3):
            run_id = f"old-run-{index}"
            self.store.create_pipeline_run(run_id)
            self.store.complete_pipeline_run(run_id)
            with self.store._connect() as conn:
                conn.execute("""
                    UPDATE pipeline_run
                    SET created_at = ?,
                        updated_at = ?,
                        completed_at = ?
                    WHERE run_id = ?
                """, (
                    f"2026-01-01 00:00:0{index}",
                    f"2026-01-01 00:00:0{index}",
                    f"2026-01-01 00:00:0{index}",
                    run_id,
                ))

        first = {
            "reaped": self.store.reap_stale_leases(),
            "cleaned": self.store.cleanup_orphaned_checkpoints(str(checkpoint_dir)),
            "pruned": self.store.prune_old_pipeline_runs(keep_last=1),
        }
        second = {
            "reaped": self.store.reap_stale_leases(),
            "cleaned": self.store.cleanup_orphaned_checkpoints(str(checkpoint_dir)),
            "pruned": self.store.prune_old_pipeline_runs(keep_last=1),
        }

        self.assertEqual(first["reaped"], 1)
        self.assertEqual(first["cleaned"], {"db_rows_deleted": 1, "dirs_removed": 1})
        self.assertEqual(first["pruned"], 2)
        self.assertEqual(second["reaped"], 0)
        self.assertEqual(second["cleaned"], {"db_rows_deleted": 0, "dirs_removed": 0})
        self.assertEqual(second["pruned"], 0)


if __name__ == "__main__":
    unittest.main()
