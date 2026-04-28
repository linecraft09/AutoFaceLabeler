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
import sqlite3
import sys
import unittest
import uuid
from pathlib import Path

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

    def tearDown(self):
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


if __name__ == "__main__":
    unittest.main()
