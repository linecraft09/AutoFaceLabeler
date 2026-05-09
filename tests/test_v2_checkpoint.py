import os
import sqlite3
import sys

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from core.models.video_meta import VideoMeta
from core.storage.video_store import VideoStore


def make_video(video_id: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://example.com/{video_id}",
        platform="youtube",
        title=f"title {video_id}",
        duration_seconds=60,
        resolution="720p",
        channel="channel",
        publish_date="2026-01-01",
        view_count=100,
        tags=[],
        search_term="term",
        extra={},
    )


@pytest.fixture
def store(tmp_path):
    return VideoStore(str(tmp_path / "videos.db"))


def insert_downloaded(store: VideoStore, *video_ids: str):
    for video_id in video_ids:
        assert store.insert_or_update(make_video(video_id), file_path=f"/tmp/{video_id}.mp4")


def db_row(store: VideoStore, video_id: str):
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM videos WHERE video_id = ? AND platform = 'youtube'",
            (video_id,),
        ).fetchone()
        return dict(row)


def test_claim_grabs_downloaded_rows(store):
    insert_downloaded(store, "v1", "v2", "v3")

    claimed = store.claim_pending_videos(limit=2, owner="worker-a", lease_seconds=600)

    assert len(claimed) == 2
    claimed_ids = {video.video_id for video in claimed}
    assert claimed_ids <= {"v1", "v2", "v3"}
    for video_id in claimed_ids:
        row = db_row(store, video_id)
        assert row["status"] == "v2_in_progress"
        assert row["processing_owner"] == "worker-a"
        assert row["retry_count"] == 1

    unclaimed_ids = {"v1", "v2", "v3"} - claimed_ids
    assert len(unclaimed_ids) == 1
    assert db_row(store, unclaimed_ids.pop())["status"] == "downloaded"


def test_claim_skips_in_progress_with_valid_lease(store):
    insert_downloaded(store, "v1")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("""
            UPDATE videos
            SET status = 'v2_in_progress',
                lease_expires_at = datetime('now', '+10 minutes')
            WHERE video_id = 'v1'
        """)

    claimed = store.claim_pending_videos(limit=1, owner="worker-b", lease_seconds=600)

    assert claimed == []
    assert db_row(store, "v1")["status"] == "v2_in_progress"


def test_claim_retakes_expired_lease(store):
    insert_downloaded(store, "v1")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("""
            UPDATE videos
            SET status = 'v2_in_progress',
                retry_count = 1,
                processing_owner = 'old-worker',
                processing_started_at = datetime('now', '-20 minutes'),
                lease_expires_at = datetime('now', '-10 minutes')
            WHERE video_id = 'v1'
        """)

    claimed = store.claim_pending_videos(limit=1, owner="worker-c", lease_seconds=600)

    assert [video.video_id for video in claimed] == ["v1"]
    row = db_row(store, "v1")
    assert row["processing_owner"] == "worker-c"
    assert row["retry_count"] == 2


def test_complete_v2_guards_expected_status(store):
    insert_downloaded(store, "v1")

    assert not store.complete_v2("v1", "youtube", "v2_passed")
    assert db_row(store, "v1")["status"] == "downloaded"

    store.update_status("v1", "youtube", "v2_in_progress")
    assert store.complete_v2("v1", "youtube", "v2_passed")
    assert db_row(store, "v1")["status"] == "v2_passed"


def test_fail_v2_retries_then_gives_up(tmp_path):
    store = VideoStore(str(tmp_path / "videos.db"), max_retries=2)
    insert_downloaded(store, "v1")

    assert len(store.claim_pending_videos(limit=1, owner="worker-a", lease_seconds=600)) == 1
    assert store.fail_v2("v1", "youtube", error="first failure", retry_within_seconds=60)
    row = db_row(store, "v1")
    assert row["status"] == "downloaded"
    assert row["last_error"] == "first failure"

    with sqlite3.connect(store.db_path) as conn:
        conn.execute("""
            UPDATE videos
            SET lease_expires_at = datetime('now', '-1 second')
            WHERE video_id = 'v1'
        """)
    assert len(store.claim_pending_videos(limit=1, owner="worker-a", lease_seconds=600)) == 1
    assert store.fail_v2("v1", "youtube", error="second failure", retry_within_seconds=60)

    row = db_row(store, "v1")
    assert row["status"] == "v2_failed"
    assert row["last_error"] == "second failure"


def test_claim_orders_by_retry_count(store):
    insert_downloaded(store, "high", "low", "mid")
    with sqlite3.connect(store.db_path) as conn:
        conn.executemany(
            "UPDATE videos SET retry_count = ? WHERE video_id = ?",
            [(5, "high"), (0, "low"), (2, "mid")],
        )

    claimed = store.claim_pending_videos(limit=2, owner="worker-d", lease_seconds=600)

    assert [video.video_id for video in claimed] == ["low", "mid"]
