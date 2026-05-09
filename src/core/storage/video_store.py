from __future__ import annotations

import json
import shutil
import sqlite3
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple, Union

from aflutils.logger import get_logger
from core.models.video_meta import VideoMeta

logger = get_logger(__name__)


class VideoStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    V1_FAILED = "v1_failed"
    V1_PASSED = "v1_passed"
    V2_IN_PROGRESS = "v2_in_progress"
    V2_PASSED = "v2_passed"
    V2_FAILED = "v2_failed"
    V2_COARSE_FAILED = "v2_coarse_failed"
    V2_FINE_FAILED = "v2_fine_failed"


class VideoStore:
    """SQLite 存储视频元数据，支持插入、更新、查询"""

    def __init__(self, db_path: str = "data/videos.db", max_retries: int = 3):
        self.db_path = db_path
        self.max_retries = max_retries
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """创建带基础保护配置的 SQLite 连接。"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        """初始化数据库表"""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    title TEXT,
                    duration_seconds INTEGER,
                    resolution TEXT,
                    channel TEXT,
                    publish_date TEXT,
                    view_count INTEGER,
                    tags TEXT,               -- JSON 数组
                    search_term TEXT,
                    file_path TEXT,          -- 本地下载路径
                    status TEXT DEFAULT 'downloaded',  -- downloaded, v2_passed, v2_failed
                    v2_fail_reason TEXT,
                    retry_count INTEGER DEFAULT 0,
                    processing_owner TEXT,
                    processing_started_at TEXT,
                    lease_expires_at TEXT,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, platform)
                )
            """)
            self._init_embeddings_table(conn)
            self._init_v2_progress_table(conn)
            self._init_pipeline_run_table(conn)
            self._migrate_schema(conn)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON videos(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_platform ON videos(platform)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_videos_claim ON videos(status, lease_expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_videos_retry ON videos(status, retry_count)
            """)
            logger.info(f"VideoStore initialized at {self.db_path}")

    def _init_embeddings_table(self, conn: sqlite3.Connection):
        """Initialize the derived face embeddings table used to rebuild FAISS."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                label TEXT,
                embedding_blob BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id, platform, label)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_video
            ON embeddings(video_id, platform)
        """)

    def _init_v2_progress_table(self, conn: sqlite3.Connection):
        """Initialize restart checkpoints for per-video V2 progress."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS v2_progress (
                video_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id, platform, stage)
            )
        """)

    def _init_pipeline_run_table(self, conn: sqlite3.Connection):
        """Initialize restart checkpoints for whole-pipeline runs."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_run (
                run_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_progress',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pipeline_run_status
            ON pipeline_run(status, updated_at)
        """)

    def _migrate_schema(self, conn: sqlite3.Connection):
        """Apply additive migrations for existing SQLite databases."""
        cursor = conn.execute("PRAGMA table_info(videos)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        migrations = {
            "retry_count": "ALTER TABLE videos ADD COLUMN retry_count INTEGER DEFAULT 0",
            "processing_owner": "ALTER TABLE videos ADD COLUMN processing_owner TEXT",
            "processing_started_at": "ALTER TABLE videos ADD COLUMN processing_started_at TEXT",
            "lease_expires_at": "ALTER TABLE videos ADD COLUMN lease_expires_at TEXT",
            "last_error": "ALTER TABLE videos ADD COLUMN last_error TEXT",
        }
        for column, ddl in migrations.items():
            if column not in existing_columns:
                conn.execute(ddl)

    @staticmethod
    def _normalize_status(status: Union[str, VideoStatus]) -> str:
        if isinstance(status, VideoStatus):
            return status.value
        if isinstance(status, str):
            normalized = status.strip().lower()
            if normalized in {s.value for s in VideoStatus}:
                return normalized
            logger.warning(f"Unknown status '{status}', storing as-is for backward compatibility")
            return status
        raise TypeError(f"status must be str or VideoStatus, got {type(status).__name__}")

    @staticmethod
    def _deserialize_tags(record: Dict[str, Any]) -> Dict[str, Any]:
        tags = record.get("tags")
        if isinstance(tags, str):
            try:
                record["tags"] = json.loads(tags)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON tags in record, keeping raw string")
        return record

    @staticmethod
    def _row_to_video_meta(record: Dict[str, Any]) -> VideoMeta:
        record = VideoStore._deserialize_tags(record)
        tags = record.get("tags") if isinstance(record.get("tags"), list) else []
        meta_fields = {
            "video_id", "url", "platform", "title", "duration_seconds", "resolution",
            "channel", "publish_date", "view_count", "tags", "search_term",
        }
        extra = {key: value for key, value in record.items() if key not in meta_fields}
        return VideoMeta(
            video_id=record["video_id"],
            url=record["url"],
            platform=record["platform"],
            title=record.get("title"),
            duration_seconds=record.get("duration_seconds"),
            resolution=record.get("resolution"),
            channel=record.get("channel"),
            publish_date=record.get("publish_date"),
            view_count=record.get("view_count"),
            tags=tags,
            search_term=record.get("search_term"),
            extra=extra,
        )

    def set_max_retries(self, max_retries: int):
        self.max_retries = max(0, int(max_retries))

    @staticmethod
    def _terminal_v2_statuses() -> Tuple[str, ...]:
        return (
            VideoStatus.V2_PASSED.value,
            VideoStatus.V2_FAILED.value,
            VideoStatus.V2_COARSE_FAILED.value,
            VideoStatus.V2_FINE_FAILED.value,
        )

    def create_pipeline_run(self, run_id: str, stage: str = "search") -> None:
        """Create a new in-progress pipeline run checkpoint."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO pipeline_run (
                    run_id, stage, status, created_at, updated_at, completed_at
                ) VALUES (?, ?, 'in_progress', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)
            """, (run_id, stage))
        logger.info(f"Created pipeline run {run_id} at stage {stage}")

    def update_pipeline_run_stage(
        self,
        run_id: str,
        stage: str,
        status: str = "in_progress"
    ) -> None:
        """Update the checkpoint stage for an existing pipeline run."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE pipeline_run
                SET stage = ?,
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """, (stage, status, run_id))
        logger.debug(f"Updated pipeline run {run_id} to stage {stage} with status {status}")

    def complete_pipeline_run(self, run_id: str) -> None:
        """Mark a pipeline run as completed."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE pipeline_run
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """, (run_id,))
        logger.info(f"Completed pipeline run {run_id}")

    def fail_pipeline_run(self, run_id: str) -> None:
        """Mark a pipeline run as failed."""
        with self._connect() as conn:
            conn.execute("""
                UPDATE pipeline_run
                SET status = 'failed',
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """, (run_id,))
        logger.info(f"Marked pipeline run {run_id} as failed")

    def get_in_progress_run(self) -> Optional[Dict[str, Any]]:
        """Return the most recent in-progress pipeline run, if any."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT *
                FROM pipeline_run
                WHERE status = 'in_progress'
                ORDER BY updated_at DESC, created_at DESC, rowid DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return a pipeline run by id."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT *
                FROM pipeline_run
                WHERE run_id = ?
            """, (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_pipeline_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Compatibility alias for pipeline run lookup."""
        return self.get_run_by_id(run_id)

    def delete_pipeline_run(self, run_id: str) -> bool:
        """Delete a pipeline run checkpoint by id."""
        with self._connect() as conn:
            cursor = conn.execute("""
                DELETE FROM pipeline_run
                WHERE run_id = ?
            """, (run_id,))
            return cursor.rowcount == 1

    def prune_old_pipeline_runs(self, keep_last: int = 100) -> int:
        """Delete older completed/failed pipeline runs while preserving in-progress runs."""
        keep_last = max(0, int(keep_last))
        with self._connect() as conn:
            cursor = conn.execute("""
                DELETE FROM pipeline_run
                WHERE status IN ('completed', 'failed')
                  AND run_id NOT IN (
                      SELECT run_id
                      FROM pipeline_run
                      WHERE status IN ('completed', 'failed')
                      ORDER BY
                          COALESCE(completed_at, updated_at, created_at) DESC,
                          updated_at DESC,
                          created_at DESC,
                          run_id DESC
                      LIMIT ?
                  )
            """, (keep_last,))
            deleted = cursor.rowcount
        logger.info(f"Pruned {deleted} old pipeline runs")
        return deleted

    def list_pipeline_runs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List recent pipeline run checkpoints, optionally filtered by status."""
        limit = max(1, int(limit))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if status is None:
                cursor = conn.execute("""
                    SELECT *
                    FROM pipeline_run
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                """, (limit,))
            else:
                cursor = conn.execute("""
                    SELECT *
                    FROM pipeline_run
                    WHERE status = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                """, (status, limit))
            return [dict(row) for row in cursor.fetchall()]

    def save_embeddings(
        self,
        video_id: str,
        platform: str,
        embeddings: list[np.ndarray],
        label: str = "face"
    ) -> int:
        """
        Persist ArcFace embeddings as float32 blobs.

        The embeddings table is keyed by (video_id, platform, label). When a
        batch has multiple embeddings, deterministic label suffixes keep each
        row addressable while repeated saves remain idempotent.
        """
        import numpy as np

        if not embeddings:
            return 0

        rows = []
        multiple = len(embeddings) > 1
        for index, embedding in enumerate(embeddings):
            array = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if array.shape != (512,):
                raise ValueError(f"embedding must have shape (512,), got {array.shape}")
            row_label = f"{label}:{index}" if multiple else label
            rows.append((video_id, platform, row_label, sqlite3.Binary(array.tobytes())))

        inserted = 0
        with self._connect() as conn:
            for row in rows:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO embeddings (
                        video_id, platform, label, embedding_blob
                    ) VALUES (?, ?, ?, ?)
                """, row)
                inserted += cursor.rowcount
        return inserted

    def load_all_embeddings(self) -> tuple[list[str], list[np.ndarray]]:
        """Load every persisted embedding for rebuilding the derived FAISS index."""
        import numpy as np

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT label, embedding_blob
                FROM embeddings
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()

        labels = []
        embeddings = []
        for row in rows:
            embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32).copy().reshape(512,)
            labels.append(row["label"])
            embeddings.append(embedding)
        return labels, embeddings

    def get_embedding_count(self) -> int:
        """Return the total number of persisted embeddings."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            return int(cursor.fetchone()[0])

    def reap_stale_leases(self) -> int:
        """Return expired V2 leases to the downloaded queue."""
        with self._connect() as conn:
            cursor = conn.execute("""
                UPDATE videos
                SET status = ?,
                    processing_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = ?
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < datetime('now')
            """, (
                VideoStatus.DOWNLOADED.value,
                VideoStatus.V2_IN_PROGRESS.value,
            ))
            reclaimed = cursor.rowcount
        logger.info(f"Reclaimed {reclaimed} stale V2 leases")
        return reclaimed

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Return read-only aggregate metrics for videos, runs, and checkpoints."""
        with self._connect() as conn:
            video_total = int(conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0])
            status_rows = conn.execute("""
                SELECT status, COUNT(*)
                FROM videos
                GROUP BY status
            """).fetchall()
            run_total = int(conn.execute("SELECT COUNT(*) FROM pipeline_run").fetchone()[0])
            run_rows = conn.execute("""
                SELECT status, COUNT(*)
                FROM pipeline_run
                GROUP BY status
            """).fetchall()
            embedding_count = int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            checkpoint_count = int(conn.execute("SELECT COUNT(*) FROM v2_progress").fetchone()[0])

        run_counts = {row[0]: int(row[1]) for row in run_rows}
        return {
            "total_videos": video_total,
            "by_status": {row[0]: int(row[1]) for row in status_rows},
            "pipeline_runs": {
                "total": run_total,
                "completed": run_counts.get("completed", 0),
                "failed": run_counts.get("failed", 0),
                "in_progress": run_counts.get("in_progress", 0),
            },
            "embedding_count": embedding_count,
            "checkpoint_count": checkpoint_count,
        }

    def cleanup_orphaned_checkpoints(self, checkpoint_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Delete V2 checkpoint rows for missing or terminal videos.

        When a checkpoint directory is supplied, remove per-video checkpoint
        directories that no longer have any active non-terminal video row.
        """
        terminal_statuses = self._terminal_v2_statuses()
        placeholders = ",".join("?" for _ in terminal_statuses)

        with self._connect() as conn:
            cursor = conn.execute(f"""
                DELETE FROM v2_progress
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM videos
                    WHERE videos.video_id = v2_progress.video_id
                      AND videos.platform = v2_progress.platform
                )
                   OR EXISTS (
                    SELECT 1
                    FROM videos
                    WHERE videos.video_id = v2_progress.video_id
                      AND videos.platform = v2_progress.platform
                      AND videos.status IN ({placeholders})
                )
            """, terminal_statuses)
            db_rows_deleted = cursor.rowcount

            active_rows = conn.execute(f"""
                SELECT DISTINCT video_id
                FROM videos
                WHERE status NOT IN ({placeholders})
            """, terminal_statuses).fetchall()
            active_video_ids = {row[0] for row in active_rows}

        dirs_removed = 0
        if checkpoint_dir:
            root = Path(checkpoint_dir)
            if root.exists():
                for child in root.iterdir():
                    if not child.is_dir():
                        continue
                    if child.name in active_video_ids:
                        continue
                    shutil.rmtree(child)
                    dirs_removed += 1

        result = {"db_rows_deleted": db_rows_deleted, "dirs_removed": dirs_removed}
        logger.info(f"Cleaned orphaned V2 checkpoints: {result}")
        return result

    def save_checkpoint(
        self,
        video_id: str,
        platform: str,
        stage: str,
        progress_data: Dict[str, Any]
    ) -> None:
        """Upsert a JSON checkpoint for one V2 processing stage."""
        payload = json.dumps(progress_data)
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO v2_progress (
                    video_id, platform, stage, progress_data, updated_at
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(video_id, platform, stage) DO UPDATE SET
                    progress_data = excluded.progress_data,
                    updated_at = CURRENT_TIMESTAMP
            """, (video_id, platform, stage, payload))
        logger.debug(f"Saved V2 checkpoint {stage} for {platform}:{video_id}")

    def load_checkpoint(
        self,
        video_id: str,
        platform: str,
        stage: str
    ) -> Optional[Dict[str, Any]]:
        """Load a JSON checkpoint for one V2 processing stage."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT progress_data
                FROM v2_progress
                WHERE video_id = ? AND platform = ? AND stage = ?
            """, (video_id, platform, stage))
            row = cursor.fetchone()
        if row is None:
            return None
        progress_data = row["progress_data"]
        if not progress_data:
            return {}
        try:
            checkpoint = json.loads(progress_data)
            logger.debug(f"Loaded V2 checkpoint {stage} for {platform}:{video_id}")
            return checkpoint
        except json.JSONDecodeError:
            logger.warning(f"Invalid V2 checkpoint JSON for {platform}:{video_id}:{stage}")
            return None

    def delete_checkpoint(self, video_id: str, platform: str) -> None:
        """Delete every V2 checkpoint for a video."""
        with self._connect() as conn:
            conn.execute("""
                DELETE FROM v2_progress
                WHERE video_id = ? AND platform = ?
            """, (video_id, platform))
        logger.debug(f"Deleted V2 checkpoints for {platform}:{video_id}")

    def delete_stage_checkpoint(self, video_id: str, platform: str, stage: str) -> None:
        """Delete one V2 checkpoint stage for a video."""
        with self._connect() as conn:
            conn.execute("""
                DELETE FROM v2_progress
                WHERE video_id = ? AND platform = ? AND stage = ?
            """, (video_id, platform, stage))
        logger.debug(f"Deleted V2 checkpoint {stage} for {platform}:{video_id}")

    def insert_or_update(self, video: VideoMeta, file_path: Optional[str] = None) -> bool:
        """插入或更新视频记录（基于 video_id + platform）"""
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO videos (
                        video_id, url, platform, title, duration_seconds, resolution,
                        channel, publish_date, view_count, tags, search_term,
                        file_path, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(video_id, platform) DO UPDATE SET
                        url = excluded.url,
                        title = excluded.title,
                        duration_seconds = excluded.duration_seconds,
                        resolution = excluded.resolution,
                        channel = excluded.channel,
                        publish_date = excluded.publish_date,
                        view_count = excluded.view_count,
                        tags = excluded.tags,
                        search_term = excluded.search_term,
                        file_path = COALESCE(excluded.file_path, videos.file_path),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    video.video_id,
                    video.url,
                    video.platform,
                    video.title,
                    video.duration_seconds,
                    video.resolution,
                    video.channel,
                    video.publish_date,
                    video.view_count,
                    json.dumps(video.tags),  # 存储为 JSON
                    video.search_term,
                    file_path,
                    VideoStatus.DOWNLOADED.value
                ))
            logger.debug(f"Stored video {video.video_id} from {video.platform}")
            return True
        except Exception as e:
            logger.error(f"Failed to store video {video.video_id}: {e}")
            return False

    def insert_or_update_video(self, video: VideoMeta, file_path: Optional[str] = None) -> bool:
        """Compatibility alias for callers using the older explicit method name."""
        return self.insert_or_update(video, file_path=file_path)

    def update_status(
        self,
        video_id: str,
        platform: str,
        status: Union[str, VideoStatus],
        fail_reason: str = None
    ):
        """更新视频状态（供 V2 使用）"""
        normalized_status = self._normalize_status(status)
        with self._connect() as conn:
            conn.execute("""
                UPDATE videos
                SET status = ?,
                    v2_fail_reason = ?,
                    last_error = COALESCE(?, last_error),
                    processing_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = ? AND platform = ?
            """, (normalized_status, fail_reason, fail_reason, video_id, platform))

    def get_pending_videos(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取状态为 'downloaded' 且尚未经过 V2 过滤的视频"""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM videos WHERE status = ? ORDER BY created_at LIMIT ?
            """, (VideoStatus.DOWNLOADED.value, limit))
            rows = cursor.fetchall()
            return [self._deserialize_tags(dict(row)) for row in rows]

    def claim_pending_videos(
        self,
        limit: int,
        owner: str,
        lease_seconds: int
    ) -> List[VideoMeta]:
        """
        Atomically claim downloaded or expired in-progress videos for V2 processing.
        """
        if limit <= 0:
            return []

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute("""
                SELECT id
                FROM videos
                WHERE (
                    status = ?
                    AND (lease_expires_at IS NULL OR lease_expires_at < datetime('now'))
                ) OR (
                    status = ?
                    AND lease_expires_at < datetime('now')
                )
                ORDER BY
                    COALESCE(retry_count, 0) ASC,
                    processing_started_at IS NOT NULL ASC,
                    processing_started_at ASC,
                    id ASC
                LIMIT ?
            """, (
                VideoStatus.DOWNLOADED.value,
                VideoStatus.V2_IN_PROGRESS.value,
                int(limit),
            ))
            ids = [row["id"] for row in cursor.fetchall()]
            if not ids:
                return []

            placeholders = ",".join("?" for _ in ids)
            conn.execute(f"""
                UPDATE videos
                SET status = ?,
                    processing_owner = ?,
                    processing_started_at = datetime('now'),
                    lease_expires_at = datetime('now', '+' || ? || ' seconds'),
                    retry_count = COALESCE(retry_count, 0) + 1,
                    last_error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
            """, (
                VideoStatus.V2_IN_PROGRESS.value,
                owner,
                int(lease_seconds),
                *ids,
            ))
            cursor = conn.execute(f"""
                SELECT * FROM videos
                WHERE id IN ({placeholders})
            """, ids)
            rows = cursor.fetchall()
            rows_by_id = {row["id"]: dict(row) for row in rows}
            return [self._row_to_video_meta(rows_by_id[row_id]) for row_id in ids]

    def complete_v2(
        self,
        video_id: str,
        platform: str,
        final_status: Union[str, VideoStatus],
        expected_status: Union[str, VideoStatus] = VideoStatus.V2_IN_PROGRESS.value
    ) -> bool:
        """Complete a claimed V2 row only if it is still in the expected state."""
        normalized_final = self._normalize_status(final_status)
        normalized_expected = self._normalize_status(expected_status)
        with self._connect() as conn:
            cursor = conn.execute("""
                UPDATE videos
                SET status = ?,
                    processing_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = ? AND platform = ? AND status = ?
            """, (normalized_final, video_id, platform, normalized_expected))
            return cursor.rowcount == 1

    def fail_v2(
        self,
        video_id: str,
        platform: str,
        error: str,
        retry_within_seconds: Optional[int] = None
    ) -> bool:
        """Record a V2 processing failure, optionally releasing it for retry."""
        error_text = str(error)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT retry_count FROM videos WHERE video_id = ? AND platform = ?
            """, (video_id, platform))
            row = cursor.fetchone()
            if row is None:
                return False

            retry_count = row["retry_count"] or 0
            if retry_within_seconds is not None and retry_count < self.max_retries:
                cursor = conn.execute("""
                    UPDATE videos
                    SET status = ?,
                        processing_owner = NULL,
                        lease_expires_at = datetime('now', '+' || ? || ' seconds'),
                        last_error = ?,
                        v2_fail_reason = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE video_id = ? AND platform = ?
                """, (
                    VideoStatus.DOWNLOADED.value,
                    int(retry_within_seconds),
                    error_text,
                    error_text,
                    video_id,
                    platform,
                ))
            else:
                cursor = conn.execute("""
                    UPDATE videos
                    SET status = ?,
                        processing_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = ?,
                        v2_fail_reason = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE video_id = ? AND platform = ?
                """, (
                    VideoStatus.V2_FAILED.value,
                    error_text,
                    error_text,
                    video_id,
                    platform,
                ))
            return cursor.rowcount == 1

    def get_video_by_id(self, video_id: str, platform: str) -> Optional[Dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM videos WHERE video_id = ? AND platform = ?
            """, (video_id, platform))
            row = cursor.fetchone()
            return self._deserialize_tags(dict(row)) if row else None

    def get_statistics(self) -> Dict[str, int]:
        """获取各状态的数量统计"""
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count FROM videos GROUP BY status
            """)
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_existing_video_ids(self, platform: str, video_ids: List[str]) -> Set[str]:
        """批量查询数据库中已存在的视频 ID（按平台过滤）。"""
        normalized_ids = [vid for vid in video_ids if vid]
        if not normalized_ids:
            return set()

        placeholders = ",".join("?" for _ in normalized_ids)
        sql = f"""
            SELECT video_id FROM videos
            WHERE platform = ? AND video_id IN ({placeholders})
        """
        params = [platform, *normalized_ids]
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            return {row[0] for row in cursor.fetchall()}

    def get_file_paths_by_excluded_statuses(
        self,
        excluded_statuses: Tuple[Union[str, VideoStatus], ...] = (
            VideoStatus.DOWNLOADED.value, VideoStatus.V2_PASSED.value
        )
    ) -> List[str]:
        """获取不在排除状态中的 file_path 列表（去重后返回）。"""
        if not excluded_statuses:
            return []

        normalized_excluded = tuple(self._normalize_status(s) for s in excluded_statuses)
        placeholders = ",".join("?" for _ in normalized_excluded)
        sql = f"""
            SELECT DISTINCT file_path FROM videos
            WHERE file_path IS NOT NULL
              AND file_path != ''
              AND status NOT IN ({placeholders})
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, list(normalized_excluded))
            return [row[0] for row in cursor.fetchall()]

    def claim_file_paths_by_excluded_statuses(
        self,
        excluded_statuses: Tuple[Union[str, VideoStatus], ...] = (
            VideoStatus.DOWNLOADED.value, VideoStatus.V2_PASSED.value
        )
    ) -> List[str]:
        """
        原子化领取可清理的视频文件路径：
        1) 在单个事务中查询可清理路径；
        2) 将这些路径对应记录的 file_path 置空，避免并发重复清理；
        3) 返回被领取的路径列表。
        """
        if not excluded_statuses:
            return []

        normalized_excluded = tuple(self._normalize_status(s) for s in excluded_statuses)
        placeholders = ",".join("?" for _ in normalized_excluded)
        params = list(normalized_excluded)

        select_sql = f"""
            SELECT DISTINCT v.file_path
            FROM videos v
            WHERE v.file_path IS NOT NULL
              AND v.file_path != ''
              AND v.status NOT IN ({placeholders})
              AND NOT EXISTS (
                    SELECT 1
                    FROM videos keep
                    WHERE keep.file_path = v.file_path
                      AND keep.status IN ({placeholders})
              )
        """

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(select_sql, params + params)
            claimed_paths = [row[0] for row in cursor.fetchall()]
            if not claimed_paths:
                return []

            update_path_placeholders = ",".join("?" for _ in claimed_paths)
            update_sql = f"""
                UPDATE videos
                SET file_path = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE file_path IN ({update_path_placeholders})
                  AND status NOT IN ({placeholders})
            """
            conn.execute(update_sql, claimed_paths + params)
            return claimed_paths
