import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple

from aflutils.logger import get_logger
from core.models.video_meta import VideoMeta

logger = get_logger(__name__)


class VideoStore:
    """SQLite 存储视频元数据，支持插入、更新、查询"""

    def __init__(self, db_path: str = "data/videos.db"):
        self.db_path = db_path
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, platform)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON videos(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_platform ON videos(platform)
            """)
            logger.info(f"VideoStore initialized at {self.db_path}")

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
                    'downloaded'
                ))
            logger.debug(f"Stored video {video.video_id} from {video.platform}")
            return True
        except Exception as e:
            logger.error(f"Failed to store video {video.video_id}: {e}")
            return False

    def update_status(self, video_id: str, platform: str, status: str, fail_reason: str = None):
        """更新视频状态（供 V2 使用）"""
        with self._connect() as conn:
            conn.execute("""
                UPDATE videos SET status = ?, v2_fail_reason = ?, updated_at = CURRENT_TIMESTAMP
                WHERE video_id = ? AND platform = ?
            """, (status, fail_reason, video_id, platform))

    def get_pending_videos(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取状态为 'downloaded' 且尚未经过 V2 过滤的视频"""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM videos WHERE status = 'downloaded' ORDER BY created_at LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_video_by_id(self, video_id: str, platform: str) -> Optional[Dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM videos WHERE video_id = ? AND platform = ?
            """, (video_id, platform))
            row = cursor.fetchone()
            return dict(row) if row else None

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
        excluded_statuses: Tuple[str, ...] = ("downloaded", "v2_passed")
    ) -> List[str]:
        """获取不在排除状态中的 file_path 列表（去重后返回）。"""
        if not excluded_statuses:
            return []

        placeholders = ",".join("?" for _ in excluded_statuses)
        sql = f"""
            SELECT DISTINCT file_path FROM videos
            WHERE file_path IS NOT NULL
              AND file_path != ''
              AND status NOT IN ({placeholders})
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, list(excluded_statuses))
            return [row[0] for row in cursor.fetchall()]

    def claim_file_paths_by_excluded_statuses(
        self,
        excluded_statuses: Tuple[str, ...] = ("downloaded", "v2_passed")
    ) -> List[str]:
        """
        原子化领取可清理的视频文件路径：
        1) 在单个事务中查询可清理路径；
        2) 将这些路径对应记录的 file_path 置空，避免并发重复清理；
        3) 返回被领取的路径列表。
        """
        if not excluded_statuses:
            return []

        placeholders = ",".join("?" for _ in excluded_statuses)
        params = list(excluded_statuses)

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
