# src/core/models/video_meta.py
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class VideoMeta:
    video_id: str
    url: str
    platform: str  # "youtube" or "bilibili"
    title: str
    duration_seconds: int
    resolution: str  # e.g. "1080p", "720p"
    channel: str
    publish_date: str  # YYYY-MM-DD
    view_count: int
    tags: List[str]
    search_term: str  # 产生该视频的搜索词
    # 后续过滤会添加额外字段
    extra: Dict = None
