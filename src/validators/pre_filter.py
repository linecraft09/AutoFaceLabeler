# src/validators/pre_filter.py
import re
from typing import List, Dict, Any, Tuple

from aflutils.logger import get_logger
from core.models.video_meta import VideoMeta

logger = get_logger(__name__)


class PreFilter:
    """基于元数据的预筛选器 (V1)"""

    def __init__(self, config: Dict[str, Any]):
        """
        config 包含 filters 部分，如:
        {
            'min_duration': 30,
            'max_duration': 1800,
            'min_resolution': 720,
            'title_blacklist': [...],
            'channel_blacklist': [...],
            'min_views': 1000,
            'tags_whitelist': [...]   # 可选
        }
        """
        self.filters = config.get('filters', {})
        self.min_dur = self.filters.get('min_duration', 30)
        self.max_dur = self.filters.get('max_duration', 1800)
        self.min_res = self.filters.get('min_resolution', 720)
        self.title_blacklist = [re.compile(pat, re.IGNORECASE) for pat in self.filters.get('title_blacklist', [])]
        self.channel_blacklist = [re.compile(pat, re.IGNORECASE) for pat in self.filters.get('channel_blacklist', [])]
        self.min_views = self.filters.get('min_views', 0)
        self.tags_whitelist = [tag.lower() for tag in self.filters.get('tags_whitelist', [])]

    def filter(self, videos: List[VideoMeta], search_term: str) -> Tuple[List[VideoMeta], Dict[str, Any]]:
        """
        对一批视频进行预筛选
        :return: (passed_videos, feedback)
        """
        passed = []
        fail_reasons = {
            'duration': 0,
            'resolution': 0,
            'title_blacklist': 0,
            'channel_blacklist': 0,
            'views': 0,
            'other': 0,
        }
        for v in videos:
            reason = self._check_video(v)
            if reason is None:
                passed.append(v)
            else:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        feedback = {
            'search_term': search_term,
            'platform': videos[0].platform if videos else 'unknown',
            'total_received': len(videos),
            'pass_count': len(passed),
            'v1_pass_rate': len(passed) / len(videos) if videos else 0,
            'fail_reasons': fail_reasons,
            'candidate_urls': [v.url for v in passed],
        }
        logger.info(f"PreFilter for '{search_term}': {len(passed)}/{len(videos)} passed")
        return passed, feedback

    def _check_video(self, video: VideoMeta) -> str | None:
        """返回 None 表示通过，否则返回失败原因 key"""
        # 时长
        if not (self.min_dur <= video.duration_seconds <= self.max_dur):
            return 'duration'
        # 分辨率
        res = self._parse_resolution(video.resolution)
        if res < self.min_res:
            return 'resolution'
        # 标题黑名单
        for pat in self.title_blacklist:
            if pat.search(video.title):
                return 'title_blacklist'
        # 频道黑名单
        for pat in self.channel_blacklist:
            if pat.search(video.channel):
                return 'channel_blacklist'
        # 播放量
        if video.view_count < self.min_views:
            return 'views'
        # 可选：标签白名单（非强制，但可记录为加分，此处不过滤）
        # 若需要严格过滤，可取消下面注释
        # if self.tags_whitelist:
        #     video_tags_lower = [t.lower() for t in video.tags]
        #     if not any(w in video_tags_lower for w in self.tags_whitelist):
        #         return 'tags'
        return None

    @staticmethod
    def _parse_resolution(res_str: str) -> int:
        """从 '1080p', '720p', 'unknown' 中提取数字"""
        if not res_str:
            return 0
        match = re.search(r'(\d+)p', res_str.lower())
        return int(match.group(1)) if match else 0
