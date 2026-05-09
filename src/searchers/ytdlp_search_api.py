import random
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError

from aflutils.logger import get_logger
from core.models.video_meta import VideoMeta
from core.storage.video_store import VideoStore
from .search_api import SearchApi

logger = get_logger(__name__)


class SearchRateLimitError(Exception):
    pass


class SearchNetworkError(Exception):
    pass


class YtDlpSearchApi(SearchApi):
    """基于 yt-dlp 的搜索实现 (支持 YouTube 和 BiliBili)，采用混合策略获取分辨率"""

    def __init__(self, platform: str, proxy: str = None, user_agent: str = None,
                 video_store: Optional[VideoStore] = None, search_config: Optional[dict] = None,
                 cookies: str = None):
        """
        :param platform: 'youtube' 或 'bilibili'
        :param cookies: 可选的 cookies.txt 文件路径（用于 yt-dlp 认证访问）
        """
        self.platform = platform.lower()
        self.video_store = video_store
        self.search_config = search_config or {}
        self._cookies = cookies or self.search_config.get('cookies')
        if self.platform not in ['youtube', 'bilibili']:
            raise ValueError("platform must be 'youtube' or 'bilibili'")

        fast_cfg = self.search_config.get('fast_search', {})
        detail_cfg = self.search_config.get('detail_fetch', {})
        # 基础配置（用于快速搜索，extract_flat=True）
        self.ydl_opts_fast = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,  # 仅获取元数据，不下载
            'force_generic_extractor': False,
            # 反检测配置
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36',
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
            },
            'sleep_interval': random.uniform(
                fast_cfg.get('sleep_interval_min', 3),
                fast_cfg.get('sleep_interval_max', 6),
            ),
            'max_sleep_interval': fast_cfg.get('max_sleep_interval', 10),
            'sleep_interval_requests': random.uniform(2, 5),
            'nocheckcertificate': False,
            'prefer_insecure': False,
            'extractor_retries': fast_cfg.get('extractor_retries', 3),
            'file_access_retries': fast_cfg.get('file_access_retries', 3),
            "socket_timeout": fast_cfg.get('socket_timeout', 30),
            'remote_components': ['ejs:github'],
        }
        if proxy:
            self.ydl_opts_fast['proxy'] = proxy
        if user_agent:
            self.ydl_opts_fast['user_agent'] = user_agent
        if self._cookies:
            self.ydl_opts_fast['cookiefile'] = self._cookies

        # 详细配置（用于获取单个视频详情，extract_flat=False）
        self.ydl_opts_detail = copy.deepcopy(self.ydl_opts_fast)
        self.ydl_opts_detail['extract_flat'] = False
        if self._cookies:
            self.ydl_opts_detail['cookiefile'] = self._cookies
        self.ydl_opts_detail['remote_components'] = ['ejs:github']
        # 详细模式可能需要更长的超时
        self.ydl_opts_detail['socket_timeout'] = detail_cfg.get('socket_timeout', 60)

    def get_platform(self) -> str:
        return self.platform

    def search(self, query: str, max_results: int = 50) -> List[VideoMeta]:
        """
        混合策略搜索：
        1. 快速扁平搜索，获取视频列表（无分辨率）。
        2. 对每个视频单独请求详情，提取真实分辨率。
        """
        max_results = max(1, min(max_results, 50))

        if self.platform == 'youtube':
            search_query = f"ytsearch{max_results}:{query}"
        else:  # bilibili
            search_query = f"bilisearch{max_results}:{query}"

        # 阶段1：快速搜索，获取扁平条目列表
        with yt_dlp.YoutubeDL(self.ydl_opts_fast) as ydl:
            try:
                info = ydl.extract_info(search_query, download=False)
                entries = info.get('entries', [])
                if not entries:
                    logger.warning(f"No entries found for query '{query}'")
                    return []
            except (DownloadError, ExtractorError) as e:
                kind = self._classify_ytdlp_error(e)
                if kind == "rate_limit":
                    backoff = 5
                    logger.error(f"Rate limited for query '{query}', backoff {backoff}s: {e}")
                    time.sleep(backoff)
                    raise SearchRateLimitError(str(e)) from e
                if kind == "network":
                    logger.error(f"Network failure for query '{query}': {e}")
                    raise SearchNetworkError(str(e)) from e
                logger.warning(f"No results or unsupported query '{query}': {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected fast search failure for query '{query}': {e}", exc_info=True)
                raise
            except BaseException as e:
                if isinstance(e, SystemExit):
                    logger.warning(
                        f"yt-dlp exited during fast search for query '{query}' "
                        f"(SystemExit code: {e.code!r})"
                    )
                    return []
                raise
        logger.info(f"Search for {query} on {self.platform} receive {len(entries)} video infos")

        # 快速扁平搜索结束后先去重：
        # 1) 去掉本次搜索结果内重复
        # 2) 去掉数据库已存在的 (video_id, platform)
        unique_entries = []
        seen_video_ids = set()
        for entry in entries[:max_results]:
            video_id = entry.get('id')
            if not video_id or video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)
            unique_entries.append(entry)

        if self.video_store and unique_entries:
            existing_ids = self.video_store.get_existing_video_ids(
                platform=self.platform,
                video_ids=[entry.get('id') for entry in unique_entries]
            )
            if existing_ids:
                unique_entries = [
                    entry for entry in unique_entries
                    if entry.get('id') not in existing_ids
                ]

        logger.info(
            f"After dedup for '{query}' on {self.platform}, "
            f"{len(unique_entries)} entries remain"
        )

        # 阶段2：对每个条目获取详细信息
        results = []
        entry_with_urls = []
        for entry in unique_entries:
            video_url = entry.get('webpage_url') or entry.get('url')
            if not video_url:
                logger.warning(f"Skipping entry without URL: {entry}")
                continue
            entry_with_urls.append((entry, video_url))

        details_by_url = {}
        if entry_with_urls:
            with ThreadPoolExecutor(max_workers=self.search_config.get('detail_workers', 5)) as executor:
                future_to_url = {
                    executor.submit(self._get_video_details, video_url): video_url
                    for _, video_url in entry_with_urls
                }
                for future in as_completed(future_to_url):
                    video_url = future_to_url[future]
                    try:
                        details_by_url[video_url] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to get details for {video_url}: {e}")
                        details_by_url[video_url] = None
                    except BaseException as e:
                        if isinstance(e, SystemExit):
                            logger.warning(
                                f"yt-dlp exited while fetching details for {video_url} "
                                f"(SystemExit code: {e.code!r})"
                            )
                            details_by_url[video_url] = None
                            continue
                        raise

        for entry, video_url in entry_with_urls:
            # 获取详细信息
            details = details_by_url.get(video_url)
            if not details:
                logger.warning(f"Failed to get details for {video_url}, skipping")
                continue

            # 提取分辨率
            resolution = self._extract_resolution_from_details(details)

            # 构建 VideoMeta（优先使用详情中的字段，回退到扁平条目）
            video = VideoMeta(
                video_id=entry.get('id', ''),
                url=video_url,
                platform=self.platform,
                title=details.get('title', entry.get('title', '')),
                duration_seconds=details.get('duration', entry.get('duration', 0)) or 0,
                resolution=resolution,
                channel=details.get('uploader', entry.get('uploader', '')),
                publish_date=self._format_date(details.get('upload_date', entry.get('upload_date', ''))),
                view_count=details.get('view_count', entry.get('view_count', 0)) or 0,
                tags=details.get('tags', entry.get('tags', [])),
                search_term=query,
            )
            results.append(video)

        logger.info(f"Search for '{query}' returned {len(results)} videos with resolution info")
        return results

    def _get_video_details(self, url: str) -> Optional[dict]:
        """
        使用 extract_flat=False 获取单个视频的详细信息。
        返回信息字典，失败返回 None。
        """
        with yt_dlp.YoutubeDL(self.ydl_opts_detail) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return info
            except (DownloadError, ExtractorError) as e:
                kind = self._classify_ytdlp_error(e)
                if kind == "network":
                    logger.error(f"Network error while fetching details for {url}: {e}")
                elif kind == "rate_limit":
                    logger.error(f"Rate limit while fetching details for {url}: {e}")
                else:
                    logger.warning(f"No detail result for {url}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error getting details for {url}: {e}", exc_info=True)
                return None
            except BaseException as e:
                if isinstance(e, SystemExit):
                    logger.warning(
                        f"yt-dlp exited while fetching details for {url} "
                        f"(SystemExit code: {e.code!r})"
                    )
                    return None
                raise

    @staticmethod
    def _classify_ytdlp_error(exc: Exception) -> str:
        msg = str(exc).lower()
        if any(k in msg for k in ("429", "too many requests", "rate limit", "ratelimit")):
            return "rate_limit"
        if any(k in msg for k in (
                "timed out", "timeout", "connection", "network", "name resolution", "temporary failure",
                "connection reset", "proxy error", "unreachable"
        )):
            return "network"
        if any(k in msg for k in ("no results", "did not match any", "not found", "video unavailable")):
            return "no_results"
        return "other"

    @staticmethod
    def _extract_resolution_from_details(details: dict) -> str:
        """
        从视频详细信息中提取分辨率字符串。
        综合三种方法（formats、height字段、resolution字段）取最高分辨率。
        """
        max_height = 0

        # 方法1：从 formats 中找最佳分辨率（遍历所有格式取最大高度）
        formats = details.get('formats', [])
        for f in formats:
            h = f.get('height')
            if isinstance(h, int) and h > max_height:
                max_height = h

        # 方法2：直接获取 height 字段
        height = details.get('height')
        if isinstance(height, int) and height > max_height:
            max_height = height

        # 方法3：使用 resolution 字段（如 "1920x1080"）
        resolution = details.get('resolution')
        if resolution and isinstance(resolution, str) and 'x' in resolution:
            parts = resolution.split('x')
            if len(parts) == 2:
                h_str = parts[1]
                if h_str.isdigit():
                    h = int(h_str)
                    if h > max_height:
                        max_height = h

        # 返回结果
        if max_height > 0:
            return f"{max_height}p"
        return "unknown"

    @staticmethod
    def _format_date(date_str: str) -> str:
        """将 YYYYMMDD 格式转为 YYYY-MM-DD，若无效返回空字符串"""
        if not date_str or len(date_str) != 8:
            return ""
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
