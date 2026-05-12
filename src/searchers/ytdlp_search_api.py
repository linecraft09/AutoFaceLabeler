import random
import copy
import queue
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError

from aflutils.logger import get_logger
from aflutils.platform_cookies import resolve_platform_cookies
from core.models.video_meta import VideoMeta
from core.storage.video_store import VideoStore
from .search_api import SearchApi

logger = get_logger(__name__)


class SearchRateLimitError(Exception):
    pass


class SearchNetworkError(Exception):
    pass


class SearchCooldownTriggerError(Exception):
    def __init__(self, message: str, platform: str, stage: str, reason: str):
        super().__init__(message)
        self.platform = platform
        self.stage = stage
        self.reason = reason


class SearchTimeoutError(SearchCooldownTriggerError):
    pass


class SearchMaxRetriesExceededError(SearchCooldownTriggerError):
    pass


def _stage_config(search_config: dict, platform: str, stage: str) -> dict:
    global_cfg = search_config.get(stage, {})
    if not isinstance(global_cfg, dict):
        global_cfg = {}
    platform_cfg = search_config.get('platforms', {}).get(platform, {})
    if not isinstance(platform_cfg, dict):
        platform_cfg = {}
    platform_stage_cfg = platform_cfg.get(stage, {})
    if not isinstance(platform_stage_cfg, dict):
        platform_stage_cfg = {}
    merged = copy.deepcopy(global_cfg)
    merged.update(copy.deepcopy(platform_stage_cfg))
    return merged


def _retry_policy(config: dict, default_attempts: int = 1) -> dict:
    backoff_cfg = config.get('backoff', {}) if isinstance(config.get('backoff'), dict) else {}
    return {
        'max_attempts': max(1, int(config.get('max_attempts', config.get('retries', default_attempts)))),
        'initial_seconds': float(config.get('backoff_initial_seconds', backoff_cfg.get('initial_seconds', 1))),
        'max_seconds': float(config.get('backoff_max_seconds', backoff_cfg.get('max_seconds', 30))),
        'multiplier': max(1.0, float(config.get('backoff_multiplier', backoff_cfg.get('multiplier', 2)))),
        'jitter_seconds': max(0.0, float(config.get('backoff_jitter_seconds', backoff_cfg.get('jitter_seconds', 0)))),
    }


def _backoff_delay(policy: dict, attempt_index: int) -> float:
    delay = policy['initial_seconds'] * (policy['multiplier'] ** attempt_index)
    delay = min(policy['max_seconds'], delay)
    if policy['jitter_seconds']:
        delay += random.uniform(0, policy['jitter_seconds'])
    return delay


def _timeout_seconds(config: dict, default_seconds: float) -> float:
    value = config.get(
        'timeout_seconds',
        config.get('operation_timeout_seconds', config.get('wall_timeout_seconds', default_seconds)),
    )
    return max(0.0, float(value))


def _extract_info_direct(ydl_options: dict, target: str, download: bool) -> Optional[dict]:
    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        return ydl.extract_info(target, download=download)


def _log_anti_bot_wait(
        stage: str,
        target: str,
        platform: str,
        attempt: int,
        max_attempts: int,
        use_cookies: bool,
        wait_seconds: float,
        action: str,
        error: Exception,
) -> None:
    logger.warning(
        "Anti-bot detected during %s for %s on %s "
        "(attempt %s/%s, cookies=%s); anti_bot_wait_seconds=%.1f; %s: %s",
        stage,
        target,
        platform,
        attempt,
        max_attempts,
        use_cookies,
        wait_seconds,
        action,
        error,
    )


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
        self._cookies = cookies or resolve_platform_cookies(self.search_config, self.platform)
        self.cookie_fallback_on_anti_bot = bool(self.search_config.get('cookie_fallback_on_anti_bot', False))
        if self.platform not in ['youtube', 'bilibili']:
            raise ValueError("platform must be 'youtube' or 'bilibili'")

        fast_cfg = _stage_config(self.search_config, self.platform, 'fast_search')
        detail_cfg = _stage_config(self.search_config, self.platform, 'detail_fetch')
        self.fast_retry_policy = _retry_policy(fast_cfg, default_attempts=1)
        self.detail_retry_policy = _retry_policy(detail_cfg, default_attempts=1)
        self.fast_timeout_seconds = _timeout_seconds(
            fast_cfg,
            float(fast_cfg.get('socket_timeout', 30)) + 15,
        )
        self.detail_timeout_seconds = _timeout_seconds(
            detail_cfg,
            float(detail_cfg.get('socket_timeout', 60)) + 30,
        )
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
        # Keep the resolved cookie path on the instance for callers that need it,
        # but do not pass cookies to yt-dlp during search/detail metadata fetches.

        # 详细配置（用于获取单个视频详情，extract_flat=False）
        self.ydl_opts_detail = copy.deepcopy(self.ydl_opts_fast)
        self.ydl_opts_detail['extract_flat'] = False
        self.ydl_opts_detail['remote_components'] = ['ejs:github']
        # 详细模式可能需要更长的超时
        self.ydl_opts_detail['socket_timeout'] = detail_cfg.get('socket_timeout', 60)

    def _extract_info_with_timeout(
            self,
            ydl_options: dict,
            target: str,
            stage: str,
            timeout_seconds: float,
            download: bool = False,
    ) -> Optional[dict]:
        if timeout_seconds <= 0:
            return _extract_info_direct(ydl_options, target, download)

        result_queue = queue.Queue(maxsize=1)

        def run_extract() -> None:
            try:
                result_queue.put_nowait((True, _extract_info_direct(ydl_options, target, download)))
            except BaseException as exc:
                try:
                    result_queue.put_nowait((False, exc))
                except queue.Full:
                    pass

        thread = threading.Thread(
            target=run_extract,
            name=f"yt-dlp-{self.platform}-{stage}",
            daemon=True,
        )
        thread.start()
        try:
            ok, payload = result_queue.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            raise SearchTimeoutError(
                f"{stage} timed out after {timeout_seconds:.1f}s for {target}",
                platform=self.platform,
                stage=stage,
                reason="timeout",
            ) from exc
        if ok:
            return payload
        raise payload

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
        entries = self._extract_fast_search(search_query, query)
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
            detail_workers = max(1, int(self.search_config.get('detail_workers', 5)))
            executor = ThreadPoolExecutor(max_workers=detail_workers)
            cooldown_error = None
            future_to_url = {}
            try:
                future_to_url = {
                    executor.submit(self._get_video_details, video_url): video_url
                    for _, video_url in entry_with_urls
                }
                for future in as_completed(future_to_url):
                    video_url = future_to_url[future]
                    try:
                        details_by_url[video_url] = future.result()
                    except SearchCooldownTriggerError as e:
                        logger.error(f"Platform cooldown triggered while fetching details for {video_url}: {e}")
                        details_by_url[video_url] = None
                        cooldown_error = e
                        break
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
            finally:
                if cooldown_error:
                    for future in future_to_url:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=True)
            if cooldown_error:
                raise cooldown_error

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

    def _options_for_search_attempt(self, base_options: dict, use_cookies: bool) -> dict:
        options = base_options.copy()
        if use_cookies and self._cookies:
            options['cookiefile'] = self._cookies
        return options

    def _can_fallback_to_cookies(self, use_cookies: bool) -> bool:
        return self.cookie_fallback_on_anti_bot and self._cookies and not use_cookies

    def _extract_fast_search(self, search_query: str, query: str) -> List[dict]:
        policy = self.fast_retry_policy
        max_attempts = policy['max_attempts']
        use_cookies = False
        while True:
            restart_with_cookies = False
            for attempt in range(max_attempts):
                ydl_options = self._options_for_search_attempt(self.ydl_opts_fast, use_cookies)
                try:
                    info = self._extract_info_with_timeout(
                        ydl_options,
                        search_query,
                        stage="fast_search",
                        timeout_seconds=self.fast_timeout_seconds,
                    )
                    entries = info.get('entries', []) if info else []
                    if not entries:
                        logger.warning(f"No entries found for query '{query}'")
                        return []
                    return entries
                except SearchTimeoutError as e:
                    logger.error(
                        "Fast search timed out for query '%s' on %s after %.1fs",
                        query,
                        self.platform,
                        self.fast_timeout_seconds,
                    )
                    raise
                except (DownloadError, ExtractorError) as e:
                    kind = self._classify_ytdlp_error(e)
                    if kind == "anti_bot" and self._can_fallback_to_cookies(use_cookies):
                        _log_anti_bot_wait(
                            stage="fast search",
                            target=f"query '{query}'",
                            platform=self.platform,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            use_cookies=use_cookies,
                            wait_seconds=0.0,
                            action="retrying with cookie fallback",
                            error=e,
                        )
                        use_cookies = True
                        restart_with_cookies = True
                        break
                    if kind in {"rate_limit", "network", "anti_bot"} and attempt < max_attempts - 1:
                        delay = _backoff_delay(policy, attempt)
                        if kind == "anti_bot":
                            _log_anti_bot_wait(
                                stage="fast search",
                                target=f"query '{query}'",
                                platform=self.platform,
                                attempt=attempt + 1,
                                max_attempts=max_attempts,
                                use_cookies=use_cookies,
                                wait_seconds=delay,
                                action="backing off before retry",
                                error=e,
                            )
                        else:
                            logger.warning(
                                "Retryable fast search failure for query '%s' on %s "
                                "(kind=%s, attempt %s/%s, cookies=%s), backing off %.1fs: %s",
                                query,
                                self.platform,
                                kind,
                                attempt + 1,
                                max_attempts,
                                use_cookies,
                                delay,
                                e,
                            )
                        time.sleep(delay)
                        continue
                    if kind in {"rate_limit", "network", "anti_bot"}:
                        if kind == "anti_bot":
                            logger.error(
                                "Anti-bot blocked fast search for query '%s' on %s; "
                                "anti_bot_wait_seconds=0.0; no retry attempt remains: %s",
                                query,
                                self.platform,
                                e,
                            )
                        else:
                            logger.error(
                                "Retryable fast search failure exhausted for query '%s' on %s "
                                "(kind=%s): %s",
                                query,
                                self.platform,
                                kind,
                                e,
                            )
                        raise SearchMaxRetriesExceededError(
                            f"fast_search exhausted {max_attempts} attempt(s) for {query}: {e}",
                            platform=self.platform,
                            stage="fast_search",
                            reason=kind,
                        ) from e
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
            if restart_with_cookies:
                continue
            return []

    def _get_video_details(self, url: str) -> Optional[dict]:
        """
        使用 extract_flat=False 获取单个视频的详细信息。
        返回信息字典，失败返回 None。
        """
        policy = self.detail_retry_policy
        max_attempts = policy['max_attempts']
        use_cookies = False
        while True:
            restart_with_cookies = False
            for attempt in range(max_attempts):
                ydl_options = self._options_for_search_attempt(self.ydl_opts_detail, use_cookies)
                try:
                    info = self._extract_info_with_timeout(
                        ydl_options,
                        url,
                        stage="detail_fetch",
                        timeout_seconds=self.detail_timeout_seconds,
                    )
                    return info
                except SearchTimeoutError:
                    logger.error(
                        "Detail fetch timed out for %s on %s after %.1fs",
                        url,
                        self.platform,
                        self.detail_timeout_seconds,
                    )
                    raise
                except (DownloadError, ExtractorError) as e:
                    kind = self._classify_ytdlp_error(e)
                    if kind == "anti_bot" and self._can_fallback_to_cookies(use_cookies):
                        _log_anti_bot_wait(
                            stage="detail fetch",
                            target=url,
                            platform=self.platform,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            use_cookies=use_cookies,
                            wait_seconds=0.0,
                            action="retrying with cookie fallback",
                            error=e,
                        )
                        use_cookies = True
                        restart_with_cookies = True
                        break
                    if kind in {"rate_limit", "network", "anti_bot"} and attempt < max_attempts - 1:
                        delay = _backoff_delay(policy, attempt)
                        if kind == "anti_bot":
                            _log_anti_bot_wait(
                                stage="detail fetch",
                                target=url,
                                platform=self.platform,
                                attempt=attempt + 1,
                                max_attempts=max_attempts,
                                use_cookies=use_cookies,
                                wait_seconds=delay,
                                action="backing off before retry",
                                error=e,
                            )
                        else:
                            logger.warning(
                                "Retryable detail fetch failure for %s "
                                "(kind=%s, attempt %s/%s, cookies=%s), backing off %.1fs: %s",
                                url,
                                kind,
                                attempt + 1,
                                max_attempts,
                                use_cookies,
                                delay,
                                e,
                            )
                        time.sleep(delay)
                        continue
                    if kind in {"rate_limit", "network", "anti_bot"}:
                        if kind == "anti_bot":
                            logger.error(
                                "Anti-bot blocked detail fetch for %s on %s; "
                                "anti_bot_wait_seconds=0.0; no retry attempt remains: %s",
                                url,
                                self.platform,
                                e,
                            )
                        else:
                            logger.error(
                                "Retryable detail fetch failure exhausted for %s on %s "
                                "(kind=%s): %s",
                                url,
                                self.platform,
                                kind,
                                e,
                            )
                        raise SearchMaxRetriesExceededError(
                            f"detail_fetch exhausted {max_attempts} attempt(s) for {url}: {e}",
                            platform=self.platform,
                            stage="detail_fetch",
                            reason=kind,
                        ) from e
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
            if restart_with_cookies:
                continue
            return None

    @staticmethod
    def _classify_ytdlp_error(exc: Exception) -> str:
        msg = str(exc).lower()
        if any(k in msg for k in (
                "sign in to confirm", "not a bot", "precondition failed", "http error 412",
                "captcha", "verify you are human"
        )):
            return "anti_bot"
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
