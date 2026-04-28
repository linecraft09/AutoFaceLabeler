import time
import threading
from pathlib import Path

import aflutils.video_utils as VU
from aflutils.config_loader import ConfigLoader
from aflutils.logger import get_logger
from core.storage.video_store import VideoStore
from downloaders.base_downloader import BaseDownloader
from explorers.adaptive_scheduler import AdaptiveScheduler
from searchers.ytdlp_search_api import YtDlpSearchApi
from validators.pre_filter import PreFilter
from validators.validator import V2ContentFilter

logger = get_logger(__name__)
claim_lock = threading.Lock()


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_markers = (
        "timeout",
        "timed out",
        "temporar",
        "connection",
        "network",
        "reset by peer",
        "unavailable",
        "429",
        "too many requests",
    )
    return any(marker in msg for marker in transient_markers)


def run_pipeline(config=None):
    if config is None:
        config = ConfigLoader("./config/config.yaml")
    explorer = AdaptiveScheduler(config.get('explorer', {}))
    v1 = PreFilter(config.get('validator1', {}))
    project_root = Path(__file__).resolve().parent.parent.parent
    downloader = BaseDownloader(config_path=str(project_root / "config/download_conf.yaml"))

    # 初始化视频存储
    video_store = VideoStore(db_path=str(project_root / "data/videos.db"))
    yt_searcher = YtDlpSearchApi(platform='youtube', video_store=video_store)
    bl_searcher = YtDlpSearchApi(platform='bilibili', video_store=video_store)

    # 初始化并启动 V2 后台线程
    v2_config = config.get('validator2', {})
    v2_filter = V2ContentFilter(v2_config, video_store, explorer)
    v2_filter.start()
    loop_retry_attempt = 0

    try:
        while True:
            try:
                stats = video_store.get_statistics()
                logger.info(f"Storage stats: {stats}")
                while sum(stats.values()) > 0 and stats.get("downloaded", 0) / sum(stats.values()) >= 0.5:
                    logger.warning(f"downloaded videos so many, wait to be consumed.")
                    time.sleep(60)
                    stats = video_store.get_statistics()
                batch = explorer.generate_batch(batch_size=10)
                for term in batch:
                    max_retries = 3
                    for attempt in range(max_retries + 1):
                        try:
                            # 搜索
                            videos = yt_searcher.search(term.text, max_results=30)
                            videos.extend(bl_searcher.search(term.text, max_results=30))

                            # V1 预筛选
                            passed, feedback_v1 = v1.filter(videos, term.text)
                            explorer.receive_feedback(feedback_v1)

                            # 下载
                            downloaded = {}
                            try:
                                downloaded = downloader.download([video.url for video in passed])
                            except Exception as download_exc:
                                logger.error(
                                    f"Download failed for term '{term.text}': {download_exc}",
                                    exc_info=True
                                )

                            # 存储每个视频（无论是否下载成功，都保存元数据）
                            for video in passed:
                                file_path = downloaded.get(video.url)  # 可能为 None 如果下载失败
                                video_store.insert_or_update(video, file_path)

                            # 可选：打印当前存储统计
                            stats = video_store.get_statistics()
                            logger.info(f"Storage stats: {stats}")
                            break
                        except Exception as e:
                            is_transient = _is_transient_error(e)
                            if is_transient and attempt < max_retries:
                                backoff = 2 ** attempt
                                logger.warning(
                                    f"Transient failure for term '{term.text}' (attempt {attempt + 1}/{max_retries + 1}), "
                                    f"retrying in {backoff}s: {e}"
                                )
                                time.sleep(backoff)
                                continue
                            logger.error(
                                f"Failed to process term '{term.text}', skipping: {e}",
                                exc_info=True
                            )
                            break
                if stats.get("v2_passed", 0) >= config.get("orchestrator", {}).get("target_qualified", 200):
                    break

                # 每轮后自适应
                explorer.adapt_strategy()
                logger.info(f"explorer status:{explorer.get_status()}")

                # 每轮结束后清理已无保留必要状态的视频文件（仅删文件，不删数据库记录）
                with claim_lock:
                    stale_video_paths = video_store.claim_file_paths_by_excluded_statuses(
                        excluded_statuses=("downloaded", "v2_passed")
                    )
                if stale_video_paths:
                    logger.info(f"Cleaning stale videos: {len(stale_video_paths)} files")
                    VU.remove_videos(stale_video_paths)

                loop_retry_attempt = 0
            except Exception as e:
                is_transient = _is_transient_error(e)
                if is_transient:
                    backoff = min(60, 2 ** loop_retry_attempt)
                    logger.warning(
                        f"Transient pipeline loop failure, retrying in {backoff}s: {e}",
                        exc_info=True
                    )
                    time.sleep(backoff)
                    loop_retry_attempt += 1
                    continue

                logger.error(
                    f"Non-transient pipeline loop failure, continuing next cycle: {e}",
                    exc_info=True
                )
                time.sleep(1)
                loop_retry_attempt = 0
                continue
    finally:
        v2_filter.stop()
        explorer.save_state()

    # 循环结束后，可以启动过滤（或者单独运行）
    logger.info("Collection finished. Pending videos ready for Labeler processing.")
