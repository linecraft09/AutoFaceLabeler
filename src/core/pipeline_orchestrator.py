import os
import time
import threading
import uuid

from aflutils.logger import get_logger
from core.storage.video_store import VideoStore

logger = get_logger(__name__)
claim_lock = threading.Lock()
PIPELINE_STAGES = ("search", "download", "v1", "v2")
NEXT_PIPELINE_STAGE = {
    "search": "download",
    "download": "v1",
    "v1": "v2",
}

VU = None
DefaultDownloader = None
AdaptiveScheduler = None
YtDlpSearchApi = None
PreFilter = None
V2ContentFilter = None


def _load_config_loader():
    from aflutils.config_loader import ConfigLoader
    return ConfigLoader


def _load_pipeline_components():
    """Lazy-load heavy pipeline dependencies so restart tests can monkeypatch them."""
    global DefaultDownloader
    global AdaptiveScheduler
    global YtDlpSearchApi
    global PreFilter
    global V2ContentFilter

    if DefaultDownloader is None:
        from downloaders.base_downloader import DefaultDownloader as downloader_cls
        DefaultDownloader = downloader_cls
    if AdaptiveScheduler is None:
        from explorers.adaptive_scheduler import AdaptiveScheduler as scheduler_cls
        AdaptiveScheduler = scheduler_cls
    if YtDlpSearchApi is None:
        from searchers.ytdlp_search_api import YtDlpSearchApi as searcher_cls
        YtDlpSearchApi = searcher_cls
    if PreFilter is None:
        from validators.pre_filter import PreFilter as pre_filter_cls
        PreFilter = pre_filter_cls
    if V2ContentFilter is None:
        from validators.validator import V2ContentFilter as v2_filter_cls
        V2ContentFilter = v2_filter_cls


def _remove_videos(paths):
    global VU
    if VU is None:
        import aflutils.video_utils as video_utils
        VU = video_utils
    VU.remove_videos(paths)


def _normalize_resume_stage(stage):
    if stage in PIPELINE_STAGES:
        return stage
    logger.warning(f"Unknown pipeline resume stage '{stage}', restarting from search")
    return "search"


def _stage_completed(resume_stage, stage_name):
    if resume_stage is None:
        return False
    if resume_stage not in PIPELINE_STAGES or stage_name not in PIPELINE_STAGES:
        return False
    return PIPELINE_STAGES.index(stage_name) < PIPELINE_STAGES.index(resume_stage)


def _skip_stage(resume_stage, stage_name):
    """Return True when a stage is before the checkpoint's next stage to run."""
    return _stage_completed(resume_stage, stage_name)


def _advance_pipeline_stage(video_store, run_id, stage_name):
    next_stage = NEXT_PIPELINE_STAGE.get(stage_name)
    if next_stage:
        video_store.update_pipeline_run_stage(run_id, next_stage)


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


def run_housekeeping(video_store: VideoStore, config: dict = None):
    """Run startup cleanup tasks for restart-safe pipeline state."""
    config = config or {}
    housekeeping_cfg = config.get("housekeeping", {})
    if housekeeping_cfg.get("enabled", True) is False:
        logger.info("Housekeeping disabled")
        return {
            "enabled": False,
            "stale_leases_reclaimed": 0,
            "orphaned_checkpoints": {"db_rows_deleted": 0, "dirs_removed": 0},
            "pipeline_runs_pruned": 0,
        }

    results = {"enabled": True}

    if housekeeping_cfg.get("stale_lease_reap", True):
        reclaimed = video_store.reap_stale_leases()
        logger.info(f"Housekeeping stale lease reaper reclaimed {reclaimed} rows")
        results["stale_leases_reclaimed"] = reclaimed
    else:
        results["stale_leases_reclaimed"] = 0
        logger.info("Housekeeping stale lease reaper disabled")

    if housekeeping_cfg.get("orphaned_checkpoint_cleanup", True):
        checkpoint_dir = (
            config.get("v2_filter", {})
            .get("checkpoint", {})
            .get("dir", "./data/checkpoints")
        )
        cleanup_result = video_store.cleanup_orphaned_checkpoints(checkpoint_dir=checkpoint_dir)
        logger.info(f"Housekeeping orphaned checkpoint cleanup result: {cleanup_result}")
        results["orphaned_checkpoints"] = cleanup_result
    else:
        results["orphaned_checkpoints"] = {"db_rows_deleted": 0, "dirs_removed": 0}
        logger.info("Housekeeping orphaned checkpoint cleanup disabled")

    keep_last = housekeeping_cfg.get("keep_last_runs", 100)
    pruned = video_store.prune_old_pipeline_runs(keep_last=keep_last)
    logger.info(f"Housekeeping pruned {pruned} old pipeline runs")
    results["pipeline_runs_pruned"] = pruned

    return results


def run_pipeline(config=None):
    if config is None:
        ConfigLoader = _load_config_loader()
        config = ConfigLoader("./config/config.yaml")

    # 初始化视频存储
    video_store = VideoStore(db_path=config.get('project', {}).get('video_db', './data/videos.db'))
    run_housekeeping(video_store, config)
    existing_run = video_store.get_in_progress_run()
    if existing_run:
        run_id = existing_run["run_id"]
        resume_stage = _normalize_resume_stage(existing_run["stage"])
        logger.info(f"Found crashed pipeline run {run_id}, resuming at stage {resume_stage}")
    else:
        run_id = str(uuid.uuid4())
        video_store.create_pipeline_run(run_id, stage="search")
        resume_stage = "search"

    explorer = None
    v2_filter = None
    completed = False

    try:
        _load_pipeline_components()

        explorer_cfg = config.get('explorer', {})
        explorer_cfg['pipeline'] = config.get('pipeline', {})
        explorer_cfg['project'] = config.get('project', {})
        explorer = AdaptiveScheduler(explorer_cfg)
        v1 = PreFilter(config.get('v1_filter', {})) if not _skip_stage(resume_stage, "v1") else None
        downloader = (
            DefaultDownloader(config_dict=config.get('download', {}))
            if not _skip_stage(resume_stage, "download") else None
        )

        search_cfg = config.get('search', {})
        # 尝试从环境变量获取代理（.env 或系统环境变量）
        proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or None

        yt_searcher = None
        bl_searcher = None
        if not _skip_stage(resume_stage, "search"):
            yt_searcher = YtDlpSearchApi(
                platform='youtube', proxy=proxy, video_store=video_store,
                search_config=search_cfg, cookies=search_cfg.get('cookies')
            )
            bl_searcher = YtDlpSearchApi(
                platform='bilibili', proxy=proxy, video_store=video_store,
                search_config=search_cfg, cookies=search_cfg.get('cookies')
            )

        # 初始化 V2，实际启动延后到 v1 阶段之后，便于按阶段恢复。
        v2_config = dict(config.get('v2_filter', {}))
        v2_config['project'] = config.get('project', {})
        v2_filter = V2ContentFilter(v2_config, video_store, explorer)
        v2_started = False
        loop_retry_attempt = 0

        while True:
            try:
                stats = video_store.get_statistics()
                logger.info(f"Storage stats: {stats}")
                pipeline_cfg = config.get('pipeline', {})
                pending_ratio_threshold = pipeline_cfg.get('pending_ratio_threshold', 0.5)
                pending_wait_seconds = pipeline_cfg.get('pending_wait_seconds', 60)
                while sum(stats.values()) > 0 and stats.get("downloaded", 0) / sum(stats.values()) >= pending_ratio_threshold:
                    logger.warning(f"downloaded videos so many, wait to be consumed.")
                    time.sleep(pending_wait_seconds)
                    stats = video_store.get_statistics()
                batch = explorer.generate_batch(batch_size=pipeline_cfg.get('batch_size', 10))
                for term in batch:
                    max_retries = pipeline_cfg.get('max_retries', 3)
                    for attempt in range(max_retries + 1):
                        try:
                            videos = []
                            if _skip_stage(resume_stage, "search"):
                                logger.info(f"Skipping search stage for run {run_id}")
                            else:
                                max_results = search_cfg.get('per_platform_results', 20)
                                videos = yt_searcher.search(term.text, max_results=max_results)
                                videos.extend(bl_searcher.search(term.text, max_results=max_results))
                            _advance_pipeline_stage(video_store, run_id, "search")

                            downloaded = {}
                            if _skip_stage(resume_stage, "download"):
                                logger.info(f"Skipping download stage for run {run_id}")
                            else:
                                try:
                                    downloaded = downloader.download([video.url for video in videos])
                                except Exception as download_exc:
                                    logger.error(
                                        f"Download failed for term '{term.text}': {download_exc}",
                                        exc_info=True
                                    )
                            _advance_pipeline_stage(video_store, run_id, "download")

                            if _skip_stage(resume_stage, "v1"):
                                logger.info(f"Skipping v1 stage for run {run_id}")
                            else:
                                passed, feedback_v1 = v1.filter(videos, term.text)
                                explorer.receive_feedback(feedback_v1)

                                # 存储每个视频（无论是否下载成功，都保存元数据）
                                for video in passed:
                                    file_path = downloaded.get(video.url)  # 可能为 None 如果下载失败
                                    video_store.insert_or_update(video, file_path)
                            _advance_pipeline_stage(video_store, run_id, "v1")

                            if not v2_started and not _skip_stage(resume_stage, "v2"):
                                v2_filter.start()
                                v2_started = True

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
                if not v2_started and not _skip_stage(resume_stage, "v2"):
                    v2_filter.start()
                    v2_started = True

                if stats.get("v2_passed", 0) >= config.get("pipeline", {}).get("target_qualified", 200):
                    completed = True
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
                    _remove_videos(stale_video_paths)

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
    except Exception:
        video_store.fail_pipeline_run(run_id)
        raise
    finally:
        try:
            if v2_filter is not None:
                v2_filter.stop()
        finally:
            thread = getattr(v2_filter, "thread", None) if v2_filter is not None else None
            if thread and thread.is_alive():
                logger.error("V2 filter thread still alive after stop(); waiting up to 5s")
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.error("V2 filter thread still alive after timeout; leaving daemon thread to exit with process")
            if explorer is not None:
                explorer.save_state()
        if completed:
            video_store.complete_pipeline_run(run_id)

    # 循环结束后，可以启动过滤（或者单独运行）
    logger.info("Collection finished. Pending videos ready for Labeler processing.")
