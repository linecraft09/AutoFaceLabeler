import argparse
import json
import os
import random
import signal
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

from aflutils.config_loader import ConfigLoader
from core import pipeline_orchestrator as orchestrator

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


MAX_TEST_RUNTIME_SEC = _env_int('AFL_MAX_TEST_RUNTIME_SEC', 1800)
MONITOR_INTERVAL_SEC = _env_int('AFL_MONITOR_INTERVAL_SEC', 15)
CPU_WARN_PCT = 90
GPU_MEM_WARN_PCT = 90
CPU_SUSTAINED_SEC = 60
BREAKPOINT_RESTARTS_ENABLED = _env_bool('AFL_BREAKPOINT_RESTARTS_ENABLED', True)
MAX_KILL_RESUME_CYCLES = _env_int('AFL_MAX_KILL_RESUME_CYCLES', 3)
KILL_AFTER_RANGE_SEC = (60, 180)
RESUME_WAIT_RANGE_SEC = (5, 10)
TARGET_QUALIFIED = _env_int('AFL_TARGET_QUALIFIED', 1)
MAX_TOTAL_PIPELINE_LOOPS = _env_int('AFL_MAX_TOTAL_PIPELINE_LOOPS', 10)
EXIT_MAX_PIPELINE_LOOPS = 75
LOG_PATH = Path('/tmp/afl_integration_test.jsonl')
CONFIG_PATH = str(Path(__file__).resolve().parent / 'config_test.yaml')

_STAGE = {'name': 'init'}
_LOOPS = {'count': 0}
_PASSES = {'v2_coarse': 0, 'v2_fine': 0}


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, 'a', encoding='utf-8')

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        payload = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'event': event_type,
            **data,
        }
        line = json.dumps(payload, ensure_ascii=True)
        with self._lock:
            self._fh.write(line + '\n')
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()


def _read_gpu_stats() -> Dict[str, Any]:
    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        for candidate in (
            '/usr/bin/nvidia-smi',
            '/usr/local/cuda/bin/nvidia-smi',
            '/bin/nvidia-smi',
        ):
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                nvidia_smi = candidate
                break

    if not nvidia_smi:
        return {
            'gpu_mem_used_mb': None,
            'gpu_mem_total_mb': None,
            'gpu_mem_pct': None,
            'gpu_util_pct': None,
            'gpu_smi_path': None,
            'gpu_error': 'nvidia-smi not found in PATH or common locations',
        }

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.strip().splitlines()
        if not lines:
            return {
                'gpu_mem_used_mb': None,
                'gpu_mem_total_mb': None,
                'gpu_mem_pct': None,
                'gpu_util_pct': None,
                'gpu_smi_path': nvidia_smi,
                'gpu_error': 'nvidia-smi returned no output',
            }
        line = lines[0]
        mem_used, mem_total, util = [x.strip() for x in line.split(',')]
        mem_used_f = float(mem_used)
        mem_total_f = float(mem_total)
        util_f = float(util)
        mem_pct = (mem_used_f / mem_total_f * 100.0) if mem_total_f > 0 else 0.0
        return {
            'gpu_mem_used_mb': mem_used_f,
            'gpu_mem_total_mb': mem_total_f,
            'gpu_mem_pct': mem_pct,
            'gpu_util_pct': util_f,
            'gpu_smi_path': nvidia_smi,
            'gpu_error': None,
        }
    except subprocess.CalledProcessError as e:
        return {
            'gpu_mem_used_mb': None,
            'gpu_mem_total_mb': None,
            'gpu_mem_pct': None,
            'gpu_util_pct': None,
            'gpu_smi_path': nvidia_smi,
            'gpu_error': f'nvidia-smi failed with exit {e.returncode}: {e.stderr.strip() or e.stdout.strip()}',
        }
    except Exception as e:
        return {
            'gpu_mem_used_mb': None,
            'gpu_mem_total_mb': None,
            'gpu_mem_pct': None,
            'gpu_util_pct': None,
            'gpu_smi_path': nvidia_smi,
            'gpu_error': f'nvidia-smi error: {e}',
        }


def _safe_stop_pipeline_thread(thread: threading.Thread, logger: JsonlLogger) -> None:
    if not thread.is_alive():
        return
    tid = thread.ident
    if tid is None:
        return

    logger.log('pipeline_stop_attempt', {'method': 'thread_async_exc', 'signal': 'SIGTERM-equivalent'})
    try:
        import ctypes

        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid), ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), 0)
    except Exception as e:
        logger.log('pipeline_stop_error', {'error': str(e)})

    thread.join(timeout=10)
    if thread.is_alive():
        logger.log('pipeline_stop_attempt', {'method': 'thread_async_exc', 'signal': 'SIGKILL-equivalent'})
        try:
            import ctypes

            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(tid), ctypes.py_object(KeyboardInterrupt)
            )
        except Exception as e:
            logger.log('pipeline_stop_error', {'error': str(e)})


def _config_value(*keys, default=None):
    node = ConfigLoader(CONFIG_PATH).as_dict()
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node


def _db_path() -> str:
    return str(_config_value('project', 'video_db', default='/tmp/afl_videos_test.db'))


def _checkpoint_dir() -> str:
    return str(_config_value('v2_filter', 'checkpoint', 'dir', default='/tmp/afl_checkpoints_test'))


def _qualified_count() -> int:
    db_path = _db_path()
    if not os.path.exists(db_path):
        return 0
    status = 'v2_passed' if _config_value('pipeline', 'enable_v2', default=True) else 'downloaded'
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM videos WHERE status = ?",
                (status,),
            ).fetchone()
            return int(row[0] if row else 0)
    except sqlite3.Error:
        return 0


def _pipeline_runs() -> List[Dict[str, Any]]:
    db_path = _db_path()
    if not os.path.exists(db_path):
        return []
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT *
                FROM pipeline_run
                ORDER BY created_at, updated_at, run_id
            """).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def _loop_count_from_log() -> int:
    if not LOG_PATH.exists():
        return 0
    count = 0
    try:
        with LOG_PATH.open('r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    if json.loads(line).get('event') == 'loop_complete':
                        count += 1
                except json.JSONDecodeError:
                    continue
    except OSError:
        return count
    return count


def _process_snapshot() -> str:
    try:
        result = subprocess.run(
            "ps aux | grep python | grep -v grep",
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"process_snapshot_error: {e}"


def _kill_process_group(proc: subprocess.Popen, logger: JsonlLogger, reason: str) -> None:
    if proc.poll() is not None:
        return
    logger.log('pipeline_kill', {'pid': proc.pid, 'signal': 'SIGKILL', 'reason': reason})
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception as e:
        logger.log('pipeline_kill_error', {'pid': proc.pid, 'error': str(e)})


def _terminate_process_group(proc: subprocess.Popen, logger: JsonlLogger, reason: str) -> None:
    if proc.poll() is not None:
        return
    logger.log('pipeline_stop_attempt', {'pid': proc.pid, 'method': 'process_group', 'signal': 'SIGTERM', 'reason': reason})
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(proc, logger, f'{reason}_timeout')
    except ProcessLookupError:
        return
    except Exception as e:
        logger.log('pipeline_stop_error', {'pid': proc.pid, 'error': str(e)})


def _start_pipeline_worker(
    logger: JsonlLogger,
    restart_index: int,
    target_qualified: int,
    max_total_pipeline_loops: int,
) -> subprocess.Popen:
    env = os.environ.copy()
    env['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    env.setdefault('HTTP_PROXY', 'http://127.0.0.1:7890')
    env['PYTHONPATH'] = SRC_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['AFL_INITIAL_LOOP_COUNT'] = str(_loop_count_from_log())
    env['AFL_TARGET_QUALIFIED'] = str(target_qualified)
    env['AFL_MAX_TOTAL_PIPELINE_LOOPS'] = str(max_total_pipeline_loops)
    proc = subprocess.Popen(
        [sys.executable, __file__, '--pipeline-worker'],
        cwd=PROJECT_ROOT,
        env=env,
        start_new_session=True,
    )
    logger.log('pipeline_process_start', {
        'pid': proc.pid,
        'restart_index': restart_index,
        'in_progress_runs_before_start': _pipeline_runs(),
    })
    return proc


def _install_stage_instrumentation(logger: JsonlLogger) -> None:
    from explorers.adaptive_scheduler import AdaptiveScheduler
    from searchers.ytdlp_search_api import YtDlpSearchApi
    from validators.pre_filter import PreFilter
    from downloaders.base_downloader import DefaultDownloader
    from validators.validator import V2ContentFilter
    from core.storage.video_store import VideoStore

    orig_search = YtDlpSearchApi.search
    orig_filter = PreFilter.filter
    orig_download = DefaultDownloader.download
    orig_coarse = V2ContentFilter._coarse_filter
    orig_fine = V2ContentFilter._fine_filter
    orig_adapt = AdaptiveScheduler.adapt_strategy
    orig_get_in_progress_run = VideoStore.get_in_progress_run
    orig_create_pipeline_run = VideoStore.create_pipeline_run
    orig_update_pipeline_run_stage = VideoStore.update_pipeline_run_stage
    orig_complete_pipeline_run = VideoStore.complete_pipeline_run
    orig_save_checkpoint = VideoStore.save_checkpoint
    orig_load_checkpoint = VideoStore.load_checkpoint
    initial_loop_count = int(os.environ.get('AFL_INITIAL_LOOP_COUNT', '0') or '0')
    last_stage_log = {'key': None}
    seen_pipeline_run_stages = set()

    def log_stage_event(event_type: str, data: Dict[str, Any]) -> None:
        key = (event_type, data.get('stage'))
        if key == last_stage_log['key']:
            return
        last_stage_log['key'] = key
        logger.log(event_type, data)

    def remove_search_cookiefiles(searcher) -> Dict[str, Optional[str]]:
        original = {
            'fast_cookiefile': getattr(searcher, 'ydl_opts_fast', {}).get('cookiefile'),
            'detail_cookiefile': getattr(searcher, 'ydl_opts_detail', {}).get('cookiefile'),
        }
        if hasattr(searcher, 'ydl_opts_fast'):
            searcher.ydl_opts_fast.pop('cookiefile', None)
        if hasattr(searcher, 'ydl_opts_detail'):
            searcher.ydl_opts_detail.pop('cookiefile', None)
        return original

    def restore_search_cookiefiles(searcher, original: Dict[str, Optional[str]]) -> None:
        for opts_name, original_key in (
            ('ydl_opts_fast', 'fast_cookiefile'),
            ('ydl_opts_detail', 'detail_cookiefile'),
        ):
            opts = getattr(searcher, opts_name, None)
            if opts is None:
                continue
            if original[original_key] is None:
                opts.pop('cookiefile', None)
            else:
                opts['cookiefile'] = original[original_key]

    def patched_search(self, query: str, max_results: int = 50):
        _STAGE['name'] = 'search'
        original_cookiefiles = remove_search_cookiefiles(self)
        log_stage_event('stage_start', {
            'stage': 'search',
            'platform': getattr(self, 'platform', 'unknown'),
            'query': query,
            'max_results_requested': max_results,
            'max_results_used': 5,
            'cookies_used': False,
            'cookiefile_removed': any(original_cookiefiles.values()),
        })
        try:
            result = orig_search(self, query, max_results=5)
            log_stage_event('stage_result', {'stage': 'search', 'platform': getattr(self, 'platform', 'unknown'), 'query': query, 'result_count': len(result)})
            return result
        finally:
            restore_search_cookiefiles(self, original_cookiefiles)

    def patched_filter(self, videos, search_term):
        _STAGE['name'] = 'v1_pre_filter'
        log_stage_event('stage_start', {'stage': 'v1_pre_filter', 'search_term': search_term, 'input_count': len(videos)})
        passed, feedback = orig_filter(self, videos, search_term)
        log_stage_event('stage_result', {'stage': 'v1_pre_filter', 'search_term': search_term, 'input_count': len(videos), 'passed_count': len(passed)})
        return passed, feedback

    def patched_download(self, urls):
        _STAGE['name'] = 'download'
        log_stage_event('stage_start', {'stage': 'download', 'url_count': len(urls)})
        result = orig_download(self, urls)
        downloaded_paths = [path for path in result.values() if path]
        missing_count = len(result) - len(downloaded_paths)
        log_stage_event('stage_result', {
            'stage': 'download',
            'url_count': len(urls),
            'result_count': len(result),
            'downloaded_count': len(downloaded_paths),
            'missing_file_count': missing_count,
            'sample_paths': downloaded_paths[:3],
        })
        return result

    def patched_coarse(self, video_path):
        _STAGE['name'] = 'v2_coarse'
        log_stage_event('stage_start', {'stage': 'v2_coarse', 'video_path': video_path})
        result = orig_coarse(self, video_path)
        _PASSES['v2_coarse'] += int(bool(result[0]))
        log_stage_event('stage_result', {
            'stage': 'v2_coarse',
            'video_path': video_path,
            'passed': bool(result[0]),
            'fail_reason': result[1],
            'clip_count': len(result[2]),
        })
        return result

    def patched_fine(self, video_paths, video_id="", platform=""):
        _STAGE['name'] = 'v2_fine'
        log_stage_event('stage_start', {'stage': 'v2_fine', 'clip_count': len(video_paths), 'video_id': video_id, 'platform': platform})
        result = orig_fine(self, video_paths, video_id=video_id, platform=platform)
        _PASSES['v2_fine'] += int(bool(result[0]))
        log_stage_event('stage_result', {
            'stage': 'v2_fine',
            'clip_count': len(video_paths),
            'video_id': video_id,
            'platform': platform,
            'passed': bool(result[0]),
            'fail_reason': result[1],
        })
        return result

    def patched_adapt(self):
        result = orig_adapt(self)
        _LOOPS['count'] += 1
        total_loop_count = initial_loop_count + _LOOPS['count']
        logger.log('loop_complete', {
            'loop_count': _LOOPS['count'],
            'total_loop_count': total_loop_count,
        })
        if (
            total_loop_count >= MAX_TOTAL_PIPELINE_LOOPS
            and _qualified_count() < TARGET_QUALIFIED
        ):
            logger.log('max_pipeline_loops_reached', {
                'total_loop_count': total_loop_count,
                'max_total_pipeline_loops': MAX_TOTAL_PIPELINE_LOOPS,
            })
            sys.exit(EXIT_MAX_PIPELINE_LOOPS)
        return result

    def patched_get_in_progress_run(self):
        result = orig_get_in_progress_run(self)
        if result:
            logger.log('resume_detected', {'run': result})
        else:
            logger.log('resume_check', {'run': None})
        return result

    def patched_create_pipeline_run(self, run_id: str, stage: str = 'search'):
        logger.log('pipeline_run_create', {'run_id': run_id, 'stage': stage})
        return orig_create_pipeline_run(self, run_id, stage=stage)

    def patched_update_pipeline_run_stage(self, run_id: str, stage: str, status: str = 'in_progress'):
        key = (run_id, stage, status)
        if key not in seen_pipeline_run_stages:
            seen_pipeline_run_stages.add(key)
            logger.log('pipeline_run_stage', {'run_id': run_id, 'stage': stage, 'status': status})
        return orig_update_pipeline_run_stage(self, run_id, stage, status=status)

    def patched_complete_pipeline_run(self, run_id: str):
        logger.log('pipeline_run_complete', {'run_id': run_id})
        return orig_complete_pipeline_run(self, run_id)

    def patched_save_checkpoint(self, video_id: str, platform: str, stage: str, progress_data):
        logger.log('v2_checkpoint_save', {'video_id': video_id, 'platform': platform, 'stage': stage})
        return orig_save_checkpoint(self, video_id, platform, stage, progress_data)

    def patched_load_checkpoint(self, video_id: str, platform: str, stage: str):
        result = orig_load_checkpoint(self, video_id, platform, stage)
        logger.log('v2_checkpoint_load', {'video_id': video_id, 'platform': platform, 'stage': stage, 'found': result is not None})
        return result

    YtDlpSearchApi.search = patched_search
    PreFilter.filter = patched_filter
    DefaultDownloader.download = patched_download
    V2ContentFilter._coarse_filter = patched_coarse
    V2ContentFilter._fine_filter = patched_fine
    AdaptiveScheduler.adapt_strategy = patched_adapt
    VideoStore.get_in_progress_run = patched_get_in_progress_run
    VideoStore.create_pipeline_run = patched_create_pipeline_run
    VideoStore.update_pipeline_run_stage = patched_update_pipeline_run_stage
    VideoStore.complete_pipeline_run = patched_complete_pipeline_run
    VideoStore.save_checkpoint = patched_save_checkpoint
    VideoStore.load_checkpoint = patched_load_checkpoint


def _run_pipeline(config_path: str, state: Dict[str, Any], logger: JsonlLogger) -> None:
    try:
        cfg = ConfigLoader(config_path)
        logger.log('pipeline_start', {'config_path': config_path})
        orchestrator.run_pipeline(cfg)
        state['exit'] = 'normal'
    except Exception as e:
        state['exit'] = 'error'
        state['error'] = str(e)
        state['traceback'] = traceback.format_exc()
        logger.log('pipeline_exception', {'error': str(e), 'traceback': state['traceback']})


def worker_main() -> int:
    logger = JsonlLogger(LOG_PATH)
    pipeline_state: Dict[str, Any] = {'exit': None}
    _install_stage_instrumentation(logger)
    try:
        _run_pipeline(CONFIG_PATH, pipeline_state, logger)
    except SystemExit as e:
        pipeline_state['exit'] = 'system_exit'
        pipeline_state['error'] = f'SystemExit({e.code!r})'
        logger.log('pipeline_worker_exit', {
            'pipeline_exit': pipeline_state.get('exit'),
            'error': pipeline_state.get('error'),
            'exit_code': e.code,
            'v2_coarse_passes': _PASSES['v2_coarse'],
            'v2_fine_passes': _PASSES['v2_fine'],
        })
        logger.close()
        raise
    logger.log('pipeline_worker_exit', {
        'pipeline_exit': pipeline_state.get('exit'),
        'error': pipeline_state.get('error'),
        'v2_coarse_passes': _PASSES['v2_coarse'],
        'v2_fine_passes': _PASSES['v2_fine'],
    })
    logger.close()
    if pipeline_state.get('traceback'):
        print(pipeline_state['traceback'])
    return 0 if pipeline_state.get('exit') == 'normal' else 1


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the AutoFaceLabeler integration test.')
    parser.add_argument('--pipeline-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument(
        '--enable-breakpoint-restarts',
        dest='breakpoint_restarts_enabled',
        action='store_true',
        default=BREAKPOINT_RESTARTS_ENABLED,
        help='Enable SIGKILL/restart checkpoint validation during the test.',
    )
    parser.add_argument(
        '--disable-breakpoint-restarts',
        dest='breakpoint_restarts_enabled',
        action='store_false',
        help='Disable SIGKILL/restart checkpoint validation during the test.',
    )
    parser.add_argument(
        '--max-kill-resume-cycles',
        type=int,
        default=MAX_KILL_RESUME_CYCLES,
        help='Maximum breakpoint restart validation cycles when restarts are enabled.',
    )
    parser.add_argument(
        '--max-runtime-sec',
        type=int,
        default=MAX_TEST_RUNTIME_SEC,
        help='Maximum wall-clock runtime before the integration test stops.',
    )
    parser.add_argument(
        '--max-total-pipeline-loops',
        type=int,
        default=MAX_TOTAL_PIPELINE_LOOPS,
        help='Maximum total pipeline loops before the integration test stops.',
    )
    parser.add_argument(
        '--target-qualified',
        type=int,
        default=TARGET_QUALIFIED,
        help='Stop once this many qualified videos have been obtained.',
    )
    parser.add_argument(
        '--monitor-interval-sec',
        type=int,
        default=MONITOR_INTERVAL_SEC,
        help='Seconds between monitor samples.',
    )
    parser.add_argument(
        '--kill-after-min-sec',
        type=int,
        default=KILL_AFTER_RANGE_SEC[0],
        help='Minimum child runtime before a breakpoint restart validation kill.',
    )
    parser.add_argument(
        '--kill-after-max-sec',
        type=int,
        default=KILL_AFTER_RANGE_SEC[1],
        help='Maximum child runtime before a breakpoint restart validation kill.',
    )
    args = parser.parse_args(argv)

    args.max_kill_resume_cycles = max(0, args.max_kill_resume_cycles)
    args.max_runtime_sec = max(1, args.max_runtime_sec)
    args.max_total_pipeline_loops = max(1, args.max_total_pipeline_loops)
    args.target_qualified = max(1, args.target_qualified)
    args.monitor_interval_sec = max(1, args.monitor_interval_sec)
    args.kill_after_min_sec = max(1, args.kill_after_min_sec)
    args.kill_after_max_sec = max(args.kill_after_min_sec, args.kill_after_max_sec)

    if not args.breakpoint_restarts_enabled:
        args.max_kill_resume_cycles = 0
    return args


def main(args: argparse.Namespace) -> int:
    start = time.time()
    logger = JsonlLogger(LOG_PATH)

    cpu_hot_since = None
    reason = None
    kill_count = 0
    restart_count = 0
    proc = _start_pipeline_worker(
        logger,
        restart_count,
        target_qualified=args.target_qualified,
        max_total_pipeline_loops=args.max_total_pipeline_loops,
    )
    proc_started_at = time.time()
    kill_after_range = (args.kill_after_min_sec, args.kill_after_max_sec)
    next_kill_after = (
        random.randint(*kill_after_range)
        if args.max_kill_resume_cycles > 0
        else None
    )

    logger.log('monitor_start', {
        'max_test_runtime_sec': args.max_runtime_sec,
        'monitor_interval_sec': args.monitor_interval_sec,
        'cpu_warn_pct': CPU_WARN_PCT,
        'gpu_mem_warn_pct': GPU_MEM_WARN_PCT,
        'breakpoint_restarts_enabled': args.breakpoint_restarts_enabled,
        'max_kill_resume_cycles': args.max_kill_resume_cycles,
        'max_total_pipeline_loops': args.max_total_pipeline_loops,
        'kill_after_range_sec': kill_after_range,
        'resume_wait_range_sec': RESUME_WAIT_RANGE_SEC,
        'target_qualified': args.target_qualified,
        'download_format': _config_value('download', 'format'),
        'enable_v2': _config_value('pipeline', 'enable_v2'),
        'checkpoint_dir': _checkpoint_dir(),
        'video_db': _db_path(),
        'first_kill_after_sec': next_kill_after,
    })

    try:
        while True:
            now = time.time()
            elapsed = now - start
            child_elapsed = now - proc_started_at

            if psutil:
                cpu_pct = psutil.cpu_percent(interval=0.2)
            else:
                cpu_pct = None

            gpu_stats = _read_gpu_stats()
            qualified_count = _qualified_count()
            total_loop_count = _loop_count_from_log()
            proc_returncode = proc.poll()

            monitor_event = {
                'elapsed_sec': round(elapsed, 2),
                'child_elapsed_sec': round(child_elapsed, 2),
                'pid': proc.pid,
                'proc_returncode': proc_returncode,
                'qualified_count': qualified_count,
                'total_loop_count': total_loop_count,
                'kill_count': kill_count,
                'restart_count': restart_count,
                'cpu_pct': cpu_pct,
                **gpu_stats,
            }
            logger.log('monitor', monitor_event)
            logger.log('process_snapshot', {'elapsed_sec': round(elapsed, 2), 'snapshot': _process_snapshot()})

            if qualified_count >= args.target_qualified:
                reason = 'SUCCESS'
                logger.log('qualified_stop_detected', {'qualified_count': qualified_count})
                break

            if proc_returncode is not None:
                if proc_returncode == 0:
                    reason = 'SUCCESS'
                    break
                if proc_returncode == EXIT_MAX_PIPELINE_LOOPS:
                    reason = 'MAX_PIPELINE_LOOPS'
                    logger.log('pipeline_process_exit', {'pid': proc.pid, 'returncode': proc_returncode})
                    break
                reason = 'ERROR'
                logger.log('pipeline_process_exit', {'pid': proc.pid, 'returncode': proc_returncode})
                break

            if (
                total_loop_count >= args.max_total_pipeline_loops
                and qualified_count < args.target_qualified
            ):
                reason = 'MAX_PIPELINE_LOOPS'
                logger.log('max_pipeline_loops_reached', {
                    'total_loop_count': total_loop_count,
                    'max_total_pipeline_loops': args.max_total_pipeline_loops,
                })
                break

            if elapsed >= args.max_runtime_sec:
                reason = 'TIMEOUT'
                break

            if cpu_pct is not None and cpu_pct > CPU_WARN_PCT:
                if cpu_hot_since is None:
                    cpu_hot_since = now
                elif (now - cpu_hot_since) >= CPU_SUSTAINED_SEC:
                    reason = 'SAFETY_KILL_CPU'
                    break
            else:
                cpu_hot_since = None

            gpu_mem_pct = gpu_stats.get('gpu_mem_pct')
            if gpu_mem_pct is not None and gpu_mem_pct > GPU_MEM_WARN_PCT:
                reason = 'SAFETY_KILL_GPU'
                break

            if (
                kill_count < args.max_kill_resume_cycles
                and next_kill_after is not None
                and child_elapsed >= next_kill_after
            ):
                _kill_process_group(proc, logger, reason='checkpoint_resume_validation')
                proc.wait(timeout=30)
                kill_count += 1
                if kill_count == args.max_kill_resume_cycles:
                    logger.log('max_kill_resume_cycles_reached', {
                        'kill_count': kill_count,
                        'message': 'no further random kills will be scheduled',
                    })
                wait_before_resume = random.randint(*RESUME_WAIT_RANGE_SEC)
                logger.log('resume_wait', {
                    'kill_count': kill_count,
                    'wait_sec': wait_before_resume,
                    'pipeline_runs_after_kill': _pipeline_runs(),
                })
                time.sleep(wait_before_resume)
                restart_count += 1
                proc = _start_pipeline_worker(
                    logger,
                    restart_count,
                    target_qualified=args.target_qualified,
                    max_total_pipeline_loops=args.max_total_pipeline_loops,
                )
                proc_started_at = time.time()
                next_kill_after = (
                    random.randint(*kill_after_range)
                    if kill_count < args.max_kill_resume_cycles
                    else None
                )
                logger.log('resume_restart', {
                    'kill_count': kill_count,
                    'restart_count': restart_count,
                    'next_kill_after_sec': next_kill_after,
                    'pipeline_runs_at_restart': _pipeline_runs(),
                })

            time.sleep(args.monitor_interval_sec)
    except KeyboardInterrupt:
        reason = 'INTERRUPTED'
    finally:
        if proc.poll() is None:
            if reason == 'SUCCESS':
                _terminate_process_group(proc, logger, reason='qualified_video_found')
            elif reason in {'TIMEOUT', 'SAFETY_KILL_CPU', 'SAFETY_KILL_GPU', 'INTERRUPTED', 'ERROR', 'MAX_PIPELINE_LOOPS'}:
                _terminate_process_group(proc, logger, reason=str(reason))

        final_elapsed = time.time() - start
        summary = {
            'result': reason,
            'elapsed_sec': round(final_elapsed, 2),
            'kill_count': kill_count,
            'restart_count': restart_count,
            'qualified_count': _qualified_count(),
            'total_loop_count': _loop_count_from_log(),
            'pipeline_runs': _pipeline_runs(),
            'checkpoint_dir': _checkpoint_dir(),
            'video_db': _db_path(),
        }
        logger.log('summary', summary)
        logger.close()

    # Cleanup test artifacts (keep log for reporting)
    for p in ['/tmp/afl_explorer_test.json', '/tmp/afl_test_faces.faiss']:
        try:
            os.remove(p)
        except OSError:
            pass

    print(json.dumps(summary, ensure_ascii=True, indent=2))

    return 0 if reason in {'SUCCESS', 'TIMEOUT', 'MAX_PIPELINE_LOOPS'} else 1


if __name__ == '__main__':
    parsed_args = _parse_args()
    if parsed_args.pipeline_worker:
        sys.exit(worker_main())
    sys.exit(main(parsed_args))
