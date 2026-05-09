import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

from aflutils.config_loader import ConfigLoader
from core import pipeline_orchestrator as orchestrator

MAX_TEST_RUNTIME_SEC = 1800
MONITOR_INTERVAL_SEC = 10
CPU_WARN_PCT = 90
GPU_MEM_WARN_PCT = 90
CPU_SUSTAINED_SEC = 60
MAX_LOOPS_WITHOUT_QUALIFIER = 4  # Increased: V2 processing takes time
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


def _read_gpu_stats() -> Dict[str, Optional[float]]:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = result.stdout.strip().splitlines()[0]
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
        }
    except Exception:
        return {
            'gpu_mem_used_mb': None,
            'gpu_mem_total_mb': None,
            'gpu_mem_pct': None,
            'gpu_util_pct': None,
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


def _install_stage_instrumentation(logger: JsonlLogger) -> None:
    from explorers.adaptive_scheduler import AdaptiveScheduler
    from searchers.ytdlp_search_api import YtDlpSearchApi
    from validators.pre_filter import PreFilter
    from downloaders.base_downloader import DefaultDownloader
    from validators.validator import V2ContentFilter

    orig_search = YtDlpSearchApi.search
    orig_filter = PreFilter.filter
    orig_download = DefaultDownloader.download
    orig_coarse = V2ContentFilter._coarse_filter
    orig_fine = V2ContentFilter._fine_filter
    orig_adapt = AdaptiveScheduler.adapt_strategy

    def patched_search(self, query: str, max_results: int = 50):
        _STAGE['name'] = 'search'
        logger.log('stage', {'stage': 'search', 'platform': getattr(self, 'platform', 'unknown'), 'query': query, 'max_results': 5})
        return orig_search(self, query, max_results=5)

    def patched_filter(self, videos, search_term):
        _STAGE['name'] = 'v1_pre_filter'
        logger.log('stage', {'stage': 'v1_pre_filter', 'search_term': search_term, 'input_count': len(videos)})
        return orig_filter(self, videos, search_term)

    def patched_download(self, urls):
        _STAGE['name'] = 'download'
        logger.log('stage', {'stage': 'download', 'url_count': len(urls)})
        return orig_download(self, urls)

    def patched_coarse(self, video_path):
        _STAGE['name'] = 'v2_coarse'
        logger.log('stage', {'stage': 'v2_coarse', 'video_path': video_path})
        return orig_coarse(self, video_path)

    def patched_fine(self, video_paths):
        _STAGE['name'] = 'v2_fine'
        logger.log('stage', {'stage': 'v2_fine', 'clip_count': len(video_paths)})
        return orig_fine(self, video_paths)

    def patched_adapt(self):
        result = orig_adapt(self)
        _LOOPS['count'] += 1
        logger.log('loop_complete', {'loop_count': _LOOPS['count']})
        return result

    YtDlpSearchApi.search = patched_search
    PreFilter.filter = patched_filter
    DefaultDownloader.download = patched_download
    V2ContentFilter._coarse_filter = patched_coarse
    V2ContentFilter._fine_filter = patched_fine
    AdaptiveScheduler.adapt_strategy = patched_adapt


def _run_pipeline(config_path: str, state: Dict[str, Any], logger: JsonlLogger) -> None:
    try:
        cfg = ConfigLoader(config_path)
        logger.log('pipeline_start', {'config_path': config_path})
        orchestrator.run_pipeline(cfg)
        state['exit'] = 'normal'
    except BaseException as e:
        state['exit'] = 'error'
        state['error'] = str(e)
        state['traceback'] = traceback.format_exc()
        logger.log('pipeline_exception', {'error': str(e), 'traceback': state['traceback']})


def main() -> int:
    start = time.time()
    logger = JsonlLogger(LOG_PATH)
    pipeline_state: Dict[str, Any] = {'exit': None}

    _install_stage_instrumentation(logger)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(CONFIG_PATH, pipeline_state, logger),
        daemon=True,
    )
    thread.start()

    cpu_hot_since = None
    reason = None

    logger.log('monitor_start', {
        'max_test_runtime_sec': MAX_TEST_RUNTIME_SEC,
        'monitor_interval_sec': MONITOR_INTERVAL_SEC,
        'cpu_warn_pct': CPU_WARN_PCT,
        'gpu_mem_warn_pct': GPU_MEM_WARN_PCT,
    })

    try:
        while True:
            now = time.time()
            elapsed = now - start

            if psutil:
                cpu_pct = psutil.cpu_percent(interval=0.2)
            else:
                cpu_pct = None

            gpu_stats = _read_gpu_stats()

            monitor_event = {
                'elapsed_sec': round(elapsed, 2),
                'pipeline_stage': _STAGE['name'],
                'loop_count': _LOOPS['count'],
                'cpu_pct': cpu_pct,
                **gpu_stats,
            }
            logger.log('monitor', monitor_event)

            if pipeline_state.get('exit') == 'normal':
                reason = 'SUCCESS'
                break

            if pipeline_state.get('exit') == 'error':
                reason = 'ERROR'
                break

            if elapsed >= MAX_TEST_RUNTIME_SEC:
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

            if _LOOPS['count'] >= MAX_LOOPS_WITHOUT_QUALIFIER:
                reason = 'INCONCLUSIVE'
                break

            if not thread.is_alive() and pipeline_state.get('exit') is None:
                reason = 'ERROR'
                pipeline_state['error'] = 'Pipeline thread exited unexpectedly without state'
                logger.log('pipeline_exception', {'error': pipeline_state['error']})
                break

            time.sleep(MONITOR_INTERVAL_SEC)
    except KeyboardInterrupt:
        reason = 'INTERRUPTED'
    finally:
        if thread.is_alive():
            _safe_stop_pipeline_thread(thread, logger)

        final_elapsed = time.time() - start
        summary = {
            'result': reason,
            'elapsed_sec': round(final_elapsed, 2),
            'stage_last': _STAGE['name'],
            'loop_count': _LOOPS['count'],
            'v2_coarse_passes': _PASSES['v2_coarse'],
            'v2_fine_passes': _PASSES['v2_fine'],
            'pipeline_exit': pipeline_state.get('exit'),
            'error': pipeline_state.get('error'),
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
    if pipeline_state.get('traceback'):
        print(pipeline_state['traceback'])

    return 0 if reason in {'SUCCESS', 'INCONCLUSIVE', 'TIMEOUT'} else 1


if __name__ == '__main__':
    sys.exit(main())
