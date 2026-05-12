import os
import json
import shutil
import subprocess
import sys
import tempfile
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from aflutils.logger import get_logger

def _find_ffmpeg() -> str:
    """Find ffmpeg binary: check PATH first, fall back to Windows hardcoded path."""
    path = shutil.which('ffmpeg')
    if path:
        return path
    # Fallback for Windows compatibility
    return "D:/Softwares/ffmpeg/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg"

def _find_ffprobe() -> str:
    """Find ffprobe binary: check PATH first, fall back to Windows hardcoded path."""
    path = shutil.which('ffprobe')
    if path:
        return path
    # Fallback for Windows compatibility
    return 'D:/Softwares/ffmpeg/ffmpeg-8.0-essentials_build/bin/ffprobe'

FFMPEG = _find_ffmpeg()
FFPROBE = _find_ffprobe()

logger = get_logger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    codec_name: Optional[str]
    pix_fmt: Optional[str]
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    has_audio: bool
    color_space: Optional[str] = None
    color_transfer: Optional[str] = None
    color_primaries: Optional[str] = None


def get_cuda_lib_path() -> str:
    """Build an LD_LIBRARY_PATH that exposes CUDA libs without forcing conda system libs."""
    paths = []

    paths.append("/usr/lib/wsl/lib")

    prefixes = [os.environ.get("CONDA_PREFIX"), sys.prefix]
    for prefix in [p for p in prefixes if p]:
        python_versions = ("3.10", "3.11", "3.12", f"{sys.version_info.major}.{sys.version_info.minor}")
        for python_version in python_versions:
            nvidia_root = Path(prefix) / f"lib/python{python_version}/site-packages/nvidia"
            if not nvidia_root.is_dir():
                continue
            for candidate in sorted(nvidia_root.glob("*/lib")):
                if candidate.is_dir():
                    paths.append(str(candidate))

    deduped = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return ":".join(deduped)


def _build_ffmpeg_env(use_cuda_libs: bool = False) -> dict:
    """Return a subprocess environment for ffmpeg/ffprobe.

    Plain ffmpeg/ffprobe commands should not inherit conda's LD_LIBRARY_PATH because
    system ffmpeg can otherwise load incompatible ncurses/tinfo libraries. CUDA/NVENC
    calls get the CUDA paths prepended only when they need them.
    """
    env = dict(os.environ)
    if not use_cuda_libs:
        env.pop("LD_LIBRARY_PATH", None)
        return env

    cuda_path = get_cuda_lib_path()
    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_cuda_paths = [
        path for path in current.split(":")
        if path and any(marker in path.lower() for marker in ("cuda", "nvidia", "wsl/lib"))
    ]
    paths = [path for path in [cuda_path, *current_cuda_paths] if path]
    if paths:
        env["LD_LIBRARY_PATH"] = ":".join(paths)
    return env


def _build_cuda_env() -> dict:
    return _build_ffmpeg_env(use_cuda_libs=True)


def _run_command(cmd: List[str], timeout: int = 300, use_cuda_libs: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_build_ffmpeg_env(use_cuda_libs=use_cuda_libs),
    )


def _run_ffmpeg(cmd: List[str], timeout: int = 300, use_cuda_libs: bool = False) -> subprocess.CompletedProcess:
    return _run_command(cmd, timeout=timeout, use_cuda_libs=use_cuda_libs)


def _stderr_tail(stderr: str, max_chars: int = 4000) -> str:
    if not stderr:
        return ""
    return stderr if len(stderr) <= max_chars else stderr[-max_chars:]


def _validate_output_file(path: str, description: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"{description}生成失败或文件为空")


def _remove_partial_output(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError as e:
        logger.warning(f"Failed to remove partial output {path}: {e}")


@lru_cache(maxsize=16)
def _ffmpeg_has_encoder(encoder: str) -> bool:
    try:
        result = _run_command([FFMPEG, "-hide_banner", "-encoders"], timeout=10)
        if result.returncode != 0:
            return False
        return encoder in result.stdout
    except Exception:
        return False


@lru_cache(maxsize=1)
def _has_nvidia_gpu() -> bool:
    try:
        result = _run_command(["nvidia-smi"], timeout=10)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _can_try_h264_nvenc() -> bool:
    return _has_nvidia_gpu() and _ffmpeg_has_encoder("h264_nvenc")


def _check_ffmpeg_cuda_support() -> bool:
    """Backward-compatible CUDA/NVENC capability check."""
    return _can_try_h264_nvenc()


def _parse_fps(value: Optional[str]) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return 0.0
            return float(numerator) / denominator_value
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def probe_video(video_path: str) -> Optional[VideoInfo]:
    """Return basic video/audio metadata from ffprobe."""
    cmd = [
        FFPROBE, "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        video_path,
    ]
    try:
        result = _run_command(cmd, timeout=20)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"ffprobe failed for {video_path}: {e}")
        return None
    if result.returncode != 0:
        logger.warning(f"ffprobe failed for {video_path}: {_stderr_tail(result.stderr)}")
        return None

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid ffprobe JSON for {video_path}: {e}")
        return None

    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video_stream:
        return None

    format_info = payload.get("format", {})
    fps = _parse_fps(video_stream.get("avg_frame_rate")) or _parse_fps(video_stream.get("r_frame_rate"))

    duration = 0.0
    for raw_duration in (video_stream.get("duration"), format_info.get("duration")):
        try:
            if raw_duration is not None:
                duration = float(raw_duration)
                break
        except (TypeError, ValueError):
            continue

    try:
        frame_count = int(video_stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0 and fps > 0 and duration > 0:
        frame_count = int(round(fps * duration))

    return VideoInfo(
        codec_name=(video_stream.get("codec_name") or "").lower() or None,
        pix_fmt=(video_stream.get("pix_fmt") or "").lower() or None,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=fps,
        duration=duration,
        frame_count=frame_count,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
        color_space=video_stream.get("color_space"),
        color_transfer=video_stream.get("color_transfer"),
        color_primaries=video_stream.get("color_primaries"),
    )


def sample_video_per_sec(input_path: str, suffix: str = "_sample", output_ext: str = ".mp4") -> str:
    """
    从输入视频中每秒抽取第一帧，合并生成 sample 视频，并返回其路径。

    Args:
        input_path: 原始视频文件路径
        suffix: 输出文件名后缀（不含扩展名部分），默认为 "_sample"
        output_ext: 输出视频扩展名，默认为 ".mp4"

    Returns:
        sample 视频的完整路径

    Raises:
        FileNotFoundError: 输入文件不存在或 ffmpeg 不可用
        RuntimeError: ffmpeg 命令执行失败
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"视频文件不存在: {input_path}")

    # 构造输出路径
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(base_dir, f"{base_name}{suffix}{output_ext}")

    try:
        result = _run_command([FFMPEG, "-version"], timeout=10)
        if result.returncode != 0:
            raise FileNotFoundError
    except (subprocess.SubprocessError, FileNotFoundError):
        raise FileNotFoundError("未找到 ffmpeg，请确保已安装并加入 PATH")

    vf = "fps=1,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    attempts = []
    if _can_try_h264_nvenc():
        attempts.append((
            "h264_nvenc",
            [
                FFMPEG, "-y",
                "-i", input_path,
                "-vf", vf,
                "-an",
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-cq", "23",
                output_path,
            ],
            True,
        ))
    attempts.append((
        "libx264",
        [
            FFMPEG, "-y",
            "-i", input_path,
            "-vf", vf,
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            output_path,
        ],
        False,
    ))

    errors = []
    for label, cmd, use_cuda_libs in attempts:
        result = _run_ffmpeg(cmd, timeout=300, use_cuda_libs=use_cuda_libs)
        if result.returncode == 0:
            try:
                _validate_output_file(output_path, "sample 视频")
                if label != "h264_nvenc":
                    logger.info(f"sample_video_per_sec used {label} for {input_path}")
                return output_path
            except RuntimeError as e:
                errors.append(f"{label}: {e}")
                _remove_partial_output(output_path)
                continue

        errors.append(f"{label} 返回码 {result.returncode}:\n{_stderr_tail(result.stderr)}")
        _remove_partial_output(output_path)
        if label == "h264_nvenc":
            logger.warning(f"h264_nvenc sample generation failed for {input_path}; falling back to libx264")

    raise RuntimeError(f"ffmpeg 执行失败，sample 视频生成失败:\n" + "\n\n".join(errors))


def get_video_fps(video_path: str) -> float:
    info = probe_video(video_path)
    if info and info.fps > 0:
        return info.fps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_frames(video_path: str) -> float:
    info = probe_video(video_path)
    if info and info.frame_count > 0:
        return info.frame_count
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def get_video_secs(video_path: str) -> float:
    info = probe_video(video_path)
    if info and info.duration > 0:
        return info.duration
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or not math.isfinite(fps) or fps <= 0:
        logger.warning(f"Invalid FPS ({fps}) for {video_path}, returning 0.0 seconds")
        return 0.0
    return total_frames / fps


def read_sampled_frames(video_path: str, indices: List[int]) -> Tuple[List[np.ndarray], dict]:
    """
    Read only the requested frame indices using seek, with sequential-read fallback.

    Args:
        video_path: Path to the video file.
        indices: Frame indices to read. Returned frames follow this order.

    Returns:
        (frames, meta), where meta contains total_frames, fps, width, and height.
    """
    requested = [int(idx) for idx in indices]
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        meta = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        if not requested:
            return [], meta

        frames_by_index = {}
        seek_failed = False
        for idx in sorted(set(requested)):
            if idx < 0 or idx >= meta["total_frames"]:
                seek_failed = True
                break
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
                seek_failed = True
                break
            ok, frame = cap.read()
            actual_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if not ok or actual_idx != idx:
                seek_failed = True
                break
            frames_by_index[idx] = frame

        if seek_failed:
            logger.warning(f"Frame seek validation failed for {video_path}; falling back to sequential read")
            return _read_sampled_frames_sequential(video_path, requested, meta)

        return [frames_by_index[idx] for idx in requested if idx in frames_by_index], meta
    finally:
        cap.release()


def _read_sampled_frames_sequential(video_path: str, indices: List[int], meta: dict) -> Tuple[List[np.ndarray], dict]:
    wanted = set(indices)
    frames_by_index = {}
    cap = cv2.VideoCapture(video_path)
    try:
        frame_idx = 0
        while cap.isOpened() and len(frames_by_index) < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in wanted:
                frames_by_index[frame_idx] = frame
            frame_idx += 1
    finally:
        cap.release()
    return [frames_by_index[idx] for idx in indices if idx in frames_by_index], meta


def clip_video(video_path: str, segments: List[Tuple], unit: str = "frame", max_workers: int = None) -> List[str]:
    """
    根据帧号或时间范围剪切视频（多线程并行），每个片段单独保存。
    文件名格式：原文件名_clipped_{序号}.扩展名

    :param video_path: 原视频路径
    :param segments: 列表，每个元素为 (start, end)，包含起始和结束。
                     当 unit="frame" 时，start/end 为整数帧号（闭区间）；
                     当 unit="second" 时，start/end 为浮点数秒数（闭区间，包含结束时刻的帧）。
    :param unit: 单位类型，"frame" 或 "second"，默认 "frame"
    :param max_workers: 最大并行线程数，默认为 min(4, len(segments))，可自行调整
    :return: 剪切成功的视频文件路径列表（顺序与 segments 一致，失败片段跳过）
    """
    if not segments:
        logger.warning(f"No segments to clip for {video_path}")
        return []

    # 获取帧率
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        logger.error(f"Invalid fps for {video_path}")
        return []

    base, _ = os.path.splitext(video_path)

    # 辅助函数：根据单位计算起始/结束时间（秒）
    def get_times(seg: Tuple) -> Tuple[float, float]:
        start_val, end_val = seg
        if unit == "frame":
            start_time = start_val / fps
            end_time = (end_val + 1) / fps  # 包含结束帧
        elif unit == "second":
            start_time = start_val
            end_time = end_val + (1.0 / fps)  # 包含结束时刻的帧
        else:
            raise ValueError(f"不支持的 unit 参数: {unit}，请使用 'frame' 或 'second'")
        return start_time, end_time

    # 单个片段的剪辑任务（供线程调用）
    def clip_one_segment(idx: int, seg: Tuple) -> Optional[str]:
        try:
            start_time, end_time = get_times(seg)
        except ValueError as e:
            logger.error(f"Segment {idx}: {e}")
            return None

        output_path = f"{base}_clipped_{idx}.mkv"

        success = _clip_with_copy(video_path, start_time, end_time, output_path)
        if not success:
            success = _clip_with_software(video_path, start_time, end_time, output_path)

        if success:
            logger.info(f"Clipped segment {idx} saved to {output_path}")
            return output_path
        else:
            logger.error(f"Failed to clip segment {idx} from {video_path}")
            return None

    # 并行执行
    if max_workers is None:
        max_workers = min(4, len(segments))  # 默认最多4个并发，避免资源耗尽
    output_paths = [None] * len(segments)  # 预分配，保持顺序

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(clip_one_segment, idx, seg): idx for idx, seg in enumerate(segments)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                output_paths[idx] = result
            except Exception as e:
                logger.error(f"Unexpected error in segment {idx}: {e}")

    # 过滤掉失败的片段，返回成功路径列表（保持原顺序）
    return [path for path in output_paths if path is not None]


def _get_video_codec(video_path: str) -> Optional[str]:
    """使用 ffprobe 获取视频流的编码名称，如 h264、hevc"""
    info = probe_video(video_path)
    return info.codec_name if info else None


def _clip_with_copy(video_path: str, start: float, end: float, output: str) -> bool:
    """使用关键帧流复制进行剪辑（不重新编码）。"""
    # 无损流复制 + 关键帧剪辑：
    # -ss 放在 -i 前做关键帧级 seek（速度快，不解码）
    # -c copy 不做编解码
    # 右边界微扩展，尽量覆盖 end 对应时刻
    safe_start = max(0.0, start)
    safe_end = end + 1e-3

    cmd = [
        FFMPEG,
        "-ss", f"{safe_start:.6f}",
        "-to", f"{safe_end:.6f}",
        "-i", video_path,
        "-copyinkf",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output,
        "-y"
    ]
    result = _run_ffmpeg(cmd, timeout=300)
    if result.returncode == 0:
        try:
            _validate_output_file(output, "clip 视频")
        except RuntimeError as e:
            logger.warning(f"Copy clipping produced invalid output for {video_path}: {e}")
            _remove_partial_output(output)
            return False
        return True
    logger.warning(f"Copy clipping failed for {video_path}: {_stderr_tail(result.stderr)}")
    _remove_partial_output(output)
    return False


def _clip_with_software(video_path: str, start: float, end: float, output: str) -> bool:
    """使用软件编码（libx264）进行兼容性剪辑。"""
    cmd = [
        FFMPEG, "-i", video_path,
        "-ss", str(start), "-to", str(end),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        "-avoid_negative_ts", "make_zero",
        output, "-y"
    ]
    result = _run_ffmpeg(cmd, timeout=300)
    if result.returncode == 0:
        try:
            _validate_output_file(output, "clip 视频")
        except RuntimeError as e:
            logger.error(f"Software clipping produced invalid output for {video_path}: {e}")
            _remove_partial_output(output)
            return False
        return True
    logger.error(f"Software clipping error: {_stderr_tail(result.stderr)}")
    _remove_partial_output(output)
    return False


def remove_videos(video_paths: List[str]) -> None:
    """
    批量删除视频文件，忽略不存在文件，单文件失败不中断。

    :param video_paths: 待删除视频路径列表
    """
    if not video_paths:
        return

    for video_path in video_paths:
        if not video_path:
            continue
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            logger.warning(f"Failed to remove video {video_path}: {e}")


def check_audio(video_path: str) -> bool:
    """使用 ffprobe 检查是否有音频流"""
    info = probe_video(video_path)
    return bool(info and info.has_audio)


def extract_audio(audio_path: str, video_path: str):
    cmd = [
        FFMPEG, '-i', video_path,
        '-vn',
        '-ac', '1',
        '-ar', '16000',
        '-y', audio_path,
    ]
    result = _run_ffmpeg(cmd, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 音频提取失败 (返回码 {result.returncode}):\n{_stderr_tail(result.stderr)}")
    _validate_output_file(audio_path, "audio")


def concat_videos(video_paths: List[str]) -> Optional[str]:
    """
    使用 ffmpeg 将多个视频拼接成一个视频。
    输出文件保存在第一个视频所在目录，名称为 "concat_<timestamp>.mkv"。

    :param video_paths: 视频路径列表（顺序决定拼接顺序）
    :return: 成功返回输出文件路径，失败返回 None
    """
    if not video_paths:
        return None

    # 生成输出路径（与第一个视频同目录，避免权限问题）
    first_dir = os.path.dirname(video_paths[0])
    output_path = os.path.join(
        first_dir,
        f"concat_{int(os.path.getmtime(video_paths[0]))}.mkv"
    )

    # 方法1：使用 concat demuxer + 流复制（最快，要求格式一致）
    if _concat_with_copy(video_paths, output_path):
        return output_path

    # 方法2：回退到重新编码（兼容不同格式）
    if _concat_with_reencode(video_paths, output_path):
        return output_path

    return None


def _concat_with_copy(video_paths: List[str], output_path: str) -> bool:
    """尝试使用流复制方式拼接（不重新编码）"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for path in video_paths:
            # 转义路径中的单引号
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        list_file = f.name

    try:
        cmd = [
            FFMPEG, "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path, "-y"
        ]
        result = _run_ffmpeg(cmd, timeout=300)
        if result.returncode != 0:
            logger.warning(f"Concat copy failed: {_stderr_tail(result.stderr)}")
            _remove_partial_output(output_path)
            return False
        _validate_output_file(output_path, "concat 视频")
        return True
    except RuntimeError as e:
        logger.warning(f"Concat copy produced invalid output: {e}")
        # 流复制失败（如编码/尺寸不一致），删除可能产生的半成品
        if os.path.exists(output_path):
            os.unlink(output_path)
        return False
    finally:
        os.unlink(list_file)


def _concat_with_reencode(video_paths: List[str], output_path: str) -> bool:
    """重新编码视频和音频以保证拼接成功（使用 libx264 + aac）"""
    # 创建临时文件列表（同 copy 方式，但最终重新编码）
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for path in video_paths:
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        list_file = f.name

    try:
        cmd = [
            FFMPEG, "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path, "-y"
        ]
        result = _run_ffmpeg(cmd, timeout=600)
        if result.returncode != 0:
            logger.error(f"Concat reencode failed: {_stderr_tail(result.stderr)}")
            _remove_partial_output(output_path)
            return False
        _validate_output_file(output_path, "concat 视频")
        return True
    except RuntimeError as e:
        logger.error(f"Concat reencode produced invalid output: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return False
    finally:
        os.unlink(list_file)
