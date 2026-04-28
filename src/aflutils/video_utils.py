import os
import shutil
import subprocess
import tempfile
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import cv2

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


def _check_ffmpeg_cuda_support() -> bool:
    """Check if the ffmpeg binary supports CUDA/NVENC."""
    try:
        result = subprocess.run([FFMPEG, "-encoders"], capture_output=True, text=True, timeout=10)
        return 'nvenc' in result.stdout or 'cuda' in result.stdout
    except Exception:
        return False


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
        # 检查 ffmpeg 是否可用
        subprocess.run([FFMPEG, "-version"], capture_output=True, check=True, timeout=10)
    except (subprocess.SubprocessError, FileNotFoundError):
        raise FileNotFoundError("未找到 ffmpeg，请确保已安装并加入 PATH")

    # Detect CUDA support
    has_cuda = _check_ffmpeg_cuda_support()

    # 构建 ffmpeg 命令
    if has_cuda:
        cmd = [
            FFMPEG, "-y",
            "-hwaccel", "cuda",
            "-i", input_path,
            "-vf", "fps=1",
            "-an",
            "-c:v", "h264_nvenc",
            "-preset", "p1",
            "-cq", "23",
            output_path
        ]
    else:
        cmd = [
            FFMPEG, "-y",
            "-i", input_path,
            "-vf", "fps=1",
            "-an",
            output_path
        ]

    # 执行抽取和合并
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 执行失败 (返回码 {result.returncode}):\n{result.stderr}")

    # 检查生成的文件是否有效（大小 > 0）
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("sample 视频生成失败或文件为空")

    return output_path


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_frames(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def get_video_secs(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or not math.isfinite(fps) or fps <= 0:
        logger.warning(f"Invalid FPS ({fps}) for {video_path}, returning 0.0 seconds")
        return 0.0
    return total_frames / fps


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

    # 获取视频编码格式（用于选择硬件加速器）
    codec_name = _get_video_codec(video_path)
    if not codec_name:
        logger.warning(f"Cannot detect codec of {video_path}, fallback to software encoding")
        use_cuda = False
    else:
        use_cuda = _is_cuda_supported(codec_name)

    base, ext = os.path.splitext(video_path)

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

        output_path = f"{base}_clipped_{idx}{ext}"
        success = False

        if use_cuda:
            success = _clip_with_cuda(video_path, start_time, end_time, output_path)
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
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        codec = result.stdout.strip().lower()
        return codec if codec else None
    except subprocess.CalledProcessError:
        return None


def _is_cuda_supported(codec: str) -> bool:
    """检查系统是否支持 CUDA 硬件加速且视频编码为 H.264 或 HEVC"""
    # 检查 NVIDIA GPU 是否存在（简单方法：运行 nvidia-smi）
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    # 可选：进一步检查 ffmpeg 是否支持对应的 cuvid/nvenc 解码器
    # 这里假设标准 ffmpeg 版本已支持，若不支持则在 _clip_with_cuda 中会失败并回退
    return True


def _clip_with_cuda(video_path: str, start: float, end: float, output: str) -> bool:
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
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"CUDA clipping error: {e.stderr}")
        return False


def _clip_with_software(video_path: str, start: float, end: float, output: str) -> bool:
    """使用软件编码（libx264/libx265）进行剪辑（重新编码）"""
    # 检测原视频编码以决定软件编码器
    codec = _get_video_codec(video_path)
    if codec in ("hevc", "h265"):
        vcodec = "libx265"
        crf = "28"  # x265 的 CRF 默认值稍高
    else:
        vcodec = "libx264"
        crf = "23"

    cmd = [
        FFMPEG, "-i", video_path,
        "-ss", str(start), "-to", str(end),
        "-c:v", vcodec, "-crf", crf, "-preset", "fast",
        "-c:a", "copy",
        "-avoid_negative_ts", "make_zero",
        output, "-y"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Software clipping error: {e.stderr}")
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


#
# def clip_video(video_path: str, segments: List[Tuple], unit: str = "frame") -> Optional[str]:
#     """
#     根据帧号或时间范围剪切视频，合并多个片段，输出到同一文件夹下（文件名加 _clipped 后缀）。
#
#     :param video_path: 原视频路径
#     :param segments: 列表，每个元素为 (start, end)，包含起始和结束。
#                      当 unit="frame" 时，start/end 为整数帧号（闭区间）；
#                      当 unit="second" 时，start/end 为浮点数秒数（闭区间，包含结束时刻的帧）。
#     :param unit: 单位类型，"frame" 或 "second"，默认 "frame"
#     :return: 剪切后的视频路径，失败返回 None
#     """
#     if not segments:
#         logger.warning(f"No segments to clip for {video_path}")
#         return video_path  # 无剪辑必要，返回原路径
#
#     # 获取帧率
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()
#     if fps <= 0:
#         logger.error(f"Invalid fps for {video_path}")
#         return None
#
#     # 生成输出路径
#     base, ext = os.path.splitext(video_path)
#     output_path = f"{base}_clipped{ext}"
#
#     # 辅助函数：根据单位计算起始/结束时间（秒）
#     def get_times(seg: Tuple) -> Tuple[float, float]:
#         start_val, end_val = seg
#         if unit == "frame":
#             start_time = start_val / fps
#             end_time = (end_val + 1) / fps  # 包含结束帧
#         elif unit == "second":
#             start_time = start_val
#             end_time = end_val + (1.0 / fps)  # 包含结束时刻的帧
#         else:
#             raise ValueError(f"不支持的 unit 参数: {unit}，请使用 'frame' 或 'second'")
#         return start_time, end_time
#
#     # 如果只有一个片段，直接剪辑
#     if len(segments) == 1:
#         try:
#             start_time, end_time = get_times(segments[0])
#         except ValueError as e:
#             logger.error(str(e))
#             return None
#
#         cmd = [
#             FFMPEG, "-i", video_path,
#             "-ss", str(start_time),
#             "-to", str(end_time),
#             "-c", "copy",  # 快速复制，不重新编码
#             "-avoid_negative_ts", "make_zero",
#             output_path, "-y"
#         ]
#         try:
#             subprocess.run(cmd, check=True, capture_output=True, text=True)
#             logger.info(f"Clipped video saved to {output_path}")
#             return output_path
#         except subprocess.CalledProcessError as e:
#             logger.error(f"FFmpeg error: {e.stderr}")
#             return None
#
#     # 多个片段：分别剪辑到临时文件，然后合并
#     temp_files = []
#     try:
#         for i, seg in enumerate(segments):
#             start_time, end_time = get_times(seg)
#             temp_fd, temp_path = tempfile.mkstemp(suffix=f"_seg{i}{ext}", prefix="clip_")
#             os.close(temp_fd)
#             cmd = [
#                 FFMPEG, "-i", video_path,
#                 "-ss", str(start_time),
#                 "-to", str(end_time),
#                 "-c", "copy",
#                 "-avoid_negative_ts", "make_zero",
#                 temp_path, "-y"
#             ]
#             subprocess.run(cmd, check=True, capture_output=True, text=True)
#             temp_files.append(temp_path)
#
#         # 创建 concat 文件列表
#         concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
#         for temp_path in temp_files:
#             # 转义特殊字符（例如单引号）
#             escaped = temp_path.replace("'", "'\\''")
#             concat_file.write(f"file '{escaped}'\n")
#         concat_file.close()
#
#         # 使用 concat 协议合并
#         cmd_concat = [
#             FFMPEG, "-f", "concat", "-safe", "0",
#             "-i", concat_file.name,
#             "-c", "copy",
#             output_path, "-y"
#         ]
#         subprocess.run(cmd_concat, check=True, capture_output=True, text=True)
#         logger.info(f"Merged clipped video saved to {output_path}")
#         return output_path
#     except subprocess.CalledProcessError as e:
#         logger.error(f"FFmpeg error during clipping/merging: {e.stderr}")
#         return None
#     except ValueError as e:
#         logger.error(str(e))
#         return None
#     finally:
#         # 清理临时文件
#         for temp_path in temp_files:
#             try:
#                 os.unlink(temp_path)
#             except OSError:
#                 pass
#         if 'concat_file' in locals():
#             try:
#                 os.unlink(concat_file.name)
#             except OSError:
#                 pass


def check_audio(video_path: str) -> bool:
    """使用 ffprobe 检查是否有音频流"""
    try:
        cmd = [
            FFPROBE, '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and result.stdout.strip() == 'audio'
    except Exception as e:
        logger.error(f"Audio check failed: {e}")
        return False


def extract_audio(audio_path: str, video_path: str):
    cmd = [FFMPEG, '-hwaccel', 'cuda', '-i', video_path, '-ac', '1', '-ar',
           '16000', '-y', audio_path]
    subprocess.run(cmd, capture_output=True, check=True)


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
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
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
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path, "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        if os.path.exists(output_path):
            os.unlink(output_path)
        return False
    finally:
        os.unlink(list_file)
