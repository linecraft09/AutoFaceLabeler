import os
import tempfile
from typing import List, Dict

import numpy as np
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import aflutils.video_utils as VU
from aflutils.logger import get_logger

logger = get_logger(__name__)


class SpeakerDetector:
    """使用 VAD 检测音频中是否有人说话"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            local_files_only=False
        )

    def detect_speech(self, audio_path: str) -> bool:
        """检测音频文件是否包含人声"""
        try:
            audio_np, sample_rate = sf.read(audio_path)

            # 如果是多声道，转单声道
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)

            # 确保类型为 float32
            audio_np = audio_np.astype(np.float32)

            result: List[Dict[str, List[int]]] = self.vad_pipeline(audio_np, sample_rate)

            if result and len(result) > 0 and len(result[0]) > 0:
                first = result[0]
                segments = first.get('value')
                if segments is None:
                    # Backward compatibility for possible schema variants.
                    segments = first.get('segments')
                if segments is None and 'text' in first and isinstance(first.get('text'), list):
                    segments = first.get('text')
                if segments is None:
                    logger.warning(f"Unexpected VAD output keys: {list(first.keys())}")
                    return False
                return len(segments) > 0

            return False
        except Exception as e:
            logger.error(f"Speech detection failed: {e}")
            return False

    def detect_speech_from_video(self, video_path: str) -> bool:
        """从视频提取音频并检测说话"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name

        try:
            VU.extract_audio(audio_path, video_path)
            return self.detect_speech(audio_path)
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return False
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
