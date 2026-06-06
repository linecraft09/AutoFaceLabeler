from __future__ import annotations

import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from annotators.base_annotator import BaseAnnotator
from core.models.annotation_models import Transcription

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - exercised only when dependency is absent
    WhisperModel = None  # type: ignore[assignment]


class WhisperAnnotator(BaseAnnotator):
    """Speech transcription annotator backed by faster-whisper."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_size = config.get("model_size", "large-v3")
        self.device = config.get("device", "cuda")
        self.compute_type = config.get("compute_type", "float16")
        self.transcribe_options = config.get("transcribe_options", {})
        self.model = None

    @property
    def label_name(self) -> str:
        return "transcription"

    def load(self) -> None:
        if self.model is not None:
            self._loaded = True
            return
        if WhisperModel is None:
            raise ImportError("faster-whisper is not installed")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._loaded = True

    def annotate(self, video_path: str, **kwargs: Any) -> Transcription:
        with tempfile.TemporaryDirectory(prefix="afl_whisper_") as temp_dir:
            audio_path = self.extract_audio(video_path, temp_dir)
            if audio_path is None:
                return Transcription(text="", language="", segments=[])
            return self.transcribe_audio(audio_path, **kwargs)

    def extract_audio(self, video_path: str, output_dir: str | Path) -> Optional[str]:
        output_path = Path(output_dir) / "audio.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        if not output_path.exists() or output_path.stat().st_size <= 44:
            return None
        return str(output_path)

    def transcribe_audio(self, audio_path: str, **kwargs: Any) -> Transcription:
        if self.is_silent_wav(audio_path):
            return Transcription(text="", language="", segments=[])

        if not self._loaded:
            self.load()
        assert self.model is not None

        options = dict(self.transcribe_options)
        options.update(kwargs)
        segments_iter, info = self.model.transcribe(audio_path, **options)

        segments = []
        texts = []
        for segment in segments_iter:
            text = str(getattr(segment, "text", "")).strip()
            confidence = self._segment_confidence(segment)
            segments.append(
                {
                    "start": float(getattr(segment, "start", 0.0)),
                    "end": float(getattr(segment, "end", 0.0)),
                    "text": text,
                    "confidence": confidence,
                }
            )
            if text:
                texts.append(text)

        language = str(getattr(info, "language", "") or "")
        return Transcription(text=" ".join(texts).strip(), language=language, segments=segments)

    @staticmethod
    def is_silent_wav(audio_path: str, rms_threshold: float = 1.0) -> bool:
        try:
            with wave.open(audio_path, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                if not frames:
                    return True
                samples = np.frombuffer(frames, dtype=np.int16)
        except (wave.Error, OSError):
            return False
        if samples.size == 0:
            return True
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        return rms <= rms_threshold

    @staticmethod
    def _segment_confidence(segment: Any) -> float:
        avg_logprob = getattr(segment, "avg_logprob", None)
        if avg_logprob is not None:
            try:
                return float(max(0.0, min(1.0, math.exp(float(avg_logprob)))))
            except (TypeError, ValueError, OverflowError):
                pass

        no_speech_prob = getattr(segment, "no_speech_prob", None)
        if no_speech_prob is not None:
            try:
                return float(max(0.0, min(1.0, 1.0 - float(no_speech_prob))))
            except (TypeError, ValueError):
                pass
        return 0.0
