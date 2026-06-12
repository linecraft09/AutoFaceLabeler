import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from annotators.whisper_annotator import WhisperAnnotator
from core.models.annotation_models import Transcription


def make_video_with_audio(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return path


def make_video_without_audio(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=5",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return path


def make_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> Path:
    samples = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return path


class FakeInfo:
    language = "en"


class FakeSegment:
    start = 0.0
    end = 1.0
    text = " hello"
    avg_logprob = -0.1
    no_speech_prob = 0.05


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def transcribe(self, audio_path, **kwargs):
        return iter([FakeSegment()]), FakeInfo()


def test_audio_extraction_success(tmp_path):
    video = make_video_with_audio(tmp_path / "with_audio.mp4")
    annotator = WhisperAnnotator({"model_size": "tiny", "device": "cpu", "compute_type": "int8"})

    wav_path = annotator.extract_audio(str(video), tmp_path)

    assert wav_path is not None
    assert Path(wav_path).exists()
    assert Path(wav_path).stat().st_size > 44


def test_audio_extraction_no_audio(tmp_path):
    video = make_video_without_audio(tmp_path / "no_audio.mp4")
    annotator = WhisperAnnotator({})

    assert annotator.extract_audio(str(video), tmp_path) is None


def test_model_load():
    from unittest.mock import patch

    with patch("annotators.whisper_annotator.WhisperModel", FakeWhisperModel):
        annotator = WhisperAnnotator({"model_size": "tiny", "device": "cpu", "compute_type": "int8"})
        annotator.load()

    assert annotator.model.args == ("tiny",)
    assert annotator.model.kwargs == {"device": "cpu", "compute_type": "int8"}
    assert annotator.model is not None


def test_transcribe_returns_segments(monkeypatch, tmp_path):
    monkeypatch.setattr("annotators.whisper_annotator.WhisperModel", FakeWhisperModel)
    audio = make_wav(tmp_path / "speech.wav", np.ones(16000, dtype=np.int16) * 1000)
    annotator = WhisperAnnotator({"model_size": "tiny", "device": "cpu", "compute_type": "int8"})

    result = annotator.transcribe_audio(str(audio))

    assert result.segments


def test_segment_structure(monkeypatch, tmp_path):
    monkeypatch.setattr("annotators.whisper_annotator.WhisperModel", FakeWhisperModel)
    audio = make_wav(tmp_path / "speech.wav", np.ones(16000, dtype=np.int16) * 1000)

    segment = WhisperAnnotator({}).transcribe_audio(str(audio)).segments[0]

    assert {"start", "end", "text", "confidence"} <= set(segment)


def test_language_detection(monkeypatch, tmp_path):
    monkeypatch.setattr("annotators.whisper_annotator.WhisperModel", FakeWhisperModel)
    audio = make_wav(tmp_path / "speech.wav", np.ones(16000, dtype=np.int16) * 1000)

    assert WhisperAnnotator({}).transcribe_audio(str(audio)).language == "en"


def test_empty_audio(tmp_path):
    audio = make_wav(tmp_path / "silence.wav", np.zeros(16000, dtype=np.int16))

    result = WhisperAnnotator({}).transcribe_audio(str(audio))

    assert result.text == ""
    assert result.segments == []


def test_annotate_returns_transcription(monkeypatch, tmp_path):
    monkeypatch.setattr("annotators.whisper_annotator.WhisperModel", FakeWhisperModel)
    video = make_video_with_audio(tmp_path / "with_audio.mp4")
    annotator = WhisperAnnotator({"model_size": "tiny", "device": "cpu", "compute_type": "int8"})

    result = annotator.annotate(str(video))

    assert isinstance(result, Transcription)
    assert result.text == "hello"


def test_label_name():
    assert WhisperAnnotator({}).label_name == "transcription"


def test_annotate_with_audio_file(monkeypatch, tmp_path):
    monkeypatch.setattr("annotators.whisper_annotator.WhisperModel", FakeWhisperModel)
    video = make_video_with_audio(tmp_path / "synthetic_audio.mp4")

    result = WhisperAnnotator({"model_size": "tiny", "device": "cpu", "compute_type": "int8"}).annotate(str(video))

    assert result.language == "en"
    assert result.segments[0]["text"] == "hello"
