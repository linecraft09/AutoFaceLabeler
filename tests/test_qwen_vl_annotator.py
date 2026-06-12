import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from annotators.qwen_vl_annotator import QwenVLAnnotator
from core.models.annotation_models import ExpressionIntensity, FacialMotion


def make_video(path: Path, frame_count: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (96, 64))
    for index in range(frame_count):
        frame = np.full((64, 96, 3), 40 + index * 5, dtype=np.uint8)
        cv2.putText(frame, str(index), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


VALID_RESPONSE = json.dumps(
    {
        "facial_motion": {
            "description": "轻微微笑并眨眼",
            "key_movements": ["眨眼", "微笑"],
            "duration_category": "moderate",
        },
        "expression_intensity": {
            "score": 3,
            "rationale": "表情变化中等",
            "dominant_expressions": ["smile", "neutral"],
        },
        "hair_color": "black",
    },
    ensure_ascii=False,
)


class FakeChoice:
    def __init__(self, content):
        self.message = type("Message", (), {"content": content})()


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, recorder):
        self.recorder = recorder

    def create(self, **kwargs):
        self.recorder.append(kwargs)
        return FakeResponse(VALID_RESPONSE)


class FakeOpenAI:
    calls = []

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = type("Chat", (), {"completions": FakeCompletions(self.calls)})()


def test_frame_sampling(tmp_path):
    video = make_video(tmp_path / "sample.mp4", frame_count=12)

    frames = QwenVLAnnotator({}).sample_frames(str(video), count=8)

    assert len(frames) == 8


def test_frame_sampling_short(tmp_path):
    video = make_video(tmp_path / "short.mp4", frame_count=3)

    frames = QwenVLAnnotator({}).sample_frames(str(video), count=8, min_frames=4)

    assert len(frames) == 4


def test_frame_to_base64():
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)

    encoded = QwenVLAnnotator.frame_to_base64(frame)

    assert isinstance(encoded, str)
    assert len(encoded) > 100


def test_build_message():
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8) for _ in range(2)]
    messages = QwenVLAnnotator({}).build_messages(frames)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][1]["type"] == "image_url"


def test_prompt_structure():
    prompt = QwenVLAnnotator.SYSTEM_PROMPT

    assert "facial_motion" in prompt
    assert "expression_intensity" in prompt
    assert "hair_color" in prompt
    assert "评分标准" in prompt


def test_parse_valid_response():
    parsed = QwenVLAnnotator.parse_response(VALID_RESPONSE)

    assert parsed["facial_motion"].key_movements == ["眨眼", "微笑"]
    assert parsed["expression"].intensity == 3
    assert parsed["hair_color"] == "black"


def test_parse_malformed_json():
    parsed = QwenVLAnnotator.parse_response("not json")

    assert parsed["parse_error"] is True
    assert parsed["expression"].intensity == 1


def test_parse_missing_fields():
    parsed = QwenVLAnnotator.parse_response('{"hair_color": "brown"}')

    assert parsed["facial_motion"].description == ""
    assert parsed["expression"].dominant_expressions == ["neutral"]
    assert parsed["hair_color"] == "brown"


def test_intensity_range():
    for value in range(1, 6):
        assert QwenVLAnnotator.validate_intensity(value) == value


def test_intensity_out_of_range():
    with pytest.raises(ValueError):
        QwenVLAnnotator.validate_intensity(6)


def test_api_call_mocked(monkeypatch):
    FakeOpenAI.calls = []
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("annotators.qwen_vl_annotator.OpenAI", FakeOpenAI)
    frames = [np.full((32, 32, 3), 128, dtype=np.uint8)]

    response = QwenVLAnnotator({"model": "qwen-vl-max"}).call_api(frames)

    assert response == VALID_RESPONSE
    assert FakeOpenAI.calls[0]["model"] == "qwen-vl-max"
    assert FakeOpenAI.calls[0]["messages"][0]["role"] == "system"


def test_retry_logic(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    annotator = QwenVLAnnotator({"max_retries": 3})
    mock = Mock(side_effect=[RuntimeError("temporary"), RuntimeError("temporary"), VALID_RESPONSE])
    monkeypatch.setattr(annotator, "_single_api_call", mock)

    assert annotator.call_api([np.zeros((16, 16, 3), dtype=np.uint8)]) == VALID_RESPONSE
    assert mock.call_count == 3


def test_annotate_returns_both_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(QwenVLAnnotator, "call_api", lambda self, frames: VALID_RESPONSE)
    video = make_video(tmp_path / "sample.mp4", frame_count=8)

    result = QwenVLAnnotator({}).annotate(str(video))

    assert isinstance(result["facial_motion"], FacialMotion)
    assert isinstance(result["expression"], ExpressionIntensity)


def test_label_name():
    assert QwenVLAnnotator({}).label_name == "facial_analysis"


def test_api_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    video = make_video(tmp_path / "sample.mp4", frame_count=5)

    result = QwenVLAnnotator({}).annotate(str(video))

    assert result["api_error"] == "missing_api_key"
    assert result["expression"].intensity == 1
