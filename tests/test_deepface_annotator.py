from pathlib import Path

import cv2
import numpy as np

from annotators.deepface_annotator import DeepFaceAnnotator
from core.models.annotation_models import FacialFeatures


VALID_RACES = {"asian", "indian", "black", "white", "middle eastern", "latino hispanic"}


def synthetic_face_frame():
    frame = np.full((180, 180, 3), 190, dtype=np.uint8)
    cv2.circle(frame, (90, 75), 45, (220, 200, 180), -1)
    cv2.circle(frame, (75, 68), 5, (30, 30, 30), -1)
    cv2.circle(frame, (105, 68), 5, (30, 30, 30), -1)
    cv2.ellipse(frame, (90, 91), (22, 8), 0, 0, 180, (40, 40, 40), 2)
    return frame


def make_video(path: Path, frame: np.ndarray, count: int = 5) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (frame.shape[1], frame.shape[0]),
    )
    for _ in range(count):
        writer.write(frame)
    writer.release()
    return path


class FakeDeepFace:
    @staticmethod
    def analyze(img_path, actions, enforce_detection=False):
        return {
            "age": 28,
            "dominant_race": "asian",
            "race": {"asian": 0.8, "white": 0.2},
        }


def test_deepface_import():
    import deepface

    assert deepface is not None


def test_analyze_single_image(monkeypatch):
    monkeypatch.setattr("annotators.deepface_annotator.DeepFace", FakeDeepFace)
    result = DeepFaceAnnotator({}).analyze_image(synthetic_face_frame())

    assert result["age"] == 28
    assert result["race"] == "asian"


def test_age_in_range(monkeypatch):
    monkeypatch.setattr("annotators.deepface_annotator.DeepFace", FakeDeepFace)
    result = DeepFaceAnnotator({}).analyze_image(synthetic_face_frame())

    assert 0 <= result["age"] <= 120


def test_race_is_valid(monkeypatch):
    monkeypatch.setattr("annotators.deepface_annotator.DeepFace", FakeDeepFace)
    result = DeepFaceAnnotator({}).analyze_image(synthetic_face_frame())

    assert result["race"] in VALID_RACES


def test_age_aggregation_median():
    features = DeepFaceAnnotator.aggregate_features(
        [
            {"age": 21, "race": "asian", "confidence": {}},
            {"age": 44, "race": "white", "confidence": {}},
            {"age": 29, "race": "asian", "confidence": {}},
        ]
    )

    assert features.age == 29


def test_race_aggregation_mode():
    features = DeepFaceAnnotator.aggregate_features(
        [
            {"age": 20, "race": "white", "confidence": {}},
            {"age": 30, "race": "asian", "confidence": {}},
            {"age": 40, "race": "asian", "confidence": {}},
        ]
    )

    assert features.race == "asian"


def test_confidence_structure(monkeypatch):
    monkeypatch.setattr("annotators.deepface_annotator.DeepFace", FakeDeepFace)
    result = DeepFaceAnnotator({}).analyze_image(synthetic_face_frame())

    assert "age" in result["confidence"]
    assert "race" in result["confidence"]


def test_no_face_image(monkeypatch):
    monkeypatch.setattr("annotators.deepface_annotator.DeepFace", FakeDeepFace)
    blank = np.zeros((128, 128, 3), dtype=np.uint8)

    assert DeepFaceAnnotator({}).analyze_image(blank) is None


def test_hair_color_default():
    features = DeepFaceAnnotator.aggregate_features([{"age": 28, "race": "asian", "confidence": {}}])

    assert features.hair_color == "unknown"


def test_annotate_returns_facial_features(monkeypatch, tmp_path):
    monkeypatch.setattr("annotators.deepface_annotator.DeepFace", FakeDeepFace)
    video = make_video(tmp_path / "face.mp4", synthetic_face_frame())

    result = DeepFaceAnnotator({"sample_frames": 5}).annotate(str(video))

    assert isinstance(result, FacialFeatures)
    assert result.age == 28


def test_label_name():
    assert DeepFaceAnnotator({}).label_name == "facial_features"


def test_annotate_with_sample_video(monkeypatch, tmp_path):
    monkeypatch.setattr("annotators.deepface_annotator.DeepFace", FakeDeepFace)
    video = make_video(tmp_path / "sample.mp4", synthetic_face_frame(), count=7)

    result = DeepFaceAnnotator({"sample_frames": 5}).annotate(str(video))

    assert result.race == "asian"
    assert result.hair_color == "unknown"
