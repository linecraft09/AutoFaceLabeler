from pathlib import Path

import cv2
import numpy as np
import pytest

from core.models.annotation_models import ClarityScore
from annotators.dwpose_annotator import DWPoseAnnotator
from annotators.models.dwpose_wrapper import DWPoseWrapper


MODEL_PATH = Path("models/dwpose/dw-ll_ucoco_384.onnx")


@pytest.fixture
def synthetic_face_frame():
    frame = np.full((240, 320, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (160, 95), 42, (210, 210, 210), -1)
    cv2.circle(frame, (146, 88), 4, (20, 20, 20), -1)
    cv2.circle(frame, (174, 88), 4, (20, 20, 20), -1)
    cv2.ellipse(frame, (160, 107), (18, 7), 0, 0, 180, (30, 30, 30), 2)
    return frame


def test_model_load():
    wrapper = DWPoseWrapper(model_path=MODEL_PATH, providers=["CPUExecutionProvider"])
    wrapper.load()

    assert wrapper.session is not None
    assert wrapper.input_name


def test_keypoints_shape(synthetic_face_frame):
    wrapper = DWPoseWrapper(model_path=MODEL_PATH, providers=["CPUExecutionProvider"])
    keypoints = wrapper.predict(synthetic_face_frame)

    assert keypoints.shape == (17, 3)
    assert np.isfinite(keypoints).all()


def test_face_keypoints_extraction():
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:5] = np.array(
        [
            [50, 40, 0.9],
            [42, 35, 0.8],
            [58, 35, 0.85],
            [30, 42, 0.7],
            [70, 42, 0.75],
        ],
        dtype=np.float32,
    )

    face_keypoints = DWPoseAnnotator.extract_face_keypoints(keypoints, confidence_threshold=0.5)

    assert face_keypoints.shape == (5, 3)
    assert face_keypoints[:, 2].min() >= 0.7


def test_face_bbox_from_keypoints():
    face_keypoints = np.array(
        [
            [50, 40, 0.9],
            [42, 35, 0.8],
            [58, 35, 0.85],
            [30, 42, 0.7],
            [70, 42, 0.75],
        ],
        dtype=np.float32,
    )

    bbox = DWPoseAnnotator.face_bbox_from_keypoints(face_keypoints, image_shape=(100, 100, 3), crop_padding=10)

    assert bbox == (20, 25, 80, 52)


def test_face_bbox_empty():
    assert DWPoseAnnotator.face_bbox_from_keypoints(np.empty((0, 3)), (100, 100, 3)) is None


def test_crop_face_region(synthetic_face_frame):
    bbox = (120, 50, 200, 150)
    crop = DWPoseAnnotator.crop_face_region(synthetic_face_frame, bbox)

    assert crop.shape == (100, 80, 3)
    assert crop.size > 0


def test_laplacian_on_cropped_face(synthetic_face_frame):
    crop = DWPoseAnnotator.crop_face_region(synthetic_face_frame, (120, 50, 200, 150))
    score = DWPoseAnnotator.compute_face_clarity(crop)

    assert score > 0


def test_laplacian_aggregate_stats():
    result = DWPoseAnnotator.aggregate_clarity([10.0, 20.0, 30.0], total_frames=4)

    assert result.mean_clarity == pytest.approx(20.0)
    assert result.median_clarity == pytest.approx(20.0)
    assert result.std_clarity == pytest.approx(np.std([10.0, 20.0, 30.0]))
    assert result.min_clarity == pytest.approx(10.0)
    assert result.max_clarity == pytest.approx(30.0)
    assert result.face_detected_ratio == pytest.approx(0.75)


def test_laplacian_no_face_detected():
    result = DWPoseAnnotator.aggregate_clarity([], total_frames=5)

    assert result.face_detected_ratio == 0
    assert result.mean_clarity == 0
    assert result.per_frame == []


def test_annotate_returns_clarity_score(monkeypatch, tmp_path, synthetic_face_frame):
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (synthetic_face_frame.shape[1], synthetic_face_frame.shape[0]),
    )
    for _ in range(3):
        writer.write(synthetic_face_frame)
    writer.release()

    face_keypoints = np.array(
        [[160, 95, 0.9], [146, 88, 0.9], [174, 88, 0.9], [125, 96, 0.9], [195, 96, 0.9]],
        dtype=np.float32,
    )
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:5] = face_keypoints
    monkeypatch.setattr(DWPoseWrapper, "predict", lambda self, frame: keypoints)

    annotator = DWPoseAnnotator({"model_path": str(MODEL_PATH), "sample_frames": 3})
    result = annotator.annotate(str(video_path))

    assert isinstance(result, ClarityScore)
    assert result.face_detected_ratio == pytest.approx(1.0)
    assert len(result.per_frame) == 3


def test_label_name():
    assert DWPoseAnnotator({}).label_name == "clarity"


def test_annotate_with_sample_image(synthetic_face_frame):
    annotator = DWPoseAnnotator({"model_path": str(MODEL_PATH)})
    result = annotator.analyze_frame(synthetic_face_frame)

    assert isinstance(result, float) or result is None
