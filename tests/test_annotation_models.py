import pytest
from pydantic import ValidationError

from core.models.annotation_models import (
    AnnotationResult,
    ClarityScore,
    ExpressionIntensity,
    FacialFeatures,
    FacialMotion,
    Transcription,
)


def test_facial_features_valid():
    features = FacialFeatures(
        age=28,
        race="asian",
        hair_color="black",
        confidence={"age": 0.91, "race": {"asian": 0.88}},
    )

    assert features.age == 28
    assert isinstance(features.age, int)
    assert features.race == "asian"
    assert isinstance(features.confidence, dict)


def test_facial_features_serialization():
    features = FacialFeatures(
        age=28,
        race="asian",
        hair_color="black",
        confidence={"age": 0.91},
    )

    assert features.model_dump() == {
        "age": 28,
        "race": "asian",
        "hair_color": "black",
        "confidence": {"age": 0.91},
    }


def test_facial_features_missing_field():
    with pytest.raises(ValidationError):
        FacialFeatures(age=28, race="asian", confidence={})


def test_facial_motion_valid():
    motion = FacialMotion(
        description="Subject smiles and blinks.",
        key_movements=["smile", "blink"],
        duration_category="moderate",
    )

    assert motion.description.startswith("Subject")
    assert motion.key_movements == ["smile", "blink"]
    assert motion.duration_category == "moderate"


def test_facial_motion_key_movements_type():
    with pytest.raises(ValidationError):
        FacialMotion(
            description="Subject smiles.",
            key_movements="smile",
            duration_category="moderate",
        )


def test_clarity_score_valid():
    clarity = ClarityScore(
        mean_clarity=120.123456,
        median_clarity=118.5,
        std_clarity=12.75,
        min_clarity=90.0,
        max_clarity=155.25,
        face_detected_ratio=0.95,
        per_frame=[100.1, 120.2],
    )

    assert clarity.mean_clarity == pytest.approx(120.123456)
    assert clarity.face_detected_ratio == pytest.approx(0.95)


def test_clarity_score_per_frame_optional():
    without_frames = ClarityScore(
        mean_clarity=120.0,
        median_clarity=118.0,
        std_clarity=12.0,
        min_clarity=90.0,
        max_clarity=155.0,
        face_detected_ratio=0.95,
    )
    empty_frames = ClarityScore(
        mean_clarity=120.0,
        median_clarity=118.0,
        std_clarity=12.0,
        min_clarity=90.0,
        max_clarity=155.0,
        face_detected_ratio=0.95,
        per_frame=[],
    )

    assert without_frames.per_frame is None
    assert empty_frames.per_frame == []


def test_expression_intensity_valid():
    expression = ExpressionIntensity(
        intensity=3,
        rationale="Moderate smile with visible cheek movement.",
        dominant_expressions=["smile", "neutral"],
    )

    assert expression.intensity == 3
    assert expression.dominant_expressions == ["smile", "neutral"]


def test_expression_intensity_out_of_range():
    for value in (0, 6):
        with pytest.raises(ValidationError):
            ExpressionIntensity(
                intensity=value,
                rationale="Out of range.",
                dominant_expressions=["neutral"],
            )


def test_expression_intensity_boundary():
    low = ExpressionIntensity(intensity=1, rationale="Minimal.", dominant_expressions=["neutral"])
    high = ExpressionIntensity(intensity=5, rationale="Strong.", dominant_expressions=["surprise"])

    assert low.intensity == 1
    assert high.intensity == 5


def test_transcription_valid():
    transcription = Transcription(
        text="Hello world.",
        language="en",
        segments=[{"start": 0.0, "end": 1.2, "text": "Hello world.", "confidence": 0.98}],
    )

    assert transcription.text == "Hello world."
    assert transcription.language == "en"
    assert len(transcription.segments) == 1


def test_transcription_segments_structure():
    transcription = Transcription(
        text="Hello world.",
        language="en",
        segments=[{"start": 0.0, "end": 1.2, "text": "Hello world.", "confidence": 0.98}],
    )

    segment = transcription.segments[0]
    assert {"start", "end", "text", "confidence"} <= set(segment)


def test_annotation_result_composition():
    result = AnnotationResult(
        video_id="vid-1",
        platform="youtube",
        facial_features=FacialFeatures(age=28, race="asian", hair_color="black", confidence={}),
        facial_motion=FacialMotion(
            description="Smiles.",
            key_movements=["smile"],
            duration_category="short",
        ),
        clarity=ClarityScore(
            mean_clarity=120.0,
            median_clarity=118.0,
            std_clarity=12.0,
            min_clarity=90.0,
            max_clarity=155.0,
            face_detected_ratio=0.95,
            per_frame=[120.0],
        ),
        expression=ExpressionIntensity(
            intensity=4,
            rationale="Clear expression.",
            dominant_expressions=["smile"],
        ),
        transcription=Transcription(text="Hi.", language="en", segments=[]),
    )

    assert result.video_id == "vid-1"
    assert result.facial_features.age == 28
    assert result.transcription.text == "Hi."


def test_annotation_result_partial():
    result = AnnotationResult(video_id="vid-1", platform="youtube", facial_features=None)

    assert result.facial_features is None
    assert result.facial_motion is None
    assert result.clarity is None
    assert result.expression is None
    assert result.transcription is None


def test_annotation_result_json_roundtrip():
    result = AnnotationResult(
        video_id="vid-1",
        platform="youtube",
        facial_features=FacialFeatures(age=28, race="asian", hair_color="black", confidence={}),
        expression=ExpressionIntensity(
            intensity=5,
            rationale="Very expressive.",
            dominant_expressions=["surprise"],
        ),
    )

    restored = AnnotationResult.model_validate_json(result.model_dump_json())

    assert restored == result
