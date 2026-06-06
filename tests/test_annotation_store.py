import pytest

from core.models.annotation_models import (
    ClarityScore,
    ExpressionIntensity,
    FacialFeatures,
    FacialMotion,
    Transcription,
)
from core.storage.annotation_store import AnnotationStore


@pytest.fixture
def store():
    annotation_store = AnnotationStore(":memory:")
    yield annotation_store
    annotation_store.close()


def test_init_creates_tables(store):
    expected = {
        "annotations",
        "label_1_facial_features",
        "label_2_facial_motion",
        "label_3_clarity",
        "label_4_expression",
        "transcriptions",
    }
    tables = {
        row["name"]
        for row in store._connect().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    assert expected <= tables


def test_create_annotation(store):
    store.create_annotation("vid-1", "youtube")

    row = store.get_annotation("vid-1", "youtube")
    assert row["video_id"] == "vid-1"
    assert row["platform"] == "youtube"
    assert row["status"] == "pending"


def test_create_annotation_duplicate(store):
    store.create_annotation("vid-1", "youtube")
    store.update_annotation_status("vid-1", "youtube", "in_progress")
    store.create_annotation("vid-1", "youtube", status="pending")

    row = store.get_annotation("vid-1", "youtube")
    assert row["status"] == "pending"
    assert store._connect().execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 1


def test_get_annotation(store):
    store.create_annotation("vid-1", "youtube", status="in_progress")

    row = store.get_annotation("vid-1", "youtube")
    assert row["status"] == "in_progress"


def test_get_annotation_not_found(store):
    assert store.get_annotation("missing", "youtube") is None


def test_update_annotation_status(store):
    store.create_annotation("vid-1", "youtube")
    store.update_annotation_status("vid-1", "youtube", "completed")

    assert store.get_annotation("vid-1", "youtube")["status"] == "completed"


def test_update_label_status(store):
    store.create_annotation("vid-1", "youtube")
    store.update_label_status("vid-1", "youtube", "label_1", "completed")

    assert store.get_annotation("vid-1", "youtube")["label_1_status"] == "completed"


def test_save_facial_features(store):
    store.create_annotation("vid-1", "youtube")
    features = FacialFeatures(
        age=29,
        race="asian",
        hair_color="black",
        confidence={"age": 0.91},
    )

    store.save_facial_features("vid-1", "youtube", features, raw_response={"model": "deepface"})

    row = store._connect().execute(
        "SELECT * FROM label_1_facial_features WHERE video_id = ? AND platform = ?",
        ("vid-1", "youtube"),
    ).fetchone()
    assert row["age"] == 29
    assert row["raw_response"] == '{"model": "deepface"}'


def test_get_facial_features(store):
    store.create_annotation("vid-1", "youtube")
    store.save_facial_features(
        "vid-1",
        "youtube",
        FacialFeatures(age=29, race="asian", hair_color="black", confidence={"age": 0.91}),
    )

    data = store.get_facial_features("vid-1", "youtube")
    assert data["age"] == 29
    assert data["confidence"] == {"age": 0.91}


def test_save_facial_motion(store):
    store.create_annotation("vid-1", "youtube")
    motion = FacialMotion(
        description="Subject smiles.",
        key_movements=["smile"],
        duration_category="short",
    )

    store.save_facial_motion("vid-1", "youtube", motion)

    row = store._connect().execute(
        "SELECT * FROM label_2_facial_motion WHERE video_id = ? AND platform = ?",
        ("vid-1", "youtube"),
    ).fetchone()
    assert row["description"] == "Subject smiles."


def test_get_facial_motion(store):
    store.create_annotation("vid-1", "youtube")
    store.save_facial_motion(
        "vid-1",
        "youtube",
        FacialMotion(
            description="Subject smiles.",
            key_movements=["smile"],
            duration_category="short",
        ),
    )

    data = store.get_facial_motion("vid-1", "youtube")
    assert data["key_movements"] == ["smile"]


def test_save_clarity(store):
    store.create_annotation("vid-1", "youtube")
    clarity = ClarityScore(
        mean_clarity=120.0,
        median_clarity=118.0,
        std_clarity=12.0,
        min_clarity=90.0,
        max_clarity=155.0,
        face_detected_ratio=0.95,
        per_frame=[100.0, 120.0],
    )

    store.save_clarity("vid-1", "youtube", clarity)

    row = store._connect().execute(
        "SELECT * FROM label_3_clarity WHERE video_id = ? AND platform = ?",
        ("vid-1", "youtube"),
    ).fetchone()
    assert row["mean_clarity"] == 120.0


def test_get_clarity(store):
    store.create_annotation("vid-1", "youtube")
    store.save_clarity(
        "vid-1",
        "youtube",
        ClarityScore(
            mean_clarity=120.0,
            median_clarity=118.0,
            std_clarity=12.0,
            min_clarity=90.0,
            max_clarity=155.0,
            face_detected_ratio=0.95,
            per_frame=[100.0, 120.0],
        ),
    )

    data = store.get_clarity("vid-1", "youtube")
    assert data["per_frame"] == [100.0, 120.0]


def test_save_expression(store):
    store.create_annotation("vid-1", "youtube")
    expression = ExpressionIntensity(
        intensity=4,
        rationale="Strong smile.",
        dominant_expressions=["smile"],
    )

    store.save_expression("vid-1", "youtube", expression)

    row = store._connect().execute(
        "SELECT * FROM label_4_expression WHERE video_id = ? AND platform = ?",
        ("vid-1", "youtube"),
    ).fetchone()
    assert row["intensity"] == 4


def test_get_expression(store):
    store.create_annotation("vid-1", "youtube")
    store.save_expression(
        "vid-1",
        "youtube",
        ExpressionIntensity(
            intensity=4,
            rationale="Strong smile.",
            dominant_expressions=["smile"],
        ),
    )

    data = store.get_expression("vid-1", "youtube")
    assert data["dominant_expressions"] == ["smile"]


def test_save_transcription(store):
    store.create_annotation("vid-1", "youtube")
    transcription = Transcription(
        text="Hello.",
        language="en",
        segments=[{"start": 0.0, "end": 1.0, "text": "Hello.", "confidence": 0.99}],
    )

    store.save_transcription("vid-1", "youtube", transcription)

    row = store._connect().execute(
        "SELECT * FROM transcriptions WHERE video_id = ? AND platform = ?",
        ("vid-1", "youtube"),
    ).fetchone()
    assert row["full_text"] == "Hello."


def test_get_transcription(store):
    store.create_annotation("vid-1", "youtube")
    store.save_transcription(
        "vid-1",
        "youtube",
        Transcription(
            text="Hello.",
            language="en",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello.", "confidence": 0.99}],
        ),
    )

    data = store.get_transcription("vid-1", "youtube")
    assert data["text"] == "Hello."
    assert data["segments"][0]["confidence"] == 0.99


def test_list_pending_annotations(store):
    store.create_annotation("pending-1", "youtube", status="pending")
    store.create_annotation("completed-1", "youtube", status="completed")

    rows = store.list_pending_annotations()
    assert [row["video_id"] for row in rows] == ["pending-1"]


def test_list_completed_annotations(store):
    store.create_annotation("pending-1", "youtube", status="pending")
    store.create_annotation("completed-1", "youtube", status="completed")

    rows = store.list_completed_annotations()
    assert [row["video_id"] for row in rows] == ["completed-1"]


def test_get_full_annotation(store):
    store.create_annotation("vid-1", "youtube", status="completed")
    store.save_facial_features(
        "vid-1",
        "youtube",
        FacialFeatures(age=29, race="asian", hair_color="black", confidence={"age": 0.91}),
    )
    store.save_facial_motion(
        "vid-1",
        "youtube",
        FacialMotion(description="Smiles.", key_movements=["smile"], duration_category="short"),
    )
    store.save_clarity(
        "vid-1",
        "youtube",
        ClarityScore(
            mean_clarity=120.0,
            median_clarity=118.0,
            std_clarity=12.0,
            min_clarity=90.0,
            max_clarity=155.0,
            face_detected_ratio=0.95,
            per_frame=[100.0],
        ),
    )
    store.save_expression(
        "vid-1",
        "youtube",
        ExpressionIntensity(intensity=4, rationale="Strong smile.", dominant_expressions=["smile"]),
    )
    store.save_transcription(
        "vid-1",
        "youtube",
        Transcription(text="Hello.", language="en", segments=[]),
    )

    data = store.get_full_annotation("vid-1", "youtube")
    assert data["annotation"]["status"] == "completed"
    assert data["facial_features"]["age"] == 29
    assert data["facial_motion"]["key_movements"] == ["smile"]
    assert data["clarity"]["per_frame"] == [100.0]
    assert data["expression"]["intensity"] == 4
    assert data["transcription"]["text"] == "Hello."
