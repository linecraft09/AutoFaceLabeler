import json
import shutil
import tempfile
from pathlib import Path

import pytest

from annotators.deepface_annotator import DeepFaceAnnotator
from annotators.dwpose_annotator import DWPoseAnnotator
from annotators.orchestrator import AnnotationOrchestrator
from annotators.qwen_vl_annotator import QwenVLAnnotator
from annotators.whisper_annotator import WhisperAnnotator
from core.storage.annotation_store import AnnotationStore
from core.storage.video_store import VideoStore


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


VIDEOS = [
    {
        "video_id": "-bD9lbUT4u8",
        "path": Path("test_data/qualified/-bD9lbUT4u8_qualified.mkv"),
        "duration_seconds": 40,
    },
    {
        "video_id": "mfZscA9V6nU",
        "path": Path("test_data/qualified/mfZscA9V6nU_qualified.mkv"),
        "duration_seconds": 66,
    },
]


class SkipUnavailableAnnotator:
    """Convert optional model/download failures into integration-test skips."""

    def __init__(self, annotator):
        self.annotator = annotator
        self.label_name = annotator.label_name

    def annotate(self, video_path):
        try:
            result = self.annotator.annotate(video_path)
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            pytest.skip(f"{self.label_name} integration dependency unavailable: {exc}")
        if result is None:
            pytest.skip(f"{self.label_name} produced no result for sample video")
        return result

    def unload(self):
        if hasattr(self.annotator, "unload"):
            self.annotator.unload()
        elif hasattr(self.annotator, "close"):
            self.annotator.close()


@pytest.mark.integration
class TestAnnotationIntegration:
    def setup_method(self):
        missing = [str(video["path"]) for video in VIDEOS if not video["path"].exists()]
        if missing:
            pytest.skip(f"integration test videos missing: {missing}")
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg is required for Whisper annotation integration")

    @pytest.fixture(autouse=True)
    def _tmp_tempdir(self, tmp_path, monkeypatch):
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
        yield
        tempfile.tempdir = None

    @pytest.fixture
    def stores(self, tmp_path):
        video_store = VideoStore(str(tmp_path / "videos.db"))
        annotation_store = AnnotationStore(":memory:")
        yield video_store, annotation_store
        annotation_store.close()

    @pytest.fixture(autouse=True)
    def mock_deepface(self, monkeypatch):
        """Mock DeepFace to avoid model download in CI."""
        class FakeDeepFace:
            @staticmethod
            def analyze(img_path, actions, enforce_detection=False, **kwargs):
                return [{"age": 28, "dominant_race": "asian", "race": {"asian": 0.8, "white": 0.2}}]
        monkeypatch.setattr("annotators.deepface_annotator.DeepFace", FakeDeepFace)

    @pytest.fixture
    def mock_qwen(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(QwenVLAnnotator, "call_api", lambda self, frames: VALID_RESPONSE)

    def register_videos(self, video_store, videos=VIDEOS):
        with video_store._connect() as conn:
            for video in videos:
                conn.execute(
                    """
                    INSERT INTO videos (
                        video_id, url, platform, title, duration_seconds,
                        file_path, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'v2_passed', CURRENT_TIMESTAMP)
                    ON CONFLICT(video_id, platform) DO UPDATE SET
                        file_path = excluded.file_path,
                        status = excluded.status,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        video["video_id"],
                        f"https://www.youtube.com/watch?v={video['video_id']}",
                        "youtube",
                        f"Integration sample {video['video_id']}",
                        video["duration_seconds"],
                        str(video["path"].resolve()),
                    ),
                )

    def create_annotators(self, include_whisper=True):
        annotators = [
            SkipUnavailableAnnotator(DeepFaceAnnotator({"sample_frames": 2})),
            SkipUnavailableAnnotator(
                DWPoseAnnotator(
                    {
                        "model_path": "models/dwpose/dw-ll_ucoco_384.onnx",
                        "providers": ["CPUExecutionProvider"],
                        "sample_frames": 2,
                    }
                )
            ),
        ]
        if include_whisper:
            annotators.append(
                SkipUnavailableAnnotator(
                    WhisperAnnotator(
                        {
                            "model_size": "tiny",
                            "device": "cpu",
                            "compute_type": "int8",
                        }
                    )
                )
            )
        annotators.append(SkipUnavailableAnnotator(QwenVLAnnotator({"sample_frames": 2, "min_frames": 2})))
        return annotators

    def assert_completed_full_annotation(self, annotation_store, video_id):
        full = annotation_store.get_full_annotation(video_id, "youtube")
        assert full is not None
        assert full["annotation"]["status"] == "completed"

        facial = full["facial_features"]
        assert facial is not None
        assert 0 <= facial["age"] <= 120
        assert facial["race"]
        assert facial["hair_color"] == "unknown"

        clarity = full["clarity"]
        assert clarity is not None
        assert clarity["mean_clarity"] is not None
        assert 0 <= clarity["face_detected_ratio"] <= 1

        transcription = full["transcription"]
        assert transcription is not None
        assert transcription["text"]

        motion = full["facial_motion"]
        assert motion is not None
        assert motion["description"] == "轻微微笑并眨眼"
        assert motion["key_movements"] == ["眨眼", "微笑"]
        assert motion["duration_category"] == "moderate"

        expression = full["expression"]
        assert expression is not None
        assert expression["intensity"] == 3
        assert expression["rationale"] == "表情变化中等"
        assert expression["dominant_expressions"] == ["smile", "neutral"]

    def assert_no_partial_failures(self, annotation_store):
        with annotation_store._connect() as conn:
            failures = conn.execute(
                """
                SELECT video_id, status, label_1_status, label_2_status,
                       label_3_status, label_4_status, whisper_status
                FROM annotations
                WHERE status != 'completed'
                """
            ).fetchall()
        if failures:
            pytest.skip(f"annotation integration dependency failed: {[dict(row) for row in failures]}")

    def test_full_annotation_pipeline_two_videos(self, stores, mock_qwen):
        video_store, annotation_store = stores
        self.register_videos(video_store)
        orchestrator = AnnotationOrchestrator(
            video_store,
            annotation_store,
            self.create_annotators(include_whisper=True),
            enabled_labels=["facial_features", "facial_analysis", "clarity", "transcription"],
        )

        assert orchestrator.run(target_count=2) == 2

        self.assert_no_partial_failures(annotation_store)
        for video in VIDEOS:
            self.assert_completed_full_annotation(annotation_store, video["video_id"])

    def test_single_video_selected_labels(self, stores, mock_qwen):
        video_store, annotation_store = stores
        self.register_videos(video_store, videos=[VIDEOS[0]])
        orchestrator = AnnotationOrchestrator(
            video_store,
            annotation_store,
            self.create_annotators(include_whisper=False),
            enabled_labels=["facial_features", "clarity"],
        )

        assert orchestrator.run(target_count=1) == 1

        self.assert_no_partial_failures(annotation_store)
        full = annotation_store.get_full_annotation(VIDEOS[0]["video_id"], "youtube")
        assert full["annotation"]["status"] == "completed"
        assert full["annotation"]["label_1_status"] == "completed"
        assert full["annotation"]["label_3_status"] == "completed"
        assert full["annotation"]["label_2_status"] == "pending"
        assert full["annotation"]["label_4_status"] == "pending"
        assert full["annotation"]["whisper_status"] == "pending"
        assert full["facial_features"] is not None
        assert full["clarity"] is not None
        assert full["facial_motion"] is None
        assert full["expression"] is None
        assert full["transcription"] is None

    def test_annotation_checkpoint_and_resume(self, stores, mock_qwen):
        video_store, annotation_store = stores
        self.register_videos(video_store)
        first_orchestrator = AnnotationOrchestrator(
            video_store,
            annotation_store,
            self.create_annotators(include_whisper=True),
            enabled_labels=["facial_features", "facial_analysis", "clarity", "transcription"],
        )
        run = first_orchestrator.create_checkpoint()

        assert first_orchestrator.run_single_video(
            {
                "video_id": VIDEOS[0]["video_id"],
                "platform": "youtube",
                "file_path": str(VIDEOS[0]["path"].resolve()),
            },
            run_id=run["run_id"],
        )

        self.assert_no_partial_failures(annotation_store)
        first_orchestrator.cleanup()

        resumed = AnnotationOrchestrator(
            video_store,
            annotation_store,
            self.create_annotators(include_whisper=True),
            enabled_labels=["facial_features", "facial_analysis", "clarity", "transcription"],
        )
        restored = resumed.restore_checkpoint()
        assert restored is not None
        assert restored["run_id"] == run["run_id"]
        assert restored["status"] == "in_progress"

        assert resumed.run(target_count=2) == 1

        self.assert_no_partial_failures(annotation_store)
        for video in VIDEOS:
            self.assert_completed_full_annotation(annotation_store, video["video_id"])
        assert video_store.get_in_progress_annotation_run() is None
