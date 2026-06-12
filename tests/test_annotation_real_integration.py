"""Real-model integration test — zero mocks.

Uses DeepFace (TF CPU), DWPose (ONNX CPU), Whisper (tiny CPU),
and QwenVL (DashScope qwen3-vl-plus).

Requires: DASHSCOPE_API_KEY in .env
"""

import json
import os
from pathlib import Path

import pytest

from annotators.deepface_annotator import DeepFaceAnnotator
from annotators.dwpose_annotator import DWPoseAnnotator
from annotators.orchestrator import AnnotationOrchestrator
from annotators.qwen_vl_annotator import QwenVLAnnotator
from annotators.whisper_annotator import WhisperAnnotator
from core.storage.annotation_store import AnnotationStore
from core.storage.video_store import VideoStore

# ── Test configuration ──────────────────────────────────────

QWEN_VL_MODEL = "qwen3-vl-plus"
QWEN_VL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

VIDEOS = [
    {
        "video_id": "JPUbSvsxjsY",
        "path": Path("test_data/qualified/JPUbSvsxjsY_qualified.mkv"),
        "duration_seconds": 60,
    },
    {
        "video_id": "R2H34vS4p1s",
        "path": Path("test_data/qualified/R2H34vS4p1s_qualified.mkv"),
        "duration_seconds": 65,
    },
]

VALID_RACES = {"asian", "indian", "black", "white", "middle eastern", "latino hispanic"}


@pytest.mark.integration
class TestRealAnnotationIntegration:
    """End-to-end annotation pipeline — real models only."""

    def setup_method(self):
        if not os.environ.get("DASHSCOPE_API_KEY"):
            pytest.skip("DASHSCOPE_API_KEY not set")
        missing = [str(v["path"]) for v in VIDEOS if not v["path"].exists()]
        if missing:
            pytest.skip(f"integration test videos missing: {missing}")

    @pytest.fixture
    def stores(self, tmp_path):
        video_store = VideoStore(str(tmp_path / "videos.db"))
        annotation_store = AnnotationStore(":memory:")
        yield video_store, annotation_store
        annotation_store.close()

    @pytest.fixture(autouse=True)
    def _tmp_tempdir(self, tmp_path, monkeypatch):
        import tempfile
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
        yield
        tempfile.tempdir = None

    # ── helpers ──────────────────────────────────────────

    def _register_videos(self, video_store):
        with video_store._connect() as conn:
            for v in VIDEOS:
                conn.execute(
                    """INSERT INTO videos (
                        video_id, url, platform, title, duration_seconds,
                        file_path, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'v2_passed', CURRENT_TIMESTAMP)
                    ON CONFLICT(video_id, platform) DO UPDATE SET
                        file_path = excluded.file_path,
                        status = excluded.status,
                        updated_at = CURRENT_TIMESTAMP""",
                    (
                        v["video_id"],
                        f"https://www.youtube.com/watch?v={v['video_id']}",
                        "youtube",
                        f"Real integration {v['video_id']}",
                        v["duration_seconds"],
                        str(v["path"].resolve()),
                    ),
                )

    def _create_annotators(self):
        return [
            DeepFaceAnnotator({"sample_frames": 5}),
            DWPoseAnnotator({
                "model_path": "models/dwpose/dw-ll_ucoco_384.onnx",
                "providers": ["CPUExecutionProvider"],
                "sample_frames": 5,
            }),
            WhisperAnnotator({
                "model_size": "tiny",
                "device": "cpu",
                "compute_type": "int8",
            }),
            QwenVLAnnotator({
                "model": QWEN_VL_MODEL,
                "base_url": QWEN_VL_BASE_URL,
                "sample_frames": 8,
                "min_frames": 4,
                "temperature": 0.2,
            }),
        ]

    def _assert_valid_real_annotation(self, annotation_store, video_id):
        full = annotation_store.get_full_annotation(video_id, "youtube")
        assert full is not None, f"no annotation for {video_id}"
        status = full["annotation"]["status"]
        assert status == "completed", f"status={status}, expected completed"

        # Label 1 – DeepFace
        facial = full["facial_features"]
        assert facial is not None
        assert 0 <= facial["age"] <= 120
        assert facial["race"] in VALID_RACES
        assert facial["hair_color"] == "unknown"

        # Label 2 – QwenVL facial motion
        motion = full["facial_motion"]
        assert motion is not None
        assert isinstance(motion["description"], str) and len(motion["description"]) > 0
        assert isinstance(motion["key_movements"], list)

        # Label 3 – DWPose clarity
        clarity = full["clarity"]
        assert clarity is not None
        assert clarity["mean_clarity"] >= 0
        assert 0.0 <= clarity["face_detected_ratio"] <= 1.0

        # Label 4 – QwenVL expression
        expression = full["expression"]
        assert expression is not None
        assert 1 <= expression["intensity"] <= 5
        assert isinstance(expression["rationale"], str)

        # Transcription – Whisper
        transcription = full["transcription"]
        assert transcription is not None
        assert isinstance(transcription["text"], str)

    def _format_annotation_report(self, annotation_store, video_id):
        """Return a human-readable report string for one video."""
        full = annotation_store.get_full_annotation(video_id, "youtube")
        facial = full["facial_features"]
        clarity = full["clarity"]
        motion = full["facial_motion"]
        expression = full["expression"]
        transcription = full["transcription"]

        s = f"━━━ {video_id} ━━━\n"
        s += f"🟢 Status: {full['annotation']['status']}\n\n"

        s += "📊 Label 1 — DeepFace (age / race)\n"
        s += f"  Age: {facial['age']}\n"
        s += f"  Race: {facial['race']}\n"
        s += f"  Confidence: {json.dumps(facial.get('confidence', {}), ensure_ascii=False)}\n\n"

        s += "📝 Label 2 — QwenVL (facial motion)\n"
        s += f"  Description: {motion['description']}\n"
        s += f"  Key movements: {json.dumps(motion['key_movements'], ensure_ascii=False)}\n"
        s += f"  Duration: {motion.get('duration_category', 'N/A')}\n\n"

        s += "🔍 Label 3 — DWPose (clarity)\n"
        s += f"  Mean clarity: {clarity['mean_clarity']:.2f}\n"
        s += f"  Face ratio: {clarity['face_detected_ratio']:.2%}\n"
        s += f"  Per-frame: {json.dumps(clarity.get('per_frame', []))}\n\n"

        s += "🎭 Label 4 — QwenVL (expression intensity)\n"
        s += f"  Intensity: {expression['intensity']}/5\n"
        s += f"  Rationale: {expression['rationale']}\n"
        s += f"  Dominant: {json.dumps(expression['dominant_expressions'], ensure_ascii=False)}\n\n"

        s += "🎙️ Whisper — Transcription\n"
        s += f"  Language: {transcription.get('language', '?')}\n"
        s += f"  Segments: {len(transcription.get('segments', []))}\n"
        s += f"  Text (first 300 chars): {transcription['text'][:300]}...\n"

        return s

    # ── test ────────────────────────────────────────────

    def test_real_full_pipeline_two_videos(self, stores, capsys):
        video_store, annotation_store = stores
        self._register_videos(video_store)

        orchestrator = AnnotationOrchestrator(
            video_store,
            annotation_store,
            self._create_annotators(),
            enabled_labels=[
                "facial_features",
                "facial_analysis",
                "clarity",
                "transcription",
            ],
        )

        processed = orchestrator.run(target_count=2)
        assert processed == 2, f"expected 2, got {processed}"

        # Print reports
        print("\n" + "=" * 60)
        print("    AFL MILESTONE 2 — REAL MODEL INTEGRATION TEST REPORT")
        print("=" * 60)
        print(f"  Model: QwenVL={QWEN_VL_MODEL}")
        print(f"  Base URL: {QWEN_VL_BASE_URL}")
        print(f"  Videos: {len(VIDEOS)}")
        print("=" * 60 + "\n")

        for v in VIDEOS:
            self._assert_valid_real_annotation(annotation_store, v["video_id"])
            print(self._format_annotation_report(annotation_store, v["video_id"]))

        print("=" * 60)
        print("    ALL ANNOTATIONS PASSED ✅")
        print("=" * 60)
