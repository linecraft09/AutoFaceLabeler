import json
from pathlib import Path

import numpy as np
import pytest

from core.models.video_meta import VideoMeta
from validators import validator as validator_module
from validators.validator import V2ContentFilter


class FakeYOLO:
    def __init__(self, segments=None, total_single_secs=30.0):
        self.segments = segments or [(0, 30)]
        self.total_single_secs = total_single_secs
        self.calls = 0

    def detect_single_person_segments(self, video_path):
        self.calls += 1
        return self.segments, self.total_single_secs


class FakeFaceEmbedder:
    def __init__(self):
        self.added_embeddings = []

    def is_duplicate(self, embedding, threshold):
        return False

    def add_embedding(self, embedding):
        self.added_embeddings.append(embedding)


def make_video_meta(video_id: str, file_path: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://example.com/{video_id}",
        platform="youtube",
        title=f"title {video_id}",
        duration_seconds=60,
        resolution="720p",
        channel="channel",
        publish_date="2026-01-01",
        view_count=100,
        tags=[],
        search_term="term",
        extra={},
    )


def make_row(video_id: str, video_file: Path, status: str = "downloaded"):
    return {
        "video_id": video_id,
        "platform": "youtube",
        "file_path": str(video_file),
        "status": status,
        "search_term": "term",
    }


@pytest.fixture
def filter_factory(monkeypatch, video_store, tmp_path):
    def _make_filter(yolo=None):
        fake_yolo = yolo or FakeYOLO()
        monkeypatch.setattr(validator_module, "YOLODetector", lambda *args, **kwargs: fake_yolo)
        monkeypatch.setattr(validator_module, "ArcFaceEmbedder", lambda *args, **kwargs: FakeFaceEmbedder())
        filt = V2ContentFilter(
            {
                "device": "cpu",
                "checkpoint": {"dir": str(tmp_path / "checkpoints")},
                "coarse": {
                    "single_person_threshold": 0.5,
                    "audio_required": True,
                },
                "fine": {
                    "face_db_path": ":memory:",
                    "dedup_threshold": 0.99,
                    "speech_required": False,
                },
                "qualified_dir": str(tmp_path / "qualified"),
            },
            video_store=video_store,
        )
        return filt, fake_yolo

    return _make_filter


def test_coarse_cache_saved_and_loaded(monkeypatch, tmp_path, video_store, filter_factory):
    video_file = tmp_path / "test1.mp4"
    video_file.write_bytes(b"video")
    clip_path = str(tmp_path / "clip1.mp4")
    filt, fake_yolo = filter_factory()

    monkeypatch.setattr(validator_module.VU, "get_video_secs", lambda path: 30.0)
    monkeypatch.setattr(validator_module.VU, "check_audio", lambda path: True)
    monkeypatch.setattr(validator_module.VU, "clip_video", lambda path, segments, unit="second": [clip_path])
    monkeypatch.setattr(filt, "_fine_filter", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))

    row = make_row("test1", video_file)
    with pytest.raises(RuntimeError, match="stop"):
        filt._process_video(row)

    checkpoint = video_store.load_checkpoint("test1", "youtube", "coarse")
    assert checkpoint == {"segments": [[0.0, 30.0]], "clip_paths": [clip_path]}
    checkpoint_file = Path(filt.checkpoint_dir) / "test1" / "coarse_results.json"
    assert json.loads(checkpoint_file.read_text()) == checkpoint
    assert fake_yolo.calls == 1

    with pytest.raises(RuntimeError, match="stop"):
        filt._process_video(row)

    assert fake_yolo.calls == 1


def test_fine_progress_tracked(monkeypatch, tmp_path, video_store, filter_factory):
    video_file = tmp_path / "test2.mp4"
    video_file.write_bytes(b"video")
    clips = [str(tmp_path / f"clip{i}.mp4") for i in range(4)]
    filt, _ = filter_factory()
    monkeypatch.setattr(filt, "_coarse_filter", lambda path: (True, None, clips))
    monkeypatch.setattr(validator_module.VU, "get_video_secs", lambda path: 1.0)
    monkeypatch.setattr(validator_module.VU, "concat_videos", lambda paths: str(tmp_path / "merged.mp4"))
    monkeypatch.setattr(validator_module.VU, "remove_videos", lambda paths: None)
    monkeypatch.setattr(validator_module.os, "rename", lambda src, dst: None)

    first_run_calls = []

    def first_process_single(path):
        first_run_calls.append(path)
        if len(first_run_calls) == 3:
            raise RuntimeError("fine crash")
        return path, True, (0.9, 0.9, np.ones(512, dtype=np.float32)), None

    monkeypatch.setattr(filt, "_process_single_video", first_process_single)
    with pytest.raises(RuntimeError, match="fine crash"):
        filt._process_video(make_row("test2", video_file))

    checkpoint = video_store.load_checkpoint("test2", "youtube", "fine")
    assert checkpoint == {"processed_indices": [0, 1]}

    second_run_calls = []

    def second_process_single(path):
        second_run_calls.append(path)
        return path, True, (0.9, 0.9, np.full(512, 2.0, dtype=np.float32)), None

    monkeypatch.setattr(filt, "_process_single_video", second_process_single)
    filt._process_video(make_row("test2", video_file))

    assert second_run_calls == clips[2:]


def test_checkpoint_cleaned_on_complete(tmp_path, video_store, filter_factory):
    video_file = tmp_path / "test3.mp4"
    video_file.write_bytes(b"video")
    video_store.insert_or_update(make_video_meta("test3", str(video_file)), file_path=str(video_file))
    filt, _ = filter_factory()
    video_store.save_checkpoint("test3", "youtube", "coarse", {"clip_paths": ["clip.mp4"]})
    checkpoint_dir = Path(filt.checkpoint_dir) / "test3"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "coarse_results.json").write_text("{}")

    filt._complete_or_update(make_row("test3", video_file), "v2_passed")

    assert video_store.load_checkpoint("test3", "youtube", "coarse") is None
    assert not checkpoint_dir.exists()


def test_checkpoint_cleaned_on_fail(tmp_path, video_store, filter_factory):
    video_file = tmp_path / "test4.mp4"
    video_file.write_bytes(b"video")
    video_store.insert_or_update(make_video_meta("test4", str(video_file)), file_path=str(video_file))
    filt, _ = filter_factory()
    video_store.save_checkpoint("test4", "youtube", "fine", {"processed_indices": [0]})
    checkpoint_dir = Path(filt.checkpoint_dir) / "test4"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "coarse_results.json").write_text("{}")

    filt._fail_or_update(make_row("test4", video_file), "boom")

    assert video_store.load_checkpoint("test4", "youtube", "fine") is None
    assert not checkpoint_dir.exists()


def test_no_checkpoint_when_none_exists(monkeypatch, tmp_path, filter_factory):
    video_file = tmp_path / "test5.mp4"
    video_file.write_bytes(b"video")
    filt, fake_yolo = filter_factory()
    monkeypatch.setattr(validator_module.VU, "get_video_secs", lambda path: 30.0)
    monkeypatch.setattr(validator_module.VU, "check_audio", lambda path: True)
    monkeypatch.setattr(
        validator_module.VU,
        "clip_video",
        lambda path, segments, unit="second": [str(tmp_path / "clip5.mp4")],
    )
    monkeypatch.setattr(filt, "_fine_filter", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")))

    with pytest.raises(RuntimeError, match="stop"):
        filt._process_video(make_row("test5", video_file))

    assert fake_yolo.calls == 1
