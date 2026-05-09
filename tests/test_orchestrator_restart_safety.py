import sqlite3
from types import SimpleNamespace

from core.models.video_meta import VideoMeta
from core.storage.video_store import VideoStore
import core.pipeline_orchestrator as orchestrator


def make_video(video_id: str, platform: str = "youtube") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://example.com/{platform}/{video_id}.mp4",
        platform=platform,
        title=f"title {video_id}",
        duration_seconds=120,
        resolution="1080p",
        channel="channel",
        publish_date="2026-01-01",
        view_count=1000,
        tags=[],
        search_term="term",
        extra={},
    )


def all_pipeline_runs(store: VideoStore):
    with store._connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM pipeline_run ORDER BY created_at, run_id").fetchall()
        return [dict(row) for row in rows]


def pipeline_config(video_store: VideoStore):
    return {
        "project": {"video_db": video_store.db_path},
        "explorer": {},
        "search": {"per_platform_results": 1},
        "download": {},
        "v1_filter": {},
        "v2_filter": {},
        "pipeline": {
            "batch_size": 1,
            "max_retries": 0,
            "target_qualified": 0,
            "pending_ratio_threshold": 2.0,
            "pending_wait_seconds": 0,
        },
    }


def patch_orchestrator(monkeypatch, events):
    class FakeExplorer:
        def __init__(self, config):
            self.config = config
            events.append("explorer:init")

        def generate_batch(self, batch_size=10):
            events.append("explorer:generate")
            return [SimpleNamespace(text="term")]

        def receive_feedback(self, feedback):
            events.append("explorer:feedback")

        def adapt_strategy(self):
            events.append("explorer:adapt")

        def get_status(self):
            return {"ok": True}

        def save_state(self):
            events.append("explorer:save")

    class FakeSearcher:
        def __init__(self, platform, **kwargs):
            self.platform = platform
            events.append(f"searcher:init:{platform}")

        def search(self, query, max_results=20):
            events.append(f"search:{self.platform}")
            return [make_video(f"{self.platform}-{query}", platform=self.platform)]

    class FakeDownloader:
        def __init__(self, config_dict=None):
            events.append("downloader:init")

        def download(self, urls):
            events.append("download")
            return {url: f"/tmp/{index}.mp4" for index, url in enumerate(urls)}

    class FakePreFilter:
        def __init__(self, config):
            events.append("v1:init")

        def filter(self, videos, search_term):
            events.append("v1")
            feedback = {
                "search_term": search_term,
                "v1_pass_rate": 1.0 if videos else 0.0,
                "fail_reasons": {},
                "total_received": len(videos),
            }
            return videos, feedback

    class FakeV2ContentFilter:
        def __init__(self, config, video_store, explorer=None):
            self.thread = None
            events.append("v2:init")

        def start(self):
            events.append("v2:start")

        def stop(self):
            events.append("v2:stop")

    monkeypatch.setattr(orchestrator, "AdaptiveScheduler", FakeExplorer)
    monkeypatch.setattr(orchestrator, "YtDlpSearchApi", FakeSearcher)
    monkeypatch.setattr(orchestrator, "DefaultDownloader", FakeDownloader)
    monkeypatch.setattr(orchestrator, "PreFilter", FakePreFilter)
    monkeypatch.setattr(orchestrator, "V2ContentFilter", FakeV2ContentFilter)


def test_pipeline_run_crud(video_store):
    video_store.create_pipeline_run("run-1")

    created = video_store.get_run_by_id("run-1")
    assert created["run_id"] == "run-1"
    assert created["stage"] == "search"
    assert created["status"] == "in_progress"

    video_store.update_pipeline_run_stage("run-1", "download")
    updated = video_store.get_run_by_id("run-1")
    assert updated["stage"] == "download"
    assert updated["status"] == "in_progress"

    video_store.complete_pipeline_run("run-1")
    completed = video_store.get_run_by_id("run-1")
    assert completed["status"] == "completed"
    assert completed["completed_at"] is not None

    assert video_store.delete_pipeline_run("run-1") is True
    assert video_store.get_pipeline_run("run-1") is None


def test_get_in_progress_run(video_store):
    video_store.create_pipeline_run("run-1")

    in_progress = video_store.get_in_progress_run()
    assert in_progress["run_id"] == "run-1"

    video_store.complete_pipeline_run("run-1")
    assert video_store.get_in_progress_run() is None


def test_no_crash_clean_start(video_store, monkeypatch):
    events = []
    patch_orchestrator(monkeypatch, events)

    orchestrator.run_pipeline(pipeline_config(video_store))

    runs = all_pipeline_runs(video_store)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert "search:youtube" in events
    assert "search:bilibili" in events
    assert "download" in events
    assert "v1" in events
    assert "v2:start" in events


def test_marks_stages_complete(video_store, monkeypatch):
    events = []
    patch_orchestrator(monkeypatch, events)

    orchestrator.run_pipeline(pipeline_config(video_store))

    run = all_pipeline_runs(video_store)[0]
    assert run["stage"] == "v2"
    assert run["status"] == "completed"
    assert run["completed_at"] is not None


def test_multirun_isolation(video_store, monkeypatch):
    events = []
    patch_orchestrator(monkeypatch, events)

    orchestrator.run_pipeline(pipeline_config(video_store))
    orchestrator.run_pipeline(pipeline_config(video_store))

    runs = all_pipeline_runs(video_store)
    assert len(runs) == 2
    assert runs[0]["run_id"] != runs[1]["run_id"]
    assert {run["status"] for run in runs} == {"completed"}


def test_resumes_at_v2_without_repeating_completed_stages(video_store, monkeypatch):
    events = []
    patch_orchestrator(monkeypatch, events)
    video_store.create_pipeline_run("crashed-run", stage="v2")

    orchestrator.run_pipeline(pipeline_config(video_store))

    run = video_store.get_run_by_id("crashed-run")
    assert run["stage"] == "v2"
    assert run["status"] == "completed"
    assert "search:youtube" not in events
    assert "search:bilibili" not in events
    assert "download" not in events
    assert "v1" not in events
    assert "v2:start" in events


def test_resumes_from_checkpoint_stage_as_next_stage(video_store, monkeypatch):
    events = []
    patch_orchestrator(monkeypatch, events)
    video_store.create_pipeline_run("crashed-run", stage="download")

    orchestrator.run_pipeline(pipeline_config(video_store))

    run = video_store.get_run_by_id("crashed-run")
    assert run["stage"] == "v2"
    assert run["status"] == "completed"
    assert "search:youtube" not in events
    assert "search:bilibili" not in events
    assert "download" in events
    assert "v1" in events
    assert "v2:start" in events


def test_invalid_resume_stage_restarts_from_search(video_store, monkeypatch):
    events = []
    patch_orchestrator(monkeypatch, events)
    video_store.create_pipeline_run("crashed-run", stage="unknown")

    orchestrator.run_pipeline(pipeline_config(video_store))

    run = video_store.get_run_by_id("crashed-run")
    assert run["stage"] == "v2"
    assert run["status"] == "completed"
    assert "search:youtube" in events
    assert "download" in events
    assert "v1" in events
