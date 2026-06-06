from core.models.annotation_models import (
    ClarityScore,
    ExpressionIntensity,
    FacialFeatures,
    FacialMotion,
    Transcription,
)
from annotators.orchestrator import AnnotationOrchestrator


class FakeVideoStore:
    def __init__(self, videos=None):
        self.videos = videos or []
        self.runs = {}
        self.stages = []
        self.completed_runs = []
        self.failed_runs = []
        self.raise_pending = False

    def list_v2_passed_videos(self, limit=100):
        if self.raise_pending:
            raise RuntimeError("db down")
        return [video for video in self.videos if video.get("status") == "v2_passed"][:limit]

    def create_annotation_run(self, run_id, stage="pending"):
        self.runs[run_id] = {"run_id": run_id, "stage": stage, "status": "in_progress"}

    def get_in_progress_annotation_run(self):
        for run in self.runs.values():
            if run["status"] == "in_progress":
                return run
        return None

    def update_annotation_run_stage(self, run_id, stage, status="in_progress"):
        self.runs[run_id]["stage"] = stage
        self.runs[run_id]["status"] = status
        self.stages.append((run_id, stage, status))

    def complete_annotation_run(self, run_id):
        self.runs[run_id]["status"] = "completed"
        self.completed_runs.append(run_id)

    def fail_annotation_run(self, run_id):
        self.runs[run_id]["status"] = "failed"
        self.failed_runs.append(run_id)


class FakeAnnotationStore:
    def __init__(self):
        self.annotations = {}
        self.saved = []
        self.label_updates = []
        self.status_updates = []

    def create_annotation(self, video_id, platform, status="pending", error_message=None):
        self.annotations[(video_id, platform)] = {
            "video_id": video_id,
            "platform": platform,
            "status": status,
            "label_1_status": "pending",
            "label_2_status": "pending",
            "label_3_status": "pending",
            "label_4_status": "pending",
            "whisper_status": "pending",
            "error_message": error_message,
        }

    def get_annotation(self, video_id, platform):
        return self.annotations.get((video_id, platform))

    def update_label_status(self, video_id, platform, label, status):
        self.label_updates.append((video_id, platform, label, status))
        column = {"label_1": "label_1_status", "label_2": "label_2_status", "label_3": "label_3_status", "label_4": "label_4_status", "whisper": "whisper_status"}[label]
        self.annotations[(video_id, platform)][column] = status

    def update_annotation_status(self, video_id, platform, status, error_message=None):
        self.status_updates.append((video_id, platform, status, error_message))
        self.annotations[(video_id, platform)]["status"] = status
        self.annotations[(video_id, platform)]["error_message"] = error_message

    def save_facial_features(self, video_id, platform, result):
        self.saved.append(("facial_features", video_id, platform, result))

    def save_facial_motion(self, video_id, platform, result, raw_response=None):
        self.saved.append(("facial_motion", video_id, platform, result))

    def save_clarity(self, video_id, platform, result):
        self.saved.append(("clarity", video_id, platform, result))

    def save_expression(self, video_id, platform, result, raw_response=None):
        self.saved.append(("expression", video_id, platform, result))

    def save_transcription(self, video_id, platform, result):
        self.saved.append(("transcription", video_id, platform, result))


class MockAnnotator:
    def __init__(self, label_name, result=None, raises=None):
        self.label_name = label_name
        self.result = result
        self.raises = raises
        self.calls = []
        self.unloaded = False

    def annotate(self, video_path):
        self.calls.append(video_path)
        if self.raises:
            raise self.raises
        return self.result

    def unload(self):
        self.unloaded = True


def video(video_id="v1", status="v2_passed"):
    return {"video_id": video_id, "platform": "youtube", "file_path": f"/tmp/{video_id}.mp4", "status": status}


def features():
    return FacialFeatures(age=28, race="asian", hair_color="unknown", confidence={})


def clarity():
    return ClarityScore(mean_clarity=1, median_clarity=1, std_clarity=0, min_clarity=1, max_clarity=1, face_detected_ratio=1, per_frame=[1])


def motion():
    return FacialMotion(description="smile", key_movements=["smile"], duration_category="short")


def expression():
    return ExpressionIntensity(intensity=3, rationale="ok", dominant_expressions=["smile"])


def test_init():
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), FakeAnnotationStore(), [])

    assert orchestrator.video_store is not None
    assert orchestrator.annotation_store is not None


def test_get_pending_videos():
    annotation_store = FakeAnnotationStore()
    annotation_store.create_annotation("done", "youtube", status="completed")
    orchestrator = AnnotationOrchestrator(
        FakeVideoStore([video("todo"), video("done"), video("failed", status="v2_failed")]),
        annotation_store,
        [],
    )

    assert [item["video_id"] for item in orchestrator.get_pending_videos()] == ["todo"]


def test_create_annotation_entry():
    annotation_store = FakeAnnotationStore()
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), annotation_store, [])

    orchestrator.create_annotation_entry(video("v1"))

    assert annotation_store.get_annotation("v1", "youtube")["status"] == "in_progress"


def test_run_single_video_all_labels():
    annotation_store = FakeAnnotationStore()
    annotators = [
        MockAnnotator("facial_features", features()),
        MockAnnotator("clarity", clarity()),
        MockAnnotator("transcription", Transcription(text="hi", language="en", segments=[])),
    ]
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), annotation_store, annotators)

    orchestrator.run_single_video(video("v1"))

    assert len(annotation_store.saved) == 3
    assert annotation_store.get_annotation("v1", "youtube")["status"] == "completed"


def test_run_single_video_selected_labels():
    annotation_store = FakeAnnotationStore()
    facial = MockAnnotator("facial_features", features())
    clear = MockAnnotator("clarity", clarity())
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), annotation_store, [facial, clear], enabled_labels=["clarity"])

    orchestrator.run_single_video(video("v1"))

    assert facial.calls == []
    assert clear.calls == ["/tmp/v1.mp4"]


def test_skip_already_annotated():
    annotation_store = FakeAnnotationStore()
    annotation_store.create_annotation("v1", "youtube", status="completed")
    annotator = MockAnnotator("clarity", clarity())
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), annotation_store, [annotator])

    assert orchestrator.run_single_video(video("v1")) is False
    assert annotator.calls == []


def test_checkpoint_create():
    video_store = FakeVideoStore()
    orchestrator = AnnotationOrchestrator(video_store, FakeAnnotationStore(), [])

    run = orchestrator.create_checkpoint()

    assert run["status"] == "in_progress"
    assert run["run_id"] in video_store.runs


def test_checkpoint_resume():
    video_store = FakeVideoStore()
    video_store.create_annotation_run("existing-run")
    orchestrator = AnnotationOrchestrator(video_store, FakeAnnotationStore(), [])

    run = orchestrator.restore_checkpoint()

    assert run["run_id"] == "existing-run"


def test_checkpoint_complete():
    video_store = FakeVideoStore([video("v1")])
    annotator = MockAnnotator("clarity", clarity())
    orchestrator = AnnotationOrchestrator(video_store, FakeAnnotationStore(), [annotator])

    orchestrator.run(target_count=1)

    assert video_store.completed_runs


def test_save_results():
    annotation_store = FakeAnnotationStore()
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), annotation_store, [])
    annotation_store.create_annotation("v1", "youtube")

    orchestrator.save_result("v1", "youtube", "clarity", clarity())

    assert annotation_store.saved[0][0] == "clarity"


def test_label_status_update():
    annotation_store = FakeAnnotationStore()
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), annotation_store, [MockAnnotator("facial_features", features())])

    orchestrator.run_single_video(video("v1"))

    assert ("v1", "youtube", "label_1", "completed") in annotation_store.label_updates


def test_error_single_label():
    annotation_store = FakeAnnotationStore()
    good = MockAnnotator("clarity", clarity())
    bad = MockAnnotator("facial_features", raises=RuntimeError("bad label"))
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), annotation_store, [bad, good])

    orchestrator.run_single_video(video("v1"))

    assert ("v1", "youtube", "label_1", "failed") in annotation_store.label_updates
    assert ("clarity", "v1", "youtube", good.result) in annotation_store.saved


def test_error_fatal():
    video_store = FakeVideoStore()
    video_store.raise_pending = True
    orchestrator = AnnotationOrchestrator(video_store, FakeAnnotationStore(), [])

    try:
        orchestrator.run()
    except RuntimeError:
        pass

    assert video_store.failed_runs


def test_target_count():
    video_store = FakeVideoStore([video("v1"), video("v2")])
    annotator = MockAnnotator("clarity", clarity())
    orchestrator = AnnotationOrchestrator(video_store, FakeAnnotationStore(), [annotator])

    assert orchestrator.run(target_count=1) == 1
    assert annotator.calls == ["/tmp/v1.mp4"]


def test_cleanup():
    annotator = MockAnnotator("clarity", clarity())
    orchestrator = AnnotationOrchestrator(FakeVideoStore(), FakeAnnotationStore(), [annotator])

    orchestrator.cleanup()

    assert annotator.unloaded is True
