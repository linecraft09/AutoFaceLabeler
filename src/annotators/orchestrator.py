from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Dict, Iterable, List, Optional


class AnnotationOrchestrator:
    """Milestone 2 annotation pipeline coordinator."""

    LABEL_STATUS = {
        "facial_features": "label_1",
        "facial_motion": "label_2",
        "clarity": "label_3",
        "expression": "label_4",
        "transcription": "whisper",
    }

    def __init__(
        self,
        video_store: Any,
        annotation_store: Any,
        annotators: Iterable[Any],
        enabled_labels: Optional[Iterable[str]] = None,
    ) -> None:
        self.video_store = video_store
        self.annotation_store = annotation_store
        self.annotators = list(annotators)
        self.enabled_labels = set(enabled_labels) if enabled_labels is not None else None
        self.current_run: Optional[Dict[str, Any]] = None

    def run(self, target_count: Optional[int] = None) -> int:
        run = self.restore_checkpoint() or self.create_checkpoint()
        self.current_run = run
        run_id = run["run_id"]
        processed = 0

        try:
            pending = self.get_pending_videos(limit=target_count or 100)
            for video in pending:
                if target_count is not None and processed >= target_count:
                    break
                if self.run_single_video(video, run_id=run_id):
                    processed += 1
            self.video_store.complete_annotation_run(run_id)
            return processed
        except Exception:
            self.video_store.fail_annotation_run(run_id)
            raise
        finally:
            self.cleanup()

    def get_pending_videos(self, limit: int = 100) -> List[Dict[str, Any]]:
        if hasattr(self.video_store, "list_v2_passed_videos"):
            videos = self.video_store.list_v2_passed_videos(limit=limit)
        else:
            videos = self._query_v2_passed_videos(limit)

        pending = []
        for video in videos:
            annotation = self.annotation_store.get_annotation(video["video_id"], video["platform"])
            if annotation and annotation.get("status") == "completed":
                continue
            pending.append(video)
        return pending

    def _query_v2_passed_videos(self, limit: int) -> List[Dict[str, Any]]:
        if not hasattr(self.video_store, "_connect"):
            return []
        with self.video_store._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM videos
                WHERE status = 'v2_passed'
                  AND file_path IS NOT NULL
                  AND file_path != ''
                ORDER BY updated_at ASC, created_at ASC, id ASC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_annotation_entry(self, video: Dict[str, Any]) -> None:
        self.annotation_store.create_annotation(
            video["video_id"],
            video["platform"],
            status="in_progress",
        )

    def run_single_video(self, video: Dict[str, Any], run_id: Optional[str] = None) -> bool:
        video_id = video["video_id"]
        platform = video["platform"]
        existing = self.annotation_store.get_annotation(video_id, platform)
        if existing and existing.get("status") == "completed":
            return False

        self.create_annotation_entry(video)
        had_error = False
        for annotator in self.annotators:
            label_name = annotator.label_name
            if self.enabled_labels is not None and label_name not in self.enabled_labels:
                continue
            if run_id:
                self.video_store.update_annotation_run_stage(run_id, f"{video_id}:{label_name}")
            try:
                result = annotator.annotate(video["file_path"])
                self.save_result(video_id, platform, label_name, result)
                self.mark_label_completed(video_id, platform, label_name)
            except Exception as exc:
                had_error = True
                self.mark_label_failed(video_id, platform, label_name)
                self.annotation_store.update_annotation_status(
                    video_id,
                    platform,
                    "partial_failed",
                    error_message=str(exc),
                )

        if had_error:
            self.annotation_store.update_annotation_status(video_id, platform, "partial_failed")
        else:
            self.annotation_store.update_annotation_status(video_id, platform, "completed")
        return True

    def save_result(self, video_id: str, platform: str, label_name: str, result: Any) -> None:
        if label_name == "facial_features":
            self.annotation_store.save_facial_features(video_id, platform, result)
        elif label_name == "clarity":
            self.annotation_store.save_clarity(video_id, platform, result)
        elif label_name == "transcription":
            self.annotation_store.save_transcription(video_id, platform, result)
        elif label_name == "facial_motion":
            self.annotation_store.save_facial_motion(video_id, platform, result)
        elif label_name == "expression":
            self.annotation_store.save_expression(video_id, platform, result)
        elif label_name == "facial_analysis":
            if isinstance(result, dict):
                motion = result.get("facial_motion")
                expression = result.get("expression")
                if motion is not None:
                    self.annotation_store.save_facial_motion(video_id, platform, motion, raw_response=result.get("raw_response"))
                if expression is not None:
                    self.annotation_store.save_expression(video_id, platform, expression, raw_response=result.get("raw_response"))
        else:
            raise ValueError(f"unknown annotator label: {label_name}")

    def mark_label_completed(self, video_id: str, platform: str, label_name: str) -> None:
        for label in self.status_labels_for(label_name):
            self.annotation_store.update_label_status(video_id, platform, label, "completed")

    def mark_label_failed(self, video_id: str, platform: str, label_name: str) -> None:
        for label in self.status_labels_for(label_name):
            self.annotation_store.update_label_status(video_id, platform, label, "failed")

    def status_labels_for(self, label_name: str) -> List[str]:
        if label_name == "facial_analysis":
            return ["label_2", "label_4"]
        status = self.LABEL_STATUS.get(label_name)
        if status is None:
            return []
        return [status]

    def restore_checkpoint(self) -> Optional[Dict[str, Any]]:
        return self.video_store.get_in_progress_annotation_run()

    def create_checkpoint(self) -> Dict[str, Any]:
        run_id = f"annotation-{uuid.uuid4().hex}"
        self.video_store.create_annotation_run(run_id)
        return self.video_store.get_in_progress_annotation_run() or {
            "run_id": run_id,
            "stage": "pending",
            "status": "in_progress",
        }

    def cleanup(self) -> None:
        for annotator in self.annotators:
            if hasattr(annotator, "unload"):
                annotator.unload()
            elif hasattr(annotator, "close"):
                annotator.close()
