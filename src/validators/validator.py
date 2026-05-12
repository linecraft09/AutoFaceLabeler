import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Any, List
from typing import Tuple, Optional

import cv2
import numpy as np
from insightface.utils import face_align

import aflutils.video_utils as VU
from aflutils.logger import get_logger
from core.storage.video_store import VideoStatus, VideoStore
from .v2_models.arcface_embedder import ArcFaceEmbedder
from .v2_models.face_quality import compute_laplacian_variance
from .v2_models.landmark_batch import BatchLandmarkPose
from .v2_models.scrfd_batch_detector import BatchSCRFDDetector, Detection
from .v2_models.yolo_detector import YOLODetector

logger = get_logger(__name__)


class V2ContentFilter:

    def __init__(self, config: Dict[str, Any], video_store: VideoStore, explorer=None):
        self.config = config
        self.store = video_store
        self.explorer = explorer
        self.stop_event = threading.Event()
        self.thread = None
        self.claim_lease_seconds = int(config.get('claim_lease_seconds', 600))
        self.max_retries = int(config.get('max_retries', 3))
        checkpoint_cfg = config.get('checkpoint', {}) if isinstance(config, dict) else {}
        self.checkpoint_dir = Path(checkpoint_cfg.get('dir', './data/checkpoints'))
        if hasattr(self.store, 'set_max_retries'):
            self.store.set_max_retries(self.max_retries)

        yolo_cfg = config.get('coarse', {})
        self.yolo = YOLODetector(
            model_path=yolo_cfg.get('model_path', 'yolov8n.pt'),
            device=config.get('device', 'cpu'),
            batch_size=yolo_cfg.get('batch_size', 16),
        )

        self.single_person_threshold = yolo_cfg.get('single_person_threshold', 0.8)
        self.audio_required = yolo_cfg.get('audio_required', True)

        fine_cfg = config.get('fine', {})
        self.fine_cfg = fine_cfg
        head_pose_cfg = fine_cfg.get('head_pose', {})
        laplacian_cfg = fine_cfg.get('laplacian', {})
        face_cfg = fine_cfg.get('face', {})
        self.max_head_angle = head_pose_cfg.get('max_angle', 10)
        self.head_pose_required_ratio = head_pose_cfg.get('required_ratio', 0.8)
        self.laplacian_threshold = laplacian_cfg.get('threshold', 100)
        self.laplacian_required_ratio = laplacian_cfg.get('required_ratio', 0.8)
        self.face_required_ratio = face_cfg.get('required_ratio', 0.8)
        self.face_primary_only = bool(face_cfg.get('primary_only', False))
        self.dedup_threshold = fine_cfg.get('dedup_threshold', 0.7)
        self.min_qualified_duration = float(fine_cfg.get('min_duration', 30))
        self.batch_size = int(fine_cfg.get('batch_size', 16))
        self._batch_detector = None
        self._landmark_pose = None
        self._fine_checkpoint_context = None

        self.known_embeddings = []
        self.max_head_angle = head_pose_cfg.get('max_angle', 10)
        self.head_pose_required_ratio = head_pose_cfg.get('required_ratio', 0.8)
        self.laplacian_threshold = laplacian_cfg.get('threshold', 100)

        device = config.get('device', 'cpu')
        # ArcFace
        self.face_embedder = ArcFaceEmbedder(
            device=device,
            db_path=fine_cfg.get('face_db_path', 'data/face_index.faiss'),
            batch_size=self.batch_size,
        )
        self._maybe_rebuild_faiss_index(fine_cfg)

        self.speech_required = fine_cfg.get('speech_required', False)
        if self.speech_required:
            from .v2_models.speaker_detector import SpeakerDetector
            self.speaker_detector = SpeakerDetector()

        self.feedback = {
            'total_processed': 0,
            'coarse_pass': 0,
            'fine_pass': 0,
            'fail_reasons': {
                'no_single_person': 0,
                'no_audio': 0,
                'no_face_detected': 0,
                'head_pose_out_of_range': 0,
                'blurry_face': 0,
                'duplicate': 0,
                'merged_duration_too_short': 0,
            }
        }

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            logger.warning("V2 filter already running")
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("V2 filter thread started")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
            logger.info("V2 filter thread stopped")
        if getattr(self, 'face_embedder', None):
            try:
                self.face_embedder._save_index()
            except Exception as e:
                logger.exception(f"Failed to save face index on stop: {e}")

    def _run_loop(self):
        while not self.stop_event.is_set():
            batch_size = int(self.config.get('pending_limit', 5))
            thread_name = threading.current_thread().name
            pending = self.store.claim_pending_videos(
                limit=batch_size,
                owner=thread_name,
                lease_seconds=self.claim_lease_seconds,
            )
            if not pending:
                time.sleep(self.config.get('poll_interval_seconds', 5))
                continue

            for video_row in pending:
                if self.stop_event.is_set():
                    break
                try:
                    self._process_video(video_row)
                except Exception as e:
                    logger.error(f"V2 filter failed on video {video_row.get('video_id', '?')}: {e}", exc_info=True)
                    try:
                        self._fail_or_update(
                            video_row,
                            f'v2_exception: {e}',
                            retry_within_seconds=60,
                        )
                    except Exception as status_e:
                        logger.error(f"Failed to update error status: {status_e}")

    def _process_video(self, video_row: Dict[str, Any]):
        video_id = video_row['video_id']
        platform = video_row['platform']
        file_path = video_row['file_path']
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Video file missing: {file_path}")
            self._fail_or_update(video_row, 'file_missing')
            return

        generated_clip_paths: List[str] = []
        cleanup_generated_clips = False
        try:
            cached_coarse = self._load_coarse_checkpoint(video_id, platform)
            if cached_coarse is not None:
                coarse_result = True
                coarse_fail_reason = None
                clip_paths = cached_coarse.get('clip_paths', [])
                logger.info(f"Loaded coarse checkpoint for {platform}:{video_id}")
            else:
                coarse_result, coarse_fail_reason, clip_paths = self._coarse_filter(file_path)
                if coarse_result:
                    self._save_coarse_checkpoint(video_id, platform, clip_paths)
            generated_clip_paths = clip_paths
            if not coarse_result:
                self._complete_or_update(video_row, 'v2_coarse_failed', coarse_fail_reason)
                cleanup_generated_clips = True
                self._update_feedback(coarse_fail_reason)
                logger.info(f"Video {video_id} doesn't pass coarse filter because of {coarse_fail_reason}")
                return
            self.feedback['coarse_pass'] += 1
            logger.info(f"Video {video_id} passed coarse filter")

            fine_checkpoint = self.store.load_checkpoint(video_id, platform, 'fine')
            processed_indices = set()
            if fine_checkpoint:
                processed_indices = {
                    int(index)
                    for index in fine_checkpoint.get('processed_indices', [])
                }
                logger.info(
                    f"Loaded fine checkpoint for {platform}:{video_id}, "
                    f"processed={len(processed_indices)}/{len(clip_paths)}"
                )

            remaining_pairs = [
                (index, clip_path)
                for index, clip_path in enumerate(clip_paths)
                if index not in processed_indices
            ]
            if not remaining_pairs and processed_indices:
                clip_path = None
                fine_fail_reason = None
                logger.info(f"All fine-filter clips already processed for {platform}:{video_id}")
            else:
                remaining_indices = [index for index, _ in remaining_pairs]
                remaining_clip_paths = [path for _, path in remaining_pairs]
                self._begin_fine_checkpoint(video_id, platform, processed_indices, remaining_indices)
                try:
                    clip_path, fine_fail_reason = self._fine_filter(
                        remaining_clip_paths,
                        video_id=video_id,
                        platform=platform,
                    )
                finally:
                    self._end_fine_checkpoint()
                all_indices = list(range(len(clip_paths)))
                self.store.save_checkpoint(
                    video_id,
                    platform,
                    'fine',
                    {'processed_indices': all_indices},
                )
            if not clip_path:
                self._complete_or_update(video_row, 'v2_fine_failed', fine_fail_reason)
                cleanup_generated_clips = True
                self._update_feedback(fine_fail_reason)
                logger.info(f"Video {video_id} doesn't pass fine filter because of {fine_fail_reason}")
                return
            self.feedback['fine_pass'] += 1
            logger.info(f"Video {video_id} passed fine filter")

            self._complete_or_update(video_row, 'v2_passed')
            cleanup_generated_clips = True
            if clip_path:
                qualified_dir = Path(self.config.get('qualified_dir', './data/qualified'))
                qualified_dir.mkdir(parents=True, exist_ok=True)
                dest = qualified_dir / f"{video_id}_qualified.mkv"
                os.rename(clip_path, dest)
                logger.info(f"Qualified clip saved: {dest}")
            else:
                logger.info(f"Video {video_id} passed all filters (whole video)")

            if self.explorer:
                feedback = self._build_feedback_for_explorer(video_row['search_term'])
                self.explorer.receive_feedback(feedback)

            self.feedback['total_processed'] += 1
        finally:
            if cleanup_generated_clips:
                VU.remove_videos(generated_clip_paths)

    @staticmethod
    def _has_v2_claim(video_row: Dict[str, Any]) -> bool:
        return video_row.get('status') == VideoStatus.V2_IN_PROGRESS.value

    @staticmethod
    def _is_terminal_v2_status(status: str) -> bool:
        return status in {
            VideoStatus.V2_PASSED.value,
            VideoStatus.V2_FAILED.value,
            VideoStatus.V2_COARSE_FAILED.value,
            VideoStatus.V2_FINE_FAILED.value,
        }

    def _checkpoint_video_dir(self, video_id: str) -> Path:
        return self.checkpoint_dir / video_id

    def _coarse_checkpoint_path(self, video_id: str) -> Path:
        return self._checkpoint_video_dir(video_id) / 'coarse_results.json'

    def _load_coarse_checkpoint(self, video_id: str, platform: str) -> Optional[Dict[str, Any]]:
        db_checkpoint = self.store.load_checkpoint(video_id, platform, 'coarse')
        checkpoint_file = self._coarse_checkpoint_path(video_id)
        if db_checkpoint is None or not checkpoint_file.exists():
            return None
        try:
            with checkpoint_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data.get('clip_paths'), list):
                logger.warning(f"Invalid coarse checkpoint payload for {platform}:{video_id}")
                return None
            logger.debug(f"Loaded coarse checkpoint file for {platform}:{video_id}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load coarse checkpoint for {platform}:{video_id}: {e}")
            return None

    def _save_coarse_checkpoint(self, video_id: str, platform: str, clip_paths: List[str]) -> None:
        segments = getattr(self, '_last_coarse_segments', [])
        payload = {
            'segments': [[float(start), float(end)] for start, end in segments],
            'clip_paths': list(clip_paths),
        }
        video_checkpoint_dir = self._checkpoint_video_dir(video_id)
        video_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self._coarse_checkpoint_path(video_id)
        with checkpoint_file.open('w', encoding='utf-8') as f:
            json.dump(payload, f)
        self.store.save_checkpoint(video_id, platform, 'coarse', payload)
        logger.info(f"Saved coarse checkpoint for {platform}:{video_id}")

    def _cleanup_checkpoint(self, video_id: str, platform: str) -> None:
        try:
            self.store.delete_checkpoint(video_id, platform)
        except Exception as e:
            logger.warning(f"Failed to delete DB checkpoint for {platform}:{video_id}: {e}")
        checkpoint_path = self._checkpoint_video_dir(video_id)
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            logger.info(f"Deleted checkpoint directory for {platform}:{video_id}: {checkpoint_path}")

    def _begin_fine_checkpoint(
        self,
        video_id: str,
        platform: str,
        processed_indices: set,
        remaining_indices: List[int],
    ) -> None:
        self._fine_checkpoint_context = {
            'video_id': video_id,
            'platform': platform,
            'processed_indices': set(processed_indices),
            'remaining_indices': list(remaining_indices),
            'cursor': 0,
        }

    def _end_fine_checkpoint(self) -> None:
        self._fine_checkpoint_context = None

    def _mark_fine_clip_processed(self) -> None:
        context = getattr(self, '_fine_checkpoint_context', None)
        if not context:
            return
        cursor = context['cursor']
        remaining_indices = context['remaining_indices']
        if cursor >= len(remaining_indices):
            return
        context['processed_indices'].add(remaining_indices[cursor])
        context['cursor'] = cursor + 1
        self.store.save_checkpoint(
            context['video_id'],
            context['platform'],
            'fine',
            {'processed_indices': sorted(context['processed_indices'])},
        )

    def _complete_or_update(
        self,
        video_row: Dict[str, Any],
        status: str,
        fail_reason: Optional[str] = None
    ):
        video_id = video_row['video_id']
        platform = video_row['platform']
        updated = False
        if self._has_v2_claim(video_row):
            completed = self.store.complete_v2(video_id, platform, status)
            if completed and fail_reason:
                self.store.update_status(video_id, platform, status, fail_reason)
                updated = True
            elif completed:
                updated = True
            elif not completed:
                logger.warning(f"Skipped V2 completion for {video_id}: claim state changed")
        else:
            self.store.update_status(video_id, platform, status, fail_reason)
            updated = True
        if updated and self._is_terminal_v2_status(status):
            self._cleanup_checkpoint(video_id, platform)

    def _fail_or_update(
        self,
        video_row: Dict[str, Any],
        error: str,
        retry_within_seconds: Optional[int] = None
    ):
        video_id = video_row['video_id']
        platform = video_row['platform']
        updated = False
        terminal = False
        if self._has_v2_claim(video_row):
            updated = self.store.fail_v2(
                video_id,
                platform,
                error=error,
                retry_within_seconds=retry_within_seconds,
            )
            if updated:
                row = self.store.get_video_by_id(video_id, platform)
                terminal = bool(row and row.get('status') == VideoStatus.V2_FAILED.value)
        else:
            self.store.update_status(video_id, platform, 'v2_failed', error)
            updated = True
            terminal = True
        if updated and terminal:
            self._cleanup_checkpoint(video_id, platform)

    def _coarse_filter(self, video_path: str) -> Tuple[bool, Optional[str], List[str]]:
        segments, total_single_secs = self.yolo.detect_single_person_segments(video_path)
        self._last_coarse_segments = segments
        total_secs = VU.get_video_secs(video_path)
        if total_secs == 0:
            return False, 'no_frames', []
        single_ratio = total_single_secs / total_secs
        if single_ratio < self.single_person_threshold:
            return False, 'no_single_person', []

        # 闂婃娊?濡�?????
        if self.audio_required:
            has_audio = VU.check_audio(video_path)
            if not has_audio:
                return False, 'no_audio', []

        clip_file_paths = VU.clip_video(video_path, segments, unit="second")
        if clip_file_paths is None or len(clip_file_paths) == 0:
            return False, 'clip_failed', []
        return True, None, clip_file_paths

    def _fine_filter(
        self,
        video_paths: List[str],
        video_id: str = "",
        platform: str = ""
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process multiple clips in fine filtering and keep qualified clips in temporal order.
        """

        # Collect per-clip processing output.
        results = []  # (video_path, success, data, fail_reason)
        # data is (pose_ratio, clarity_ratio, representative_embedding) or None

        for video_path in video_paths:
            try:
                result = self._process_single_video(video_path)
                results.append(result)
                self._mark_fine_clip_processed()
            except Exception as e:
                logger.error(f"fine filter single video {video_path} cause exception: {e}")
                if getattr(self, '_fine_checkpoint_context', None):
                    raise
                results.append((video_path, False, None, f"exception: {str(e)}"))

        # Collect passed candidates from per-clip processing.
        passed_candidates = []  # (video_path, pose_ratio, clarity_ratio, embedding, duration_secs)
        fail_reasons = {}
        duration_cache = {}
        for video_path, success, data, fail_reason in results:
            if success:
                pose_ratio, clarity_ratio, embedding = data
                if video_path not in duration_cache:
                    duration_cache[video_path] = VU.get_video_secs(video_path)
                passed_candidates.append(
                    (video_path, pose_ratio, clarity_ratio, embedding, duration_cache[video_path])
                )
            else:
                fail_reasons[video_path] = fail_reason

        # Dedup in temporal order. The first occurrence is kept, and later
        # duplicate clips are skipped before embeddings are persisted.
        passed_videos = []
        already_selected_embeddings = []
        for video_path, pose_ratio, clarity_ratio, embedding, duration_secs in passed_candidates:
            if self.face_embedder.is_duplicate(embedding, self.dedup_threshold):
                fail_reasons[video_path] = 'duplicate'
                continue
            is_local_duplicate = any(
                np.linalg.norm(embedding - selected) < self.dedup_threshold
                for selected in already_selected_embeddings
            )
            if is_local_duplicate:
                fail_reasons[video_path] = 'duplicate'
                continue
            passed_videos.append((video_path, pose_ratio, clarity_ratio, embedding, duration_secs))
            already_selected_embeddings.append(embedding)

        if not passed_videos:
            all_fail_reasons = '|'.join(fail_reasons.values())
            return None, all_fail_reasons

        to_merge = [video_path for video_path, _, _, _, _ in passed_videos]
        selected_embeddings = [embedding for _, _, _, embedding, _ in passed_videos]
        total_duration = sum(duration_secs for _, _, _, _, duration_secs in passed_videos)
        logger.info(
            f"Fine filter selected {len(passed_videos)} clips, total duration {total_duration:.2f}s"
        )

        if self.min_qualified_duration > 0 and total_duration < self.min_qualified_duration:
            logger.info(
                f"Fine filter rejected selected clips because total duration "
                f"{total_duration:.2f}s is shorter than {self.min_qualified_duration:.2f}s"
            )
            return None, "merged_duration_too_short"

        if len(to_merge) == 1:
            self._save_face_embeddings(video_id, platform, selected_embeddings)
            for embedding in selected_embeddings:
                self.face_embedder.add_embedding(embedding)
            return to_merge[0], None

        merged_path = VU.concat_videos(to_merge)
        if merged_path is None:
            return None, "concat_failed"

        if self.min_qualified_duration > 0:
            merged_duration = VU.get_video_secs(merged_path)
            if merged_duration < self.min_qualified_duration:
                logger.info(
                    f"Fine filter rejected merged video {merged_path} because duration "
                    f"{merged_duration:.2f}s is shorter than {self.min_qualified_duration:.2f}s"
                )
                try:
                    os.unlink(merged_path)
                except OSError as e:
                    logger.warning(f"Failed to remove short merged video {merged_path}: {e}")
                return None, "merged_duration_too_short"

        self._save_face_embeddings(video_id, platform, selected_embeddings)
        for embedding in selected_embeddings:
            self.face_embedder.add_embedding(embedding)
        return merged_path, None

    def _maybe_rebuild_faiss_index(self, fine_cfg: Dict[str, Any]):
        face_db_path = fine_cfg.get('face_db_path', 'data/face_index.faiss')
        rebuild_requested = bool(fine_cfg.get('rebuild_faiss_on_startup', False))
        index_missing = face_db_path != ':memory:' and not os.path.exists(face_db_path)
        if not (rebuild_requested or index_missing):
            return
        if not hasattr(self.face_embedder, 'index'):
            return

        project_cfg = self.config.get('project', {}) if isinstance(self.config, dict) else {}
        video_db_path = (
            project_cfg.get('video_db')
            or self.config.get('video_db')
            or getattr(self.store, 'db_path', './data/videos.db')
        )
        rebuild_store = VideoStore(db_path=video_db_path, max_retries=self.max_retries)
        rebuilt_index = ArcFaceEmbedder.rebuild_from_db(
            rebuild_store,
            dim=getattr(self.face_embedder, 'dim', 512),
        )
        self.face_embedder.index = rebuilt_index
        if hasattr(self.face_embedder, '_last_saved_ntotal'):
            self.face_embedder._last_saved_ntotal = -1
        logger.info(
            f"Rebuilt FAISS index from SQLite embeddings at {video_db_path}, "
            f"size={rebuilt_index.ntotal}"
        )

    def _save_face_embeddings(
        self,
        video_id: str,
        platform: str,
        embeddings: List[np.ndarray]
    ) -> int:
        if not video_id or not platform or not embeddings:
            return 0
        inserted = self.store.save_embeddings(video_id, platform, embeddings)
        logger.debug(
            f"Persisted {inserted}/{len(embeddings)} face embeddings for {platform}:{video_id}"
        )
        return inserted

    def _process_single_video(self, video_path: str) -> Tuple[
        str, bool, Optional[Tuple[float, float, np.ndarray]], Optional[str]]:
        total_frames = VU.get_video_frames(video_path)
        if total_frames == 0:
            return video_path, False, None, 'no_frames'

        if self.speech_required:
            has_speech = self.speaker_detector.detect_speech_from_video(video_path)
            if not has_speech:
                return video_path, False, None, 'no_speech'

        sampling_cfg = self.config.get('fine', {}).get('sampling', {})
        sample_indices = self._sample_indices(
            total_frames,
            float(sampling_cfg.get('rate', 0.1)),
            int(sampling_cfg.get('max_frames', 200)),
        )
        sampled_frames, _ = VU.read_sampled_frames(video_path, sample_indices)
        if not sampled_frames:
            return video_path, False, None, 'no_frames'

        detections_by_frame: List[List[Detection]] = []
        detector = self._get_batch_detector()
        for frame_batch in self._batched(sampled_frames, self.batch_size):
            detections_by_frame.extend(detector.detect_batch(frame_batch))

        detected_frame_count = sum(1 for detections in detections_by_frame if detections)
        face_ratio = detected_frame_count / len(sampled_frames)
        if face_ratio < self.face_required_ratio:
            return video_path, False, None, 'no_face_detected'

        arcface_crops = []
        landmark_frames = []
        landmark_bboxes = []
        clarity_crops = []
        clarity_crop_sizes = []
        for frame, detections in zip(sampled_frames, detections_by_frame):
            for detection, bbox in self._select_frame_detections(frame, detections):
                x1, y1, x2, y2 = bbox.tolist()
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                clarity_crops.append(face_crop)
                clarity_crop_sizes.append((x2 - x1, y2 - y1))
                landmark_frames.append(frame)
                landmark_bboxes.append(bbox.astype(np.float32))
                arcface_crops.append(self._make_arcface_crop(frame, face_crop, detection))

        total_faces = len(arcface_crops)
        if total_faces == 0:
            return video_path, False, None, 'no_face_detected'

        embeddings = self.face_embedder.embed_crops_batch(arcface_crops)
        if embeddings.shape[0] == 0:
            return video_path, False, None, 'no_face_detected'
        representative_embedding = self._mean_embedding(embeddings)

        poses = self._get_landmark_pose().estimate_poses_batch(landmark_frames, landmark_bboxes)
        good_pose_count = sum(
            1 for pose in poses
            if np.all(np.abs(pose) <= float(self.max_head_angle))
        )
        pose_ratio = good_pose_count / total_faces

        clarity_values = [
            float(compute_laplacian_variance(face_crop))
            for face_crop in clarity_crops
        ]
        good_clarity_count = sum(
            1 for clarity_value in clarity_values
            if clarity_value > self.laplacian_threshold
        )
        clarity_ratio = good_clarity_count / total_faces

        if pose_ratio < self.head_pose_required_ratio:
            self._log_head_pose_rejection(
                video_path,
                pose_ratio,
                poses,
                clarity_crop_sizes,
                total_faces,
                len(sampled_frames),
            )
            return video_path, False, None, 'head_pose_out_of_range'
        if clarity_ratio < self.laplacian_required_ratio:
            self._log_blurry_face_rejection(
                video_path,
                clarity_ratio,
                clarity_values,
                clarity_crop_sizes,
                total_faces,
                len(sampled_frames),
            )
            return video_path, False, None, 'blurry_face'

        return video_path, True, (pose_ratio, clarity_ratio, representative_embedding), None

    @staticmethod
    def _sample_indices(total_frames: int, rate: float, max_frames: int) -> List[int]:
        sample_count = max(1, min(max_frames, int(total_frames * rate)))
        step = max(total_frames // sample_count, 1)
        return list(range(0, total_frames, step))[:sample_count]

    @staticmethod
    def _batched(items: List[Any], batch_size: int) -> List[List[Any]]:
        return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]

    @staticmethod
    def _clip_bbox(bbox: np.ndarray, frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = np.asarray(bbox).reshape(-1)[:4].astype(int)
        h, w = frame_shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=int)

    def _select_frame_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> List[Tuple[Detection, np.ndarray]]:
        selected = []
        for detection in detections:
            bbox = self._clip_bbox(detection.bbox, frame.shape)
            if bbox is None:
                continue
            selected.append((detection, bbox))

        if not self.face_primary_only or len(selected) <= 1:
            return selected

        return [max(selected, key=lambda item: (self._bbox_area(item[1]), item[0].score))]

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> int:
        x1, y1, x2, y2 = bbox.tolist()
        return max(x2 - x1, 0) * max(y2 - y1, 0)

    def _log_head_pose_rejection(
        self,
        video_path: str,
        pose_ratio: float,
        poses: List[np.ndarray],
        crop_sizes: List[Tuple[int, int]],
        total_faces: int,
        sampled_frame_count: int,
    ) -> None:
        threshold = float(self.max_head_angle)
        good_pose_count = sum(
            1 for pose in poses
            if np.all(np.abs(pose) <= threshold)
        )
        logger.info(
            "Fine filter rejected %s for head_pose_out_of_range: "
            "pose_ratio=%.3f required_ratio=%.3f max_angle=%.2f "
            "good_poses=%d/%d sampled_frames=%d primary_only=%s "
            "pose=%s min_face_edge=%s",
            video_path,
            pose_ratio,
            float(self.head_pose_required_ratio),
            threshold,
            good_pose_count,
            total_faces,
            sampled_frame_count,
            self.face_primary_only,
            self._format_pose_stats(poses),
            self._format_numeric_stats([min(width, height) for width, height in crop_sizes]),
        )

    def _log_blurry_face_rejection(
        self,
        video_path: str,
        clarity_ratio: float,
        clarity_values: List[float],
        crop_sizes: List[Tuple[int, int]],
        total_faces: int,
        sampled_frame_count: int,
    ) -> None:
        min_face_edges = [min(width, height) for width, height in crop_sizes]
        logger.info(
            "Fine filter rejected %s for blurry_face: "
            "clarity_ratio=%.3f required_ratio=%.3f threshold=%.2f "
            "good_faces=%d/%d sampled_frames=%d primary_only=%s "
            "laplacian=%s min_face_edge=%s",
            video_path,
            clarity_ratio,
            float(self.laplacian_required_ratio),
            float(self.laplacian_threshold),
            sum(1 for value in clarity_values if value > self.laplacian_threshold),
            total_faces,
            sampled_frame_count,
            self.face_primary_only,
            self._format_numeric_stats(clarity_values),
            self._format_numeric_stats(min_face_edges),
        )

    @staticmethod
    def _format_pose_stats(poses: List[np.ndarray]) -> str:
        if not poses:
            return "n=0"
        arr = np.asarray(poses, dtype=np.float32).reshape((-1, 3))
        abs_arr = np.abs(arr)
        max_abs = np.max(abs_arr, axis=1)
        return (
            f"n={arr.shape[0]} "
            f"pitch_abs=({V2ContentFilter._format_numeric_stats(abs_arr[:, 0].tolist())}) "
            f"yaw_abs=({V2ContentFilter._format_numeric_stats(abs_arr[:, 1].tolist())}) "
            f"roll_abs=({V2ContentFilter._format_numeric_stats(abs_arr[:, 2].tolist())}) "
            f"max_abs=({V2ContentFilter._format_numeric_stats(max_abs.tolist())})"
        )

    @staticmethod
    def _format_numeric_stats(values: List[float]) -> str:
        if not values:
            return "n=0"
        arr = np.asarray(values, dtype=np.float32)
        p25, median, p75 = np.percentile(arr, [25, 50, 75])
        return (
            f"n={arr.size} min={float(np.min(arr)):.2f} p25={float(p25):.2f} "
            f"median={float(median):.2f} p75={float(p75):.2f} "
            f"max={float(np.max(arr)):.2f}"
        )

    @staticmethod
    def _make_arcface_crop(frame: np.ndarray, face_crop: np.ndarray, detection: Detection) -> np.ndarray:
        if detection.kps is not None:
            return face_align.norm_crop(frame, landmark=detection.kps, image_size=112)
        return cv2.resize(face_crop, (112, 112))

    @staticmethod
    def _mean_embedding(embeddings: np.ndarray) -> np.ndarray:
        embedding = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _get_batch_detector(self) -> BatchSCRFDDetector:
        if self._batch_detector is None:
            detection_cfg = self.fine_cfg.get('face_detection', {})
            self._batch_detector = BatchSCRFDDetector(
                model_path=detection_cfg.get(
                    'model_path',
                    'models/insightface/scrfd_320_batched/scrfd_10g_320_batch.onnx',
                ),
                det_thresh=float(detection_cfg.get('det_thresh', 0.5)),
                det_size=tuple(detection_cfg.get('det_size', [320, 320])),
                batch_size=self.batch_size,
            )
        return self._batch_detector

    def _get_landmark_pose(self) -> BatchLandmarkPose:
        if self._landmark_pose is None:
            self._landmark_pose = BatchLandmarkPose(
                model_path=self.fine_cfg.get('landmark_model_path', '/root/.insightface/models/buffalo_m/1k3d68.onnx'),
                batch_size=self.batch_size,
            )
        return self._landmark_pose

    def _update_feedback(self, fail_reason: str):
        if fail_reason in self.feedback['fail_reasons']:
            self.feedback['fail_reasons'][fail_reason] += 1
        else:
            self.feedback['fail_reasons'][fail_reason] = 1

    def _build_feedback_for_explorer(self, search_term: str) -> Dict[str, Any]:
        """Build feedback payload for Explorer."""
        total = self.feedback['total_processed']
        coarse_pass = self.feedback['coarse_pass']
        fine_pass = self.feedback['fine_pass']
        v2_pass_rate = fine_pass / total if total > 0 else 0
        return {
            'search_term': search_term,
            'v2_pass_rate': v2_pass_rate,
            'total_processed': total,
            'total_qualified': fine_pass,
            'fail_reasons': self.feedback['fail_reasons']
        }
