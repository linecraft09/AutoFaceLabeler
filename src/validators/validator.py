import os
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
from core.storage.video_store import VideoStore
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
        self.dedup_threshold = fine_cfg.get('dedup_threshold', 0.7)
        self.batch_size = int(fine_cfg.get('batch_size', 16))
        self._batch_detector = None
        self._landmark_pose = None

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
            pending = self.store.get_pending_videos(limit=self.config.get('pending_limit', 5))
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
                        self.store.update_status(
                            video_row['video_id'], video_row.get('platform', ''),
                            'v2_failed', f'v2_exception: {e}'
                        )
                    except Exception as status_e:
                        logger.error(f"Failed to update error status: {status_e}")

    def _process_video(self, video_row: Dict[str, Any]):
        video_id = video_row['video_id']
        platform = video_row['platform']
        file_path = video_row['file_path']
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Video file missing: {file_path}")
            self.store.update_status(video_id, platform, 'v2_failed', 'file_missing')
            return

        generated_clip_paths: List[str] = []
        try:
            coarse_result, coarse_fail_reason, clip_paths = self._coarse_filter(file_path)
            generated_clip_paths = clip_paths
            if not coarse_result:
                self.store.update_status(video_id, platform, 'v2_coarse_failed', coarse_fail_reason)
                self._update_feedback(coarse_fail_reason)
                logger.info(f"Video {video_id} doesn't pass coarse filter because of {coarse_fail_reason}")
                return
            self.feedback['coarse_pass'] += 1
            logger.info(f"Video {video_id} passed coarse filter")

            clip_path, fine_fail_reason = self._fine_filter(clip_paths)
            if not clip_path:
                self.store.update_status(video_id, platform, 'v2_fine_failed', fine_fail_reason)
                self._update_feedback(fine_fail_reason)
                logger.info(f"Video {video_id} doesn't pass fine filter because of {fine_fail_reason}")
                return
            self.feedback['fine_pass'] += 1
            logger.info(f"Video {video_id} passed fine filter")

            self.store.update_status(video_id, platform, 'v2_passed')
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
            VU.remove_videos(generated_clip_paths)

    def _coarse_filter(self, video_path: str) -> Tuple[bool, Optional[str], List[str]]:
        segments, total_single_secs = self.yolo.detect_single_person_segments(video_path)
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

    def _fine_filter(self, video_paths: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Process multiple clips in fine filtering and select the best candidate.
        """

        # Collect per-clip processing output.
        results = []  # (video_path, success, data, fail_reason)
        # data is (pose_ratio, clarity_ratio, representative_embedding) or None

        for video_path in video_paths:
            try:
                result = self._process_single_video(video_path)
                results.append(result)
            except Exception as e:
                logger.error(f"fine filter single video {video_path} cause exception: {e}")
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

        # Rank by weighted normalized duration/clarity/pose score.
        def normalize_values(values: List[float]) -> List[float]:
            if not values:
                return []
            min_v = min(values)
            max_v = max(values)
            if max_v == min_v:
                return [1.0 for _ in values]
            span = max_v - min_v
            return [(v - min_v) / span for v in values]

        durations = [v[4] for v in passed_candidates]
        clarities = [v[2] for v in passed_candidates]
        poses = [v[1] for v in passed_candidates]
        norm_durations = normalize_values(durations)
        norm_clarities = normalize_values(clarities)
        norm_poses = normalize_values(poses)

        score_map = {}
        ranking_cfg = self.config.get('fine', {}).get('ranking', {})
        w_duration = ranking_cfg.get('weight_duration', 0.45)
        w_clarity = ranking_cfg.get('weight_clarity', 0.35)
        w_pose = ranking_cfg.get('weight_pose', 0.20)
        for idx, candidate in enumerate(passed_candidates):
            score_map[candidate[0]] = (
                    w_duration * norm_durations[idx]
                    + w_clarity * norm_clarities[idx]
                    + w_pose * norm_poses[idx]
            )

        def score(v):
            return score_map[v[0]]

        passed_candidates.sort(key=score, reverse=True)

        # Ordered dedup and final candidate list.
        passed_videos = []
        for video_path, pose_ratio, clarity_ratio, embedding, duration_secs in passed_candidates:
            if self.face_embedder.is_duplicate(embedding, self.dedup_threshold):
                fail_reasons[video_path] = 'duplicate'
                continue
            passed_videos.append((video_path, pose_ratio, clarity_ratio, embedding, duration_secs))

        if not passed_videos:
            all_fail_reasons = '|'.join(fail_reasons.values())
            return None, all_fail_reasons

        best_video_path, best_pose, best_clarity, best_embedding, best_duration = passed_videos[0]

        target_duration = ranking_cfg.get('target_clip_duration', 30.0)
        video_duration = best_duration
        if video_duration >= target_duration:
            self.face_embedder.add_embedding(best_embedding)
            return best_video_path, None

        to_merge = [best_video_path]
        total_duration = video_duration
        for video_path, _, _, _, dur in passed_videos[1:]:
            to_merge.append(video_path)
            total_duration += dur
            if total_duration >= target_duration:
                break

        if total_duration < target_duration:
            fail_msg = f"insufficient_duration: total {total_duration:.2f}s < {target_duration}s"
            return None, fail_msg

        merged_path = VU.concat_videos(to_merge)
        if merged_path is None:
            return None, "concat_failed"

        self.face_embedder.add_embedding(best_embedding)
        return merged_path, None

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
        for frame, detections in zip(sampled_frames, detections_by_frame):
            for detection in detections:
                bbox = self._clip_bbox(detection.bbox, frame.shape)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox.tolist()
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                clarity_crops.append(face_crop)
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

        good_clarity_count = sum(
            1 for face_crop in clarity_crops
            if compute_laplacian_variance(face_crop) > self.laplacian_threshold
        )
        clarity_ratio = good_clarity_count / total_faces

        if pose_ratio < self.head_pose_required_ratio:
            return video_path, False, None, 'head_pose_out_of_range'
        if clarity_ratio < self.laplacian_required_ratio:
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
