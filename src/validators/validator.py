import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, List
from typing import Tuple, Optional

import cv2
import numpy as np

import aflutils.video_utils as VU
from aflutils.logger import get_logger
from core.storage.video_store import VideoStore
from .v2_models.arcface_embedder import ArcFaceEmbedder
from .v2_models.face_quality import compute_laplacian_variance
from .v2_models.speaker_detector import SpeakerDetector
from .v2_models.yolo_detector import YOLODetector

logger = get_logger(__name__)


class V2ContentFilter:

    def __init__(self, config: Dict[str, Any], video_store: VideoStore, explorer=None):
        self.config = config
        self.store = video_store
        self.explorer = explorer
        self.stop_event = threading.Event()
        self.thread = None

        yolo_cfg = config.get('coarse_filter', {})
        self.yolo = YOLODetector(
            model_path=yolo_cfg.get('model_path', 'yolov8n.pt'),
            device=yolo_cfg.get('device', 'cpu')
        )

        self.single_person_threshold = yolo_cfg.get('single_person_threshold', 0.8)
        self.audio_required = yolo_cfg.get('audio_required', True)

        fine_cfg = config.get('fine_filter', {})
        self.max_head_angle = fine_cfg.get('head_pose_max_angle', 10)
        self.head_pose_required_ratio = fine_cfg.get('head_pose_required_ratio', 0.8)
        self.laplacian_threshold = fine_cfg.get('laplacian_threshold', 100)
        self.laplacian_required_ratio = fine_cfg.get('laplacian_required_ratio', 0.8)
        self.dedup_threshold = fine_cfg.get('dedup_threshold', 0.7)

        self.known_embeddings = []
        self.max_head_angle = fine_cfg.get('head_pose_max_angle', 10)
        self.head_pose_required_ratio = fine_cfg.get('head_pose_required_ratio', 0.8)
        self.laplacian_threshold = fine_cfg.get('laplacian_threshold', 100)

        device = fine_cfg.get('device', 'cpu')
        # ArcFace
        self.face_embedder = ArcFaceEmbedder(
            device=device,
            db_path=fine_cfg.get('face_db_path', 'data/face_index.faiss')
        )

        self.speech_required = fine_cfg.get('speech_required', False)
        if self.speech_required:
            self.speaker_detector = SpeakerDetector()

        self.feedback = {
            'total_processed': 0,
            'coarse_pass': 0,
            'fine_pass': 0,
            'fail_reasons': {
                'no_single_person': 0,
                'no_audio': 0,
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
            pending = self.store.get_pending_videos(limit=5)
            if not pending:
                time.sleep(5)
                continue

            for video_row in pending:
                if self.stop_event.is_set():
                    break
                self._process_video(video_row)

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

        per_video_times = {}
        for video_path in video_paths:
            try:
                result = self._process_single_video(video_path)
                results.append(result)
            except Exception as e:
                logger.error(f"fine filter single video {video_path} cause exception: {e}")
                results.append((video_path, False, None, f"exception: {str(e)}"))

        single_video_total = sum(per_video_times.values())
        if per_video_times:
            top_slowest = sorted(per_video_times.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(
                f"fine_filter timing: processed {len(per_video_times)} clips, "
                f"single_video_total={single_video_total:.3f}s, "
                f"slowest={[(Path(p).name, round(t, 3)) for p, t in top_slowest]}"
            )

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
        w_duration = 0.45
        w_clarity = 0.35
        w_pose = 0.20
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

        target_duration = 30.0
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
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return video_path, False, None, 'no_frames'

        sample_count = max(1, min(200, int(total_frames * 0.1)))
        sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        sample_indices_set = set(int(i) for i in sample_indices.tolist())
        sample_count = len(sample_indices_set)

        good_pose_count = 0
        good_clarity_count = 0
        sampled_face_detected = 0
        representative_embedding = None

        if self.speech_required:
            # speaker_detector is used in read-only mode here.
            has_speech = self.speaker_detector.detect_speech_from_video(video_path)
            if not has_speech:
                cap.release()
                return video_path, False, None, 'no_speech'

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx not in sample_indices_set:
                continue

            face = self.face_embedder.extract(frame)
            if not face:
                continue

            sampled_face_detected += 1

            bbox = face.bbox.astype(int)
            face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if face_img.size == 0:
                continue

            pose_rad = face.pose
            roll = pose_rad[0]
            pitch = pose_rad[1]
            yaw = pose_rad[2]

            max_angle = self.max_head_angle
            if abs(yaw) <= max_angle and abs(pitch) <= max_angle and abs(roll) <= max_angle:
                good_pose_count += 1

            clarity = compute_laplacian_variance(face_img)
            if clarity > self.laplacian_threshold:
                good_clarity_count += 1

            if representative_embedding is None:
                representative_embedding = face.normed_embedding

        cap.release()

        if sampled_face_detected <= sample_count * 0.8:
            return video_path, False, None, 'no_face_detected'

        pose_ratio = good_pose_count / sampled_face_detected
        clarity_ratio = good_clarity_count / sampled_face_detected

        if pose_ratio < self.head_pose_required_ratio:
            return video_path, False, None, 'head_pose_out_of_range'
        if clarity_ratio < self.laplacian_required_ratio:
            return video_path, False, None, 'blurry_face'

        return video_path, True, (pose_ratio, clarity_ratio, representative_embedding), None

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
