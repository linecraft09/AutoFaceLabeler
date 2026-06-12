from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from annotators.base_annotator import BaseAnnotator
from annotators.models.dwpose_wrapper import DWPoseWrapper
from core.models.annotation_models import ClarityScore
from validators.v2_models.face_quality import compute_laplacian_variance


class DWPoseAnnotator(BaseAnnotator):
    """Label 3 annotator: face-region clarity from DWPose face keypoints."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sample_frames = int(config.get("sample_frames", 8))
        self.crop_padding = int(config.get("crop_padding", 16))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.3))
        self.wrapper = DWPoseWrapper(
            model_path=config.get("model_path", "models/dwpose/dw-ll_ucoco_384.onnx"),
            providers=config.get("providers", ["CPUExecutionProvider"]),
        )

    @property
    def label_name(self) -> str:
        return "clarity"

    def load(self) -> None:
        self.wrapper.load()
        self._loaded = True

    def annotate(self, video_path: str, **kwargs: Any) -> ClarityScore:
        if not self._loaded:
            self.load()

        frames = self.read_sampled_frames(video_path, int(kwargs.get("sample_frames", self.sample_frames)))
        scores: List[float] = []
        for frame in frames:
            score = self.analyze_frame(frame)
            if score is not None:
                scores.append(score)
        return self.aggregate_clarity(scores, total_frames=len(frames))

    def analyze_frame(self, frame: np.ndarray) -> Optional[float]:
        keypoints = self.wrapper.predict(frame)
        face_keypoints = self.extract_face_keypoints(keypoints, self.confidence_threshold)
        bbox = self.face_bbox_from_keypoints(face_keypoints, frame.shape, self.crop_padding)
        if bbox is None:
            return None
        crop = self.crop_face_region(frame, bbox)
        if crop.size == 0:
            return None
        return self.compute_face_clarity(crop)

    @staticmethod
    def extract_face_keypoints(
        keypoints: np.ndarray,
        confidence_threshold: float = 0.3,
    ) -> np.ndarray:
        if keypoints is None or len(keypoints) == 0:
            return np.empty((0, 3), dtype=np.float32)
        face_keypoints = np.asarray(keypoints, dtype=np.float32)[:5]
        return face_keypoints[face_keypoints[:, 2] >= confidence_threshold]

    @staticmethod
    def face_bbox_from_keypoints(
        face_keypoints: np.ndarray,
        image_shape: tuple[int, ...],
        crop_padding: int = 16,
    ) -> Optional[tuple[int, int, int, int]]:
        if face_keypoints is None or len(face_keypoints) == 0:
            return None

        height, width = int(image_shape[0]), int(image_shape[1])
        xs = face_keypoints[:, 0]
        ys = face_keypoints[:, 1]
        if xs.size == 0 or ys.size == 0:
            return None

        x1 = max(0, int(np.floor(float(xs.min()))) - crop_padding)
        y1 = max(0, int(np.floor(float(ys.min()))) - crop_padding)
        x2 = min(width, int(np.ceil(float(xs.max()))) + crop_padding)
        y2 = min(height, int(np.ceil(float(ys.max()))) + crop_padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def crop_face_region(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def compute_face_clarity(face_crop: np.ndarray) -> float:
        return float(compute_laplacian_variance(face_crop))

    @staticmethod
    def aggregate_clarity(scores: List[float], total_frames: int) -> ClarityScore:
        detected = len(scores)
        if detected == 0:
            return ClarityScore(
                mean_clarity=0.0,
                median_clarity=0.0,
                std_clarity=0.0,
                min_clarity=0.0,
                max_clarity=0.0,
                face_detected_ratio=0.0,
                per_frame=[],
            )

        values = np.asarray(scores, dtype=np.float64)
        denominator = max(1, int(total_frames))
        return ClarityScore(
            mean_clarity=float(np.mean(values)),
            median_clarity=float(np.median(values)),
            std_clarity=float(np.std(values)),
            min_clarity=float(np.min(values)),
            max_clarity=float(np.max(values)),
            face_detected_ratio=float(detected / denominator),
            per_frame=[float(value) for value in scores],
        )

    @staticmethod
    def read_sampled_frames(video_path: str, sample_frames: int = 8) -> List[np.ndarray]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {video_path}")

        frames: List[np.ndarray] = []
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames <= 0:
                while len(frames) < sample_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                return frames

            count = max(1, min(sample_frames, total_frames))
            indices = np.linspace(0, total_frames - 1, count, dtype=int)
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        finally:
            cap.release()
        return frames
