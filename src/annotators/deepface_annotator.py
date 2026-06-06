from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from annotators.base_annotator import BaseAnnotator
from core.models.annotation_models import FacialFeatures

DeepFace = None


class DeepFaceAnnotator(BaseAnnotator):
    """Label 1 annotator: age/race from DeepFace, hair color left for QwenVL."""

    VALID_RACES = {"asian", "indian", "black", "white", "middle eastern", "latino hispanic"}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sample_frames = int(config.get("sample_frames", 5))

    @property
    def label_name(self) -> str:
        return "facial_features"

    def load(self) -> None:
        global DeepFace
        if DeepFace is None:
            from deepface import DeepFace as ImportedDeepFace

            DeepFace = ImportedDeepFace
        self._loaded = True

    def annotate(self, video_path: str, **kwargs: Any) -> Optional[FacialFeatures]:
        frames = self.read_sampled_frames(video_path, int(kwargs.get("sample_frames", self.sample_frames)))
        analyses = []
        for frame in frames:
            result = self.analyze_image(frame)
            if result is not None:
                analyses.append(result)
        if not analyses:
            return None
        return self.aggregate_features(analyses)

    def analyze_image(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        if self._looks_blank(frame):
            return None
        if not self._loaded:
            self.load()
        if DeepFace is None:
            return None

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as handle:
            if not cv2.imwrite(handle.name, frame):
                return None
            try:
                raw = DeepFace.analyze(
                    img_path=handle.name,
                    actions=["age", "race"],
                    enforce_detection=False,
                )
            except Exception:
                return None
        return self.normalize_analysis(raw)

    @staticmethod
    def normalize_analysis(raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, list):
            if not raw:
                return None
            raw = raw[0]
        if not isinstance(raw, dict):
            return None

        age = raw.get("age")
        race = raw.get("dominant_race") or raw.get("race")
        if isinstance(race, dict):
            race = max(race, key=race.get) if race else None
        if age is None or race is None:
            return None

        try:
            age_value = int(round(float(age)))
        except (TypeError, ValueError):
            return None
        age_value = max(0, min(120, age_value))
        race_value = str(race).strip().lower()

        return {
            "age": age_value,
            "race": race_value,
            "confidence": {
                "age": raw.get("face_confidence", 1.0),
                "race": raw.get("race", {}),
            },
            "raw": raw,
        }

    @staticmethod
    def aggregate_features(analyses: List[Dict[str, Any]]) -> FacialFeatures:
        if not analyses:
            raise ValueError("at least one DeepFace analysis is required")

        ages = [int(item["age"]) for item in analyses if item.get("age") is not None]
        races = [str(item["race"]) for item in analyses if item.get("race")]
        if not ages or not races:
            raise ValueError("analyses must include age and race")

        confidence = {
            "age": {
                "samples": len(ages),
                "median": float(np.median(ages)),
            },
            "race": {
                "samples": len(races),
                "counts": dict(Counter(races)),
            },
        }
        return FacialFeatures(
            age=int(round(float(np.median(ages)))),
            race=Counter(races).most_common(1)[0][0],
            hair_color="unknown",
            confidence=confidence,
        )

    @staticmethod
    def read_sampled_frames(video_path: str, sample_frames: int = 5) -> List[np.ndarray]:
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
            for index in np.linspace(0, total_frames - 1, count, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        finally:
            cap.release()
        return frames

    @staticmethod
    def _looks_blank(frame: np.ndarray) -> bool:
        if frame is None or frame.size == 0:
            return True
        return float(np.std(frame)) < 1.0 or float(np.mean(frame)) < 5.0
