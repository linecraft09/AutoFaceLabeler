from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from annotators.base_annotator import BaseAnnotator
from core.models.annotation_models import ExpressionIntensity, FacialMotion

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - only used when dependency is absent
    OpenAI = None  # type: ignore[assignment]


class QwenVLAnnotator(BaseAnnotator):
    """Labels 2 and 4 annotator using DashScope's OpenAI-compatible Qwen-VL API."""

    SYSTEM_PROMPT = """你是专业面部分析师。观察帧序列，输出严格 JSON：
{
  "facial_motion": { "description": "中文描述", "key_movements": ["眨眼", "微笑"], "duration_category": "moderate" },
  "expression_intensity": { "score": 3, "rationale": "中文理由", "dominant_expressions": ["smile", "neutral"] },
  "hair_color": "black"
}
评分标准: 1=几乎无变化,2=轻微,3=中等,4=明显,5=极强烈"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "qwen-vl-max")
        self.base_url = config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.sample_count = int(config.get("sample_frames", 8))
        self.min_frames = int(config.get("min_frames", 4))
        self.max_retries = int(config.get("max_retries", 3))
        self.temperature = float(config.get("temperature", 0.0))

    @property
    def label_name(self) -> str:
        return "facial_analysis"

    def load(self) -> None:
        self._loaded = True

    def annotate(self, video_path: str, **kwargs: Any) -> Dict[str, Any]:
        frames = self.sample_frames(
            video_path,
            count=int(kwargs.get("sample_frames", self.sample_count)),
            min_frames=int(kwargs.get("min_frames", self.min_frames)),
        )
        if not os.environ.get("DASHSCOPE_API_KEY"):
            return self.default_result(api_error="missing_api_key")

        try:
            content = self.call_api(frames)
        except Exception as exc:
            return self.default_result(api_error=str(exc))
        return self.parse_response(content)

    def sample_frames(self, video_path: str, count: int = 8, min_frames: int = 4) -> List[np.ndarray]:
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
                while len(frames) < count:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
            else:
                target = min(count, total_frames)
                for index in np.linspace(0, total_frames - 1, target, dtype=int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame = cap.read()
                    if ok:
                        frames.append(frame)
        finally:
            cap.release()

        while frames and len(frames) < min_frames:
            frames.append(frames[-1].copy())
        return frames

    @staticmethod
    def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise ValueError("failed to encode frame as JPEG")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def build_messages(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "请分析这些按时间顺序排列的视频帧。"}
        ]
        for frame in frames:
            image_b64 = self.frame_to_base64(frame)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def call_api(self, frames: List[np.ndarray]) -> str:
        last_error: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            try:
                return self._single_api_call(frames)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _single_api_call(self, frames: List[np.ndarray]) -> str:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("missing_api_key")
        if OpenAI is None:
            raise ImportError("openai SDK is not installed")

        client = OpenAI(api_key=api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=self.build_messages(frames),
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return str(response.choices[0].message.content)

    @classmethod
    def parse_response(cls, content: str) -> Dict[str, Any]:
        try:
            payload = json.loads(cls.extract_json(content))
        except Exception:
            return cls.default_result(parse_error=True, raw_response=content)

        motion_payload = payload.get("facial_motion") or {}
        expression_payload = payload.get("expression_intensity") or payload.get("expression") or {}
        intensity = expression_payload.get("score", expression_payload.get("intensity", 1))
        intensity = cls.validate_intensity(intensity)

        motion = FacialMotion(
            description=str(motion_payload.get("description", "")),
            key_movements=list(motion_payload.get("key_movements") or []),
            duration_category=str(motion_payload.get("duration_category", "unknown")),
        )
        expression = ExpressionIntensity(
            intensity=intensity,
            rationale=str(expression_payload.get("rationale", "")),
            dominant_expressions=list(expression_payload.get("dominant_expressions") or ["neutral"]),
        )
        return {
            "facial_motion": motion,
            "expression": expression,
            "hair_color": str(payload.get("hair_color", "unknown")),
            "raw_response": payload,
        }

    @staticmethod
    def extract_json(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return text[start : end + 1]
        return text

    @staticmethod
    def validate_intensity(value: Any) -> int:
        intensity = int(value)
        if intensity < 1 or intensity > 5:
            raise ValueError("expression intensity must be between 1 and 5")
        return intensity

    @staticmethod
    def default_result(**extra: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "facial_motion": FacialMotion(
                description="",
                key_movements=[],
                duration_category="unknown",
            ),
            "expression": ExpressionIntensity(
                intensity=1,
                rationale="",
                dominant_expressions=["neutral"],
            ),
            "hair_color": "unknown",
        }
        result.update(extra)
        return result
