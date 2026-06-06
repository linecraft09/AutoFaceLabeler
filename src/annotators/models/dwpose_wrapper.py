from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import onnxruntime as ort

from validators.v2_models.ort_utils import resolve_model_path


class DWPoseWrapper:
    """Small ONNX Runtime wrapper for the DWPose whole-body SimCC model."""

    def __init__(
        self,
        model_path: str | Path = "models/dwpose/dw-ll_ucoco_384.onnx",
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        self.providers = list(providers or ["CPUExecutionProvider"])
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.input_height = 384
        self.input_width = 384

    def load(self) -> None:
        if self.session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"DWPose model not found: {self.model_path}")

        self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        if len(shape) == 4:
            self.input_height = int(shape[2]) if isinstance(shape[2], int) else self.input_height
            self.input_width = int(shape[3]) if isinstance(shape[3], int) else self.input_width

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Return the first 17 COCO keypoints as ``[x, y, confidence]``."""
        if self.session is None or self.input_name is None:
            self.load()
        assert self.session is not None
        assert self.input_name is not None

        original_height, original_width = frame.shape[:2]
        tensor = self._preprocess(frame)
        simcc_x, simcc_y = self.session.run(None, {self.input_name: tensor})
        return self.decode_simcc(
            simcc_x,
            simcc_y,
            original_size=(original_width, original_height),
            input_size=(self.input_width, self.input_height),
            num_keypoints=17,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

    @staticmethod
    def decode_simcc(
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        original_size: tuple[int, int],
        input_size: tuple[int, int],
        num_keypoints: int = 17,
    ) -> np.ndarray:
        """Decode SimCC logits to image-space keypoints.

        DWPose exports logits with x/y bins typically at 2x input resolution.
        Confidence is the lower of the normalized x/y peak probabilities.
        """
        x_logits = np.asarray(simcc_x)[0, :num_keypoints]
        y_logits = np.asarray(simcc_y)[0, :num_keypoints]

        x_idx = np.argmax(x_logits, axis=1).astype(np.float32)
        y_idx = np.argmax(y_logits, axis=1).astype(np.float32)
        x_conf = np.max(x_logits, axis=1).astype(np.float32)
        y_conf = np.max(y_logits, axis=1).astype(np.float32)

        input_width, input_height = input_size
        original_width, original_height = original_size
        x_split = max(1.0, x_logits.shape[1] / float(input_width))
        y_split = max(1.0, y_logits.shape[1] / float(input_height))

        xs = (x_idx / x_split) * (original_width / float(input_width))
        ys = (y_idx / y_split) * (original_height / float(input_height))
        confidence = np.minimum(x_conf, y_conf)

        keypoints = np.stack([xs, ys, confidence], axis=1).astype(np.float32)
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, max(0, original_width - 1))
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, max(0, original_height - 1))
        return keypoints
