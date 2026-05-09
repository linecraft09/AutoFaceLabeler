from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD, distance2bbox, distance2kps

from aflutils.logger import get_logger
from .ort_utils import DEFAULT_SCRFD_BATCH_MODEL, create_session, resolve_model_path

logger = get_logger(__name__)


@dataclass
class Detection:
    """Single SCRFD face detection."""

    bbox: np.ndarray
    score: float
    kps: Optional[np.ndarray] = None


class BatchSCRFDDetector:
    """Batch SCRFD face detector backed by an ONNX Runtime session."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_SCRFD_BATCH_MODEL,
        det_thresh: float = 0.5,
        det_size: tuple[int, int] = (320, 320),
        batch_size: int = 16,
    ):
        self.model_path = resolve_model_path(model_path)
        self.det_thresh = float(det_thresh)
        self.det_size = tuple(int(item) for item in det_size)
        self.batch_size = int(batch_size)
        session = create_session(self.model_path)
        self.detector = SCRFD(str(self.model_path), session=session)
        self.detector.prepare(0, det_thresh=self.det_thresh, input_size=self.det_size)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detect faces for each frame in a batch."""
        if not frames:
            return []

        outputs, scales, frame_shapes = self._forward(frames)
        return self._postprocess(outputs, len(frames), scales, frame_shapes)

    def _preprocess(self, frames: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
        """Resize/pad frames and return an NCHW input tensor."""
        input_w, input_h = self.det_size
        det_imgs = []
        scales = []
        for frame in frames:
            im_ratio = float(frame.shape[0]) / frame.shape[1]
            model_ratio = float(input_h) / input_w
            if im_ratio > model_ratio:
                new_h = input_h
                new_w = int(new_h / im_ratio)
            else:
                new_w = input_w
                new_h = int(new_w * im_ratio)
            det_scale = float(new_h) / frame.shape[0]
            resized = cv2.resize(frame, (new_w, new_h))
            det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
            det_img[:new_h, :new_w, :] = resized
            det_imgs.append(det_img)
            scales.append(det_scale)

        blob = cv2.dnn.blobFromImages(
            det_imgs,
            1.0 / self.detector.input_std,
            self.det_size,
            (self.detector.input_mean, self.detector.input_mean, self.detector.input_mean),
            swapRB=True,
        )
        return blob, scales

    def _forward(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[float], list[tuple[int, int, int]]]:
        blob, scales = self._preprocess(frames)
        outputs = self.detector.session.run(self.detector.output_names, {self.detector.input_name: blob})
        for idx, output in enumerate(outputs):
            if output.shape[0] != len(frames):
                raise RuntimeError(f"SCRFD output[{idx}] is not batched: {output.shape}")
        return outputs, scales, [frame.shape for frame in frames]

    def _postprocess(
        self,
        net_outs: list[np.ndarray],
        batch_size: int,
        scales: list[float],
        frame_shapes: list[tuple[int, int, int]],
    ) -> list[list[Detection]]:
        """Decode SCRFD outputs and run NMS independently for each frame."""
        return [
            self._decode_one(net_outs, batch_index, scales[batch_index], frame_shapes[batch_index])
            for batch_index in range(batch_size)
        ]

    def _decode_one(
        self,
        outputs: list[np.ndarray],
        batch_index: int,
        det_scale: float,
        frame_shape: tuple[int, int, int],
    ) -> list[Detection]:
        input_w, input_h = self.det_size
        scores_list = []
        bboxes_list = []
        kpss_list = []

        fmc = self.detector.fmc
        for idx, stride in enumerate(self.detector._feat_stride_fpn):
            scores = outputs[idx][batch_index]
            bbox_preds = outputs[idx + fmc][batch_index] * stride
            if self.detector.use_kps:
                kps_preds = outputs[idx + fmc * 2][batch_index] * stride

            height = input_h // stride
            width = input_w // stride
            key = (height, width, stride)
            if key in self.detector.center_cache:
                anchor_centers = self.detector.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self.detector._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self.detector._num_anchors, axis=1).reshape((-1, 2))
                if len(self.detector.center_cache) < 100:
                    self.detector.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.detector.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds).reshape((-1, 5, 2))
                kpss_list.append(kpss[pos_inds])

        if sum(item.shape[0] for item in scores_list) == 0:
            return []

        scores = np.vstack(scores_list).ravel()
        order = scores.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale if self.detector.use_kps else None

        h, w = frame_shape[:2]
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)
        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.detector.nms(pre_det)
        det = pre_det[keep, :]
        if kpss is not None:
            kpss = kpss[order, :, :][keep, :, :]

        detections = []
        for idx, row in enumerate(det):
            detections.append(
                Detection(
                    bbox=row[:4].astype(np.float32),
                    score=float(row[4]),
                    kps=kpss[idx].astype(np.float32) if kpss is not None else None,
                )
            )
        return detections
