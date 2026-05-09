from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from insightface.data import get_object
from insightface.utils import face_align, transform

from .ort_utils import DEFAULT_LANDMARK_MODEL, create_session, resolve_model_path


class BatchLandmarkPose:
    """Batch landmark detection plus PnP-style head pose estimation."""

    def __init__(self, model_path: str | Path = DEFAULT_LANDMARK_MODEL, batch_size: int = 16):
        self.model_path = resolve_model_path(model_path)
        self.batch_size = int(batch_size)
        self.session = create_session(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.mean_lmk = get_object("meanshape_68.pkl")

    def estimate_poses_batch(self, face_crops: list[np.ndarray], bboxes: list[np.ndarray]) -> list[np.ndarray]:
        """Estimate [pitch, yaw, roll] for each face."""
        if not face_crops:
            return []
        if len(face_crops) != len(bboxes):
            raise ValueError("face_crops and bboxes must have the same length")

        poses: list[np.ndarray] = []
        for start in range(0, len(face_crops), self.batch_size):
            crops = face_crops[start : start + self.batch_size]
            boxes = bboxes[start : start + self.batch_size]
            aligned, inverse_mats = self._align_crops(crops, boxes)
            if not aligned:
                continue
            blob = cv2.dnn.blobFromImages(
                aligned,
                1.0,
                (192, 192),
                (0.0, 0.0, 0.0),
                swapRB=True,
            )
            raw = self.session.run(self.output_names, {self.input_name: blob})[0]
            if raw.shape[0] != len(aligned):
                raise RuntimeError(f"landmark output is not batched: {raw.shape}")
            for row, inverse in zip(raw, inverse_mats):
                pred = row.reshape((-1, 3)).astype(np.float32)[-68:, :]
                pred[:, 0:2] += 1.0
                pred[:, 0:2] *= 96.0
                pred[:, 2] *= 96.0
                pred = face_align.trans_points(pred, inverse)
                projection = transform.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
                _, rotation, _ = transform.P2sRt(projection)
                poses.append(np.array(transform.matrix2angle(rotation), dtype=np.float32))
        return poses

    @staticmethod
    def _align_crops(face_crops: list[np.ndarray], bboxes: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        aligned = []
        inverse_mats = []
        for image, bbox in zip(face_crops, bboxes):
            x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)[:4]
            width = max(float(x2 - x1), 1.0)
            height = max(float(y2 - y1), 1.0)
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            scale = 192.0 / (max(width, height) * 1.5)
            face_img, matrix = face_align.transform(image, center, 192, scale, 0)
            aligned.append(face_img)
            inverse_mats.append(cv2.invertAffineTransform(matrix))
        return aligned, inverse_mats
