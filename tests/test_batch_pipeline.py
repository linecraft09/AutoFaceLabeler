from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from insightface.data import get_object
from insightface.model_zoo.scrfd import SCRFD, distance2bbox, distance2kps
from insightface.utils import face_align, transform

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from batch_inference_common import (  # noqa: E402
    ARCFACE_BATCH_MODEL,
    LANDMARK_MODEL,
    SCRFD_BATCH_MODEL,
    VIDEO_PATH,
    preferred_providers,
)


@dataclass
class Detection:
    frame_index: int
    bbox: np.ndarray
    score: float
    kps: np.ndarray | None
    frame: np.ndarray


class BatchSCRFD:
    def __init__(self, model_path: Path, det_thresh: float = 0.5, det_size: tuple[int, int] = (320, 320)):
        self.detector = SCRFD(str(model_path), session=ort.InferenceSession(str(model_path), providers=preferred_providers()))
        self.detector.prepare(0, det_thresh=det_thresh, input_size=det_size)
        self.det_size = det_size
        self.det_thresh = det_thresh

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        input_w, input_h = self.det_size
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
        return det_img, det_scale

    def _decode_one(
        self,
        outputs: list[np.ndarray],
        batch_index: int,
        det_scale: float,
        frame_shape: tuple[int, int, int],
        max_num: int = 1,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        input_w, input_h = self.det_size
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self.detector._feat_stride_fpn):
            scores = outputs[idx][batch_index]
            bbox_preds = outputs[idx + self.detector.fmc][batch_index] * stride
            if self.detector.use_kps:
                kps_preds = outputs[idx + self.detector.fmc * 2][batch_index] * stride

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
                self.detector.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.detector.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds).reshape((-1, 5, 2))
                kpss_list.append(kpss[pos_inds])

        if not scores_list or sum(item.shape[0] for item in scores_list) == 0:
            return np.empty((0, 5), dtype=np.float32), None

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

        if max_num > 0 and det.shape[0] > max_num:
            det = det[:max_num]
            if kpss is not None:
                kpss = kpss[:max_num]
        return det, kpss

    def detect_batch(self, frames: list[tuple[int, np.ndarray]]) -> list[Detection]:
        det_imgs = []
        scales = []
        for _, frame in frames:
            det_img, det_scale = self._preprocess(frame)
            det_imgs.append(det_img)
            scales.append(det_scale)

        blob = cv2.dnn.blobFromImages(
            det_imgs,
            1.0 / self.detector.input_std,
            self.det_size,
            (self.detector.input_mean, self.detector.input_mean, self.detector.input_mean),
            swapRB=True,
        )
        outputs = self.detector.session.run(self.detector.output_names, {self.detector.input_name: blob})
        for idx, output in enumerate(outputs):
            if output.shape[0] != len(frames):
                raise AssertionError(f"SCRFD output[{idx}] is not batched: {output.shape}")

        detections: list[Detection] = []
        for batch_index, ((frame_index, frame), det_scale) in enumerate(zip(frames, scales)):
            det, kpss = self._decode_one(outputs, batch_index, det_scale, frame.shape)
            if det.shape[0] == 0:
                continue
            detections.append(
                Detection(
                    frame_index=frame_index,
                    bbox=det[0, :4].astype(np.float32),
                    score=float(det[0, 4]),
                    kps=kpss[0].astype(np.float32) if kpss is not None else None,
                    frame=frame,
                )
            )
        return detections


def load_config() -> dict[str, Any]:
    with (REPO_ROOT / "config" / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sampled_indices(total_frames: int, rate: float, max_frames: int) -> list[int]:
    sample_count = max(1, min(max_frames, int(total_frames * rate)))
    return sorted(set(np.linspace(0, total_frames - 1, sample_count, dtype=int).tolist()))


def read_sampled_frames(video_path: Path, indices: list[int]) -> tuple[list[tuple[int, np.ndarray]], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append((int(index), frame))
        meta = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        return frames, meta
    finally:
        cap.release()


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def run_arcface(detections: list[Detection], model_path: Path, batch_size: int) -> tuple[np.ndarray | None, float]:
    if not detections:
        return None, 0.0
    session = ort.InferenceSession(str(model_path), providers=preferred_providers())
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]

    all_embeddings = []
    start = time.perf_counter()
    for group in batched(detections, batch_size):
        aligned = []
        for det in group:
            if det.kps is not None:
                face = face_align.norm_crop(det.frame, landmark=det.kps, image_size=112)
            else:
                x1, y1, x2, y2 = det.bbox.astype(int).tolist()
                face = cv2.resize(det.frame[y1:y2, x1:x2], (112, 112))
            aligned.append(face)
        blob = cv2.dnn.blobFromImages(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        embeddings = session.run(output_names, {input_name: blob})[0]
        if embeddings.shape != (len(group), 512):
            raise AssertionError(f"ArcFace output shape mismatch: {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        all_embeddings.append(embeddings / np.maximum(norms, 1e-12))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return np.vstack(all_embeddings)[0], elapsed_ms


def run_landmark(detections: list[Detection], model_path: Path, batch_size: int) -> tuple[list[np.ndarray], float]:
    if not detections:
        return [], 0.0
    session = ort.InferenceSession(str(model_path), providers=preferred_providers())
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]
    mean_lmk = get_object("meanshape_68.pkl")

    poses = []
    start = time.perf_counter()
    for group in batched(detections, batch_size):
        aligned = []
        inverse_mats = []
        for det in group:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            scale = 192.0 / (max(width, height) * 1.5)
            face_img, matrix = face_align.transform(det.frame, center, 192, scale, 0)
            aligned.append(face_img)
            inverse_mats.append(cv2.invertAffineTransform(matrix))
        # The 1k3d68.onnx landmark model has input_mean=0.0 and input_std=1.0
        # (from model metadata), meaning raw [0,255] pixel values with BGR→RGB
        # swap but NO mean subtraction or scaling.
        # The detection/recognition models use (pixel-127.5)/128.0, but landmark
        # uses raw pixel values — using the wrong normalization completely
        # breaks pose estimation (e.g., 168° instead of -7°).
        blob = cv2.dnn.blobFromImages(aligned, 1.0, (192, 192), (0.0, 0.0, 0.0), swapRB=True)
        raw = session.run(output_names, {input_name: blob})[0]
        if raw.shape != (len(group), 3309):
            raise AssertionError(f"landmark output shape mismatch: {raw.shape}")
        for row, inverse in zip(raw, inverse_mats):
            pred = row.reshape((-1, 3)).astype(np.float32)
            pred = pred[-68:, :]
            pred[:, 0:2] += 1
            pred[:, 0:2] *= 96
            pred[:, 2] *= 96
            pred = face_align.trans_points(pred, inverse)
            projection = transform.estimate_affine_matrix_3d23d(mean_lmk, pred)
            _, rotation, _ = transform.P2sRt(projection)
            poses.append(np.array(transform.matrix2angle(rotation), dtype=np.float32))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return poses, elapsed_ms


def laplacian_variance(face_img: np.ndarray) -> float:
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run_pipeline() -> dict[str, Any]:
    config = load_config()
    fine = config["v2_filter"]["fine"]
    sampling = fine["sampling"]
    batch_size = int(fine.get("batch_size", 16))
    face_detection = fine.get("face_detection", {})
    det_model = REPO_ROOT / face_detection.get("model_path", str(SCRFD_BATCH_MODEL))
    arcface_model = ARCFACE_BATCH_MODEL

    for path in [det_model, arcface_model, LANDMARK_MODEL, VIDEO_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"required file not found: {path}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = sampled_indices(total_frames, float(sampling["rate"]), int(sampling["max_frames"]))

    timings: dict[str, float] = {}
    start = time.perf_counter()
    frames, video_meta = read_sampled_frames(VIDEO_PATH, indices)
    timings["read_sampled_frames_ms"] = round((time.perf_counter() - start) * 1000.0, 3)

    detector = BatchSCRFD(
        det_model,
        det_thresh=float(face_detection.get("det_thresh", 0.5)),
        det_size=tuple(face_detection.get("det_size", [320, 320])),
    )
    detections: list[Detection] = []
    start = time.perf_counter()
    for group in batched(frames, batch_size):
        detections.extend(detector.detect_batch(group))
    timings["scrfd_batch_detect_ms"] = round((time.perf_counter() - start) * 1000.0, 3)

    face_ratio = len(detections) / max(len(frames), 1)
    if face_ratio <= 0.8:
        return {
            "status": "fail",
            "reason": "no_face_detected",
            "video": video_meta,
            "sampled_frames": len(frames),
            "face_frames": len(detections),
            "face_ratio": face_ratio,
            "timings": timings,
        }

    embedding, arcface_ms = run_arcface(detections, arcface_model, batch_size)
    timings["arcface_batch_embed_ms"] = round(arcface_ms, 3)

    poses, landmark_ms = run_landmark(detections, LANDMARK_MODEL, batch_size)
    timings["landmark_batch_pose_ms"] = round(landmark_ms, 3)

    max_angle = float(fine["head_pose"]["max_angle"])
    good_pose = sum(1 for pose in poses if np.all(np.abs(pose) <= max_angle))
    pose_ratio = good_pose / max(len(poses), 1)

    clarity_values = []
    start = time.perf_counter()
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int).tolist()
        crop = det.frame[y1:y2, x1:x2]
        if crop.size:
            clarity_values.append(laplacian_variance(crop))
    timings["laplacian_cpu_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
    clarity_ratio = sum(value > float(fine["laplacian"]["threshold"]) for value in clarity_values) / max(len(detections), 1)

    status = "pass"
    reason = None
    if pose_ratio < float(fine["head_pose"]["required_ratio"]):
        status = "fail"
        reason = "head_pose_out_of_range"
    elif clarity_ratio < float(fine["laplacian"]["required_ratio"]):
        status = "fail"
        reason = "blurry_face"

    return {
        "status": status,
        "reason": reason,
        "video": video_meta,
        "sampled_frames": len(frames),
        "sampled_indices": indices,
        "face_frames": len(detections),
        "face_ratio": face_ratio,
        "pose_ratio": pose_ratio,
        "clarity_ratio": clarity_ratio,
        "embedding_shape": list(embedding.shape) if embedding is not None else None,
        "timings": timings,
    }


def test_batch_pipeline_smoke() -> None:
    missing = [path for path in [SCRFD_BATCH_MODEL, ARCFACE_BATCH_MODEL, LANDMARK_MODEL, VIDEO_PATH] if not path.exists()]
    if missing:
        import pytest

        pytest.skip("missing required files: " + ", ".join(str(item) for item in missing))
    result = run_pipeline()
    assert result["sampled_frames"] > 0
    assert result["status"] in {"pass", "fail"}


if __name__ == "__main__":
    try:
        print(json.dumps(run_pipeline(), indent=2, ensure_ascii=False, default=str))
    except Exception as exc:
        print(f"batch pipeline test failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
