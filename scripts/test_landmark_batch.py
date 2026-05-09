from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from insightface.data import get_object
from insightface.utils import transform

from batch_inference_common import (
    LANDMARK_MODEL,
    create_session,
    describe_onnx_model,
    describe_runtime_session,
    print_json,
    summarize_times,
    timed_ms,
)


def parse_batches(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def split_landmarks(raw_output: np.ndarray) -> list[np.ndarray]:
    landmarks = []
    for row in raw_output:
        pred = row.reshape((-1, 3)).astype(np.float32)
        pred[:, 0:2] += 1.0
        pred[:, 0:2] *= 96.0
        pred[:, 2] *= 96.0
        landmarks.append(pred)
    return landmarks


def estimate_pose(landmark: np.ndarray, mean_lmk: np.ndarray) -> np.ndarray:
    projection = transform.estimate_affine_matrix_3d23d(mean_lmk, landmark)
    _, rotation, _ = transform.P2sRt(projection)
    return np.array(transform.matrix2angle(rotation), dtype=np.float32)


def run(model_path: Path, batches: list[int]) -> dict:
    session = create_session(model_path)
    input_meta = session.get_inputs()[0]
    output_names = [item.name for item in session.get_outputs()]
    rng = np.random.default_rng(20260509)
    mean_lmk = get_object("meanshape_68.pkl")

    batch_results = []
    for batch_size in batches:
        tensor = rng.normal(size=(batch_size, 3, 192, 192)).astype(np.float32)
        outputs, times = timed_ms(
            lambda: session.run(output_names, {input_meta.name: tensor})[0],
            warmup=1,
            runs=3,
        )
        if outputs.shape != (batch_size, 3309):
            raise AssertionError(f"landmark output shape mismatch: expected {(batch_size, 3309)}, got {outputs.shape}")

        landmarks = split_landmarks(outputs)
        poses = [estimate_pose(item[-68:], mean_lmk) for item in landmarks]
        if any(item.shape != (68, 3) for item in (lmk[-68:] for lmk in landmarks)):
            raise AssertionError("landmark split did not produce 68x3 keypoints")

        batch_results.append(
            {
                "batch_size": batch_size,
                "output_shape": list(outputs.shape),
                "keypoints_per_face_shape": [68, 3],
                "first_pose_deg": poses[0].round(4).tolist(),
                "timing": summarize_times(times),
            }
        )

    return {
        "model": describe_onnx_model(model_path),
        "runtime": describe_runtime_session(session),
        "batches": batch_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=LANDMARK_MODEL)
    parser.add_argument("--batches", default="1,4,8,16")
    args = parser.parse_args()

    try:
        print_json(run(args.model, parse_batches(args.batches)))
        return 0
    except Exception as exc:
        print(f"Landmark batch test failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
