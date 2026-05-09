from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from batch_inference_common import (
    ARCFACE_BATCH_MODEL,
    create_session,
    describe_onnx_model,
    describe_runtime_session,
    print_json,
    summarize_times,
    timed_ms,
)


def parse_batches(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def make_blob(images: list[np.ndarray]) -> np.ndarray:
    return cv2.dnn.blobFromImages(
        images,
        1.0 / 127.5,
        (112, 112),
        (127.5, 127.5, 127.5),
        swapRB=True,
    )


def run(model_path: Path, batches: list[int]) -> dict:
    session = create_session(model_path)
    input_meta = session.get_inputs()[0]
    output_names = [item.name for item in session.get_outputs()]
    rng = np.random.default_rng(20260509)

    batch_results = []
    for batch_size in batches:
        images = [
            rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]
        blob = make_blob(images)
        if blob.shape != (batch_size, 3, 112, 112):
            raise AssertionError(f"blobFromImages shape mismatch: {blob.shape}")

        outputs, times = timed_ms(
            lambda: session.run(output_names, {input_meta.name: blob})[0],
            warmup=1,
            runs=5,
        )
        if outputs.shape != (batch_size, 512):
            raise AssertionError(f"ArcFace output shape mismatch: expected {(batch_size, 512)}, got {outputs.shape}")

        batch_results.append(
            {
                "batch_size": batch_size,
                "blob_shape": list(blob.shape),
                "output_shape": list(outputs.shape),
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
    parser.add_argument("--model", type=Path, default=ARCFACE_BATCH_MODEL)
    parser.add_argument("--batches", default="1,4,8,16")
    args = parser.parse_args()

    try:
        print_json(run(args.model, parse_batches(args.batches)))
        return 0
    except Exception as exc:
        print(f"ArcFace batch test failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
