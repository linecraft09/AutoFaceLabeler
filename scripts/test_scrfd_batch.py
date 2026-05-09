from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from batch_inference_common import (
    SCRFD_BATCH_MODEL,
    assert_first_dim,
    create_session,
    describe_onnx_model,
    describe_runtime_session,
    print_json,
    summarize_times,
    timed_ms,
)


def parse_batches(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def run(model_path: Path, batches: list[int]) -> dict:
    session = create_session(model_path)
    input_meta = session.get_inputs()[0]
    output_names = [item.name for item in session.get_outputs()]
    rng = np.random.default_rng(20260509)

    base = rng.normal(size=(1, 3, 320, 320)).astype(np.float32)
    single_outputs = session.run(output_names, {input_meta.name: base})
    assert_first_dim(single_outputs, 1, "single SCRFD")

    batch_results = []
    for batch_size in batches:
        tensor = rng.normal(size=(batch_size, 3, 320, 320)).astype(np.float32)
        tensor[0] = base[0]

        outputs, times = timed_ms(
            lambda: session.run(output_names, {input_meta.name: tensor}),
            warmup=1,
            runs=3,
        )
        assert_first_dim(outputs, batch_size, "batch SCRFD")

        max_abs_diff = 0.0
        for single, batched in zip(single_outputs, outputs):
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(single[0] - batched[0]))))
        if max_abs_diff > 5e-3:
            # SCRFD batch inference has ~1e-3 float precision differences due to
            # kernel fusion, memory alignment, and deterministic algo selection.
            # 5e-3 is the empirically determined acceptable tolerance.
            raise AssertionError(f"single vs batch[0] mismatch: max_abs_diff={max_abs_diff}")

        batch_results.append(
            {
                "batch_size": batch_size,
                "output_shapes": [list(output.shape) for output in outputs],
                "single_vs_batch0_max_abs_diff": max_abs_diff,
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
    parser.add_argument("--model", type=Path, default=SCRFD_BATCH_MODEL)
    parser.add_argument("--batches", default="1,4,8,16")
    args = parser.parse_args()

    try:
        print_json(run(args.model, parse_batches(args.batches)))
        return 0
    except Exception as exc:
        print(f"SCRFD batch test failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
