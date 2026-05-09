# AFL Batch Pipeline Integration Test

Date: 2026-05-09

## Script Created

`tests/test_batch_pipeline.py`

The script implements the planned `_process_single_video` batch flow without modifying `src/`:

1. Seek sampled frames directly with `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)`.
2. Split sampled frames by `batch_size`.
3. Run SCRFD batch detection and compute face-frame ratio.
4. Run batch ArcFace embedding for detected faces.
5. Run batch landmark inference and InsightFace PnP pose solve.
6. Run per-face CPU Laplacian clarity.
7. Return pass/fail plus `pose_ratio`, `clarity_ratio`, and representative embedding shape.

## Config Used

From `config/config.yaml`:

- `v2_filter.fine.sampling.rate`: `0.1`
- `v2_filter.fine.sampling.max_frames`: `200`
- `v2_filter.fine.batch_size`: `16`
- `v2_filter.fine.face_detection.det_size`: `[320, 320]`
- `v2_filter.fine.face_detection.det_thresh`: `0.5`

## Video Probe

Video:

```text
/root/.openclaw/workspace/shared/qME8SXVdgWA_qualified.mkv
```

OpenCV metadata:

| Field | Value |
| --- | --- |
| Opened | true |
| Total frames | 790 |
| FPS | 25.0 |
| Size | 1920 x 1080 |

Sampling count from current config:

```text
min(200, int(790 * 0.1)) = 79
```

## Run Results

Command-line run:

```bash
/root/miniconda3/bin/conda run -n afl python tests/test_batch_pipeline.py
```

Result: blocked at model preflight.

```text
required file not found: models/insightface/scrfd_320_batched/scrfd_10g_320_batch.onnx
```

Pytest smoke:

```bash
/root/miniconda3/bin/conda run -n afl python -m pytest -q tests/test_batch_pipeline.py
```

Result:

```text
1 skipped, 1 warning
```

The skip is intentional while SCRFD/ArcFace batch models are unavailable.

## Current Pipeline Comparison

The existing `_process_single_video` output was not run for comparison in this preparation step because the requested batch models are unavailable and the existing path may write/update FAISS state. The integration script is ready to produce the comparison inputs once the models are present.

