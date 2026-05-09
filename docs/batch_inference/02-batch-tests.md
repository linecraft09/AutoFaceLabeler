# AFL Batch Model Tests

Date: 2026-05-09

All Python commands were run through:

```bash
/root/miniconda3/bin/conda run -n afl python ...
```

## Scripts Created

- `scripts/test_scrfd_batch.py`
- `scripts/test_arcface_batch.py`
- `scripts/test_landmark_batch.py`
- `scripts/batch_inference_common.py`

## SCRFD Batch Detection

Command:

```bash
/root/miniconda3/bin/conda run -n afl python scripts/test_scrfd_batch.py
```

Result: blocked because the model file is missing.

```text
model not found: models/insightface/scrfd_320_batched/scrfd_10g_320_batch.onnx
```

The script is ready to test batch sizes `1,4,8,16`, validate that every output has batch dimension `N`, compare single-frame output with the same frame at `batch[0]`, and report per-batch timings.

## ArcFace Batch Embedding

Command:

```bash
/root/miniconda3/bin/conda run -n afl python scripts/test_arcface_batch.py
```

Result: blocked because the model file is missing.

```text
model not found: models/insightface/scrfd_320_batched/arcface_w600k_r50_batch.onnx
```

The script is ready to validate `cv2.dnn.blobFromImages` output shape `(N, 3, 112, 112)`, ArcFace output shape `(N, 512)`, and per-batch timings for `1,4,8,16`.

## Landmark Batch Pose

Command:

```bash
/root/miniconda3/bin/conda run -n afl python scripts/test_landmark_batch.py
```

Result: passed on CPU fallback.

| Batch | Output shape | Keypoint split | Mean latency |
| --- | --- | --- | --- |
| 1 | `(1, 3309)` | `(68, 3)` | 8.770 ms |
| 4 | `(4, 3309)` | `(68, 3)` | 30.034 ms |
| 8 | `(8, 3309)` | `(68, 3)` | 60.101 ms |
| 16 | `(16, 3309)` | `(68, 3)` | 124.263 ms |

PnP pose solve was tested through InsightFace imports:

```python
from insightface.utils import transform
transform.estimate_affine_matrix_3d23d(...)
transform.P2sRt(...)
transform.matrix2angle(...)
```

Note: the model metadata declares output `[1, 3309]`, but runtime returns `(N, 3309)` for larger batches. ONNX Runtime prints a shape warning for `N > 1`, but inference succeeds.

