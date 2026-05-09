# AFL Batch Inference Model Summary

Date: 2026-05-09

## Download Status

Target command:

```bash
huggingface-cli download alonsorobots/scrfd_320_batched \
  scrfd_10g_320_batch.onnx arcface_w600k_r50_batch.onnx \
  --local-dir models/insightface/scrfd_320_batched
```

Result: blocked by the current execution environment.

- `huggingface-cli` is not available in the shell.
- `/root/miniconda3/bin/conda run -n afl python -m pip install huggingface_hub` failed with `Operation not permitted` while connecting to the package index proxy.
- `wget https://huggingface.co/.../scrfd_10g_320_batch.onnx` failed with `Temporary failure in name resolution`.

The target local directory was created:

```text
models/insightface/scrfd_320_batched
```

## Model Inventory

| Model | Path | Status | Input shape | Output shape |
| --- | --- | --- | --- | --- |
| SCRFD batch detector | `models/insightface/scrfd_320_batched/scrfd_10g_320_batch.onnx` | missing, download blocked | not available | not available |
| ArcFace batch embedder | `models/insightface/scrfd_320_batched/arcface_w600k_r50_batch.onnx` | missing, download blocked | not available | not available |
| buffalo_m 3D landmark | `/root/.insightface/models/buffalo_m/1k3d68.onnx` | present, 136.95 MB | `data: [None, 3, 192, 192]` | `fc1: [1, 3309]` metadata; actual runtime supports `(N, 3309)` |

## Runtime Environment

`onnxruntime.get_available_providers()` returns:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

However, actual `InferenceSession(..., providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])` falls back to CPU. Details are recorded in `04-issues.md`.

