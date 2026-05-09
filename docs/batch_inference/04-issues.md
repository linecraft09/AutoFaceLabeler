# AFL Batch Inference Issues

Date: 2026-05-09

## 1. HuggingFace Download Blocked

Symptoms:

- `huggingface-cli` is not installed.
- Installing `huggingface_hub` failed:

```text
ProxyError ... Failed to establish a new connection: [Errno 1] Operation not permitted
ERROR: No matching distribution found for huggingface_hub
```

- Direct `wget` failed:

```text
Temporary failure in name resolution
```

Impact:

- `scrfd_10g_320_batch.onnx` and `arcface_w600k_r50_batch.onnx` are not present.
- SCRFD/ArcFace batch tests and the end-to-end integration run cannot complete yet.

Resolution needed:

- Re-run the HuggingFace download command in an environment with DNS/network access, or place both ONNX files under `models/insightface/scrfd_320_batched/`.

## 2. CUDAExecutionProvider Listed but Not Actually Usable

Provider list:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Actual session creation falls back to CPU. First failure without extra library path:

```text
libcufft.so.11: cannot open shared object file
```

After adding local `nvidia/cufft` and `nvidia/cudnn` library paths, CUDA still fails:

```text
CUDA failure 35: CUDA driver version is insufficient for CUDA runtime version
```

Torch also reports:

```text
torch.cuda.is_available() = False
torch.version.cuda = 12.4
```

Impact:

- Current measured landmark timings are CPU timings, not RTX 4070 Ti timings.

Resolution needed:

- Align NVIDIA driver, CUDA runtime, cuDNN, and `onnxruntime-gpu` requirements for the `afl` conda environment.
- Ensure required NVIDIA Python package libraries are visible in `LD_LIBRARY_PATH`, then verify with an actual `InferenceSession(...).get_providers()` run, not only `onnxruntime.get_available_providers()`.

## 3. Landmark ONNX Metadata Shape Warning

The buffalo_m landmark model metadata declares:

```text
fc1: [1, 3309]
```

Runtime returns:

```text
(N, 3309)
```

for `N = 4, 8, 16`, and ONNX Runtime prints:

```text
Expected shape from model of {1,3309} does not match actual shape of {N,3309}
```

Impact:

- Batch inference works, but logs are noisy.
- Any future strict shape validator must rely on actual runtime output, not metadata alone.

## 4. Non-blocking Import Warnings

During InsightFace imports:

- Albumentations version check cannot access the network.
- Matplotlib config/cache directory under `/root/.config/matplotlib` is not writable.
- Fontconfig reports no writable cache directories.

Impact:

- No observed functional failure in the batch scripts.
- The warnings add noise to command output.

