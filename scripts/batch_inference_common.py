from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import onnx
import onnxruntime as ort


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_MODEL_DIR = REPO_ROOT / "models" / "insightface" / "scrfd_320_batched"
SCRFD_BATCH_MODEL = BATCH_MODEL_DIR / "scrfd_10g_320_batch.onnx"
ARCFACE_BATCH_MODEL = BATCH_MODEL_DIR / "arcface_w600k_r50_batch.onnx"
LANDMARK_MODEL = Path("/root/.insightface/models/buffalo_m/1k3d68.onnx")
VIDEO_PATH = Path("/root/.openclaw/workspace/shared/qME8SXVdgWA_qualified.mkv")


def _ensure_cuda_lib_path() -> None:
    """Auto-detect and add CUDA runtime libraries to LD_LIBRARY_PATH.

    ONNX Runtime's CUDAExecutionProvider needs libcufft.so.11 and
    libcudnn.so.9 at load time, but conda pip-installed nvidia packages
    place them under the conda prefix, not in standard system paths.
    """
    searched = []
    prefix = os.environ.get("CONDA_PREFIX", "")
    if not prefix:
        # Try to infer from sys.executable
        import re
        import sys

        m = re.search(r"(/.*?/envs/[^/]+)", sys.executable)
        if m:
            prefix = m.group(1)

    if prefix:
        for python_version in ("3.10", "3.11", "3.12"):
            for package in ("cublas", "cuda_runtime", "cufft", "cudnn", "curand"):
                candidate = Path(prefix) / f"lib/python{python_version}/site-packages/nvidia/{package}/lib"
                searched.append(str(candidate))
                if candidate.is_dir():
                    current = os.environ.get("LD_LIBRARY_PATH", "")
                    path_str = str(candidate)
                    if path_str not in current.split(":"):
                        os.environ["LD_LIBRARY_PATH"] = f"{path_str}:{current}" if current else path_str

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")


def preferred_providers() -> list[str]:
    _ensure_cuda_lib_path()
    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def create_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    providers = preferred_providers()
    try:
        return ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        if "CUDAExecutionProvider" not in providers:
            raise
        LOGGER.warning("CUDAExecutionProvider initialization failed, falling back to CPU: %s", exc)
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _shape(value: Any) -> list[Any]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
    return dims


def describe_onnx_model(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        return {"path": str(model_path), "exists": False}

    model = onnx.load(str(model_path))
    return {
        "path": str(model_path),
        "exists": True,
        "size_mb": round(model_path.stat().st_size / 1024 / 1024, 2),
        "inputs": [{"name": item.name, "shape": _shape(item)} for item in model.graph.input],
        "outputs": [{"name": item.name, "shape": _shape(item)} for item in model.graph.output],
    }


def describe_runtime_session(session: ort.InferenceSession) -> dict[str, Any]:
    return {
        "actual_providers": session.get_providers(),
        "inputs": [{"name": item.name, "shape": item.shape} for item in session.get_inputs()],
        "outputs": [{"name": item.name, "shape": item.shape} for item in session.get_outputs()],
    }


def assert_first_dim(outputs: Iterable[np.ndarray], batch_size: int, label: str) -> None:
    for idx, output in enumerate(outputs):
        if output.shape[0] != batch_size:
            raise AssertionError(
                f"{label} output[{idx}] first dim mismatch: expected {batch_size}, got {output.shape}"
            )


def timed_ms(func: Callable[[], Any], warmup: int = 1, runs: int = 3) -> tuple[Any, list[float]]:
    result = None
    for _ in range(warmup):
        result = func()

    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        times.append((time.perf_counter() - start) * 1000.0)
    return result, times


def summarize_times(times: list[float]) -> dict[str, float]:
    return {
        "min_ms": round(float(np.min(times)), 3),
        "mean_ms": round(float(np.mean(times)), 3),
        "max_ms": round(float(np.max(times)), 3),
    }


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
