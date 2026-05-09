from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BATCH_MODEL_DIR = REPO_ROOT / "models" / "insightface" / "scrfd_320_batched"
DEFAULT_SCRFD_BATCH_MODEL = DEFAULT_BATCH_MODEL_DIR / "scrfd_10g_320_batch.onnx"
DEFAULT_ARCFACE_BATCH_MODEL = DEFAULT_BATCH_MODEL_DIR / "arcface_w600k_r50_batch.onnx"
DEFAULT_LANDMARK_MODEL = Path("/root/.insightface/models/buffalo_m/1k3d68.onnx")


def resolve_model_path(model_path: str | Path) -> Path:
    """Resolve absolute and repository-relative model paths."""
    path = Path(model_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _ensure_cuda_lib_path() -> None:
    """Add conda-packaged CUDA runtime libraries to LD_LIBRARY_PATH."""
    prefix = os.environ.get("CONDA_PREFIX", "")
    if not prefix:
        match = re.search(r"(/.*?/envs/[^/]+)", sys.executable)
        if match:
            prefix = match.group(1)
    if not prefix:
        return

    for python_version in ("3.10", "3.11", "3.12"):
        for package in ("cublas", "cuda_runtime", "cufft", "cudnn", "curand"):
            candidate = Path(prefix) / f"lib/python{python_version}/site-packages/nvidia/{package}/lib"
            if not candidate.is_dir():
                continue
            current = os.environ.get("LD_LIBRARY_PATH", "")
            path_str = str(candidate)
            if path_str not in current.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{path_str}:{current}" if current else path_str

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")


def preferred_providers() -> list[str]:
    """Return ONNX Runtime providers, preferring CUDA when available."""
    _ensure_cuda_lib_path()
    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def create_session(model_path: str | Path) -> ort.InferenceSession:
    """Create an ORT session with CUDA preference and CPU fallback."""
    path = resolve_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    providers = preferred_providers()
    try:
        return ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:
        if "CUDAExecutionProvider" not in providers:
            raise
        logger.warning("CUDAExecutionProvider failed for %s; falling back to CPU: %s", path, exc)
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
