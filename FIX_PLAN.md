# AFL ArcFace Embedder Bug — Fix Plan

## Bug Summary

Production pipeline crashes with `ValueError: input not a numpy array` in
`arcface_embedder.py:120` (`add_embedding`). This occurs AFTER the fine filter
has successfully selected clips — the selected embeddings fail when being added
to the FAISS index.

---

## Root Cause

**FAISS 1.9.0 `_swigfaiss.so` incompatible with NumPy ≥ 2.0.**

FAISS's `swig_ptr()` function reads `__array_interface__` from numpy arrays to
obtain a raw C pointer to the underlying data. NumPy 2.0 changed the
`__array_interface__` version from 2 to 3, and FAISS 1.9.0's compiled C extension
cannot handle version 3.

When any FAISS operation that takes a numpy array is called —
including `index.add()`, `index.search()`, `index.train()` — the C extension
raises `ValueError: input not a numpy array` because it fails to parse the
`__array_interface__` dict.

Since `opencv-python 4.13.0` requires `numpy>=2`, downgrading numpy alone is
not viable (it breaks OpenCV).

### Detailed Code Flow

1. `V2ContentFilter._process_single_video()` calls
   `self.face_embedder.embed_crops_batch(arcface_crops)` → returns `np.ndarray [N, 512]`
2. `V2ContentFilter._mean_embedding(embeddings)` → returns normalized `np.ndarray (512,)`
3. Result tuple `(pose_ratio, clarity_ratio, representative_embedding)` returned
4. In `_fine_filter()`, embeddings are collected into `selected_embeddings`
5. `self.face_embedder.add_embedding(embedding)` calls `faiss.IndexFlatL2.add(embedding.reshape(1, -1))`
6. FAISS's `swig_ptr()` → **crash** with `ValueError: input not a numpy array`

### Which Operations Are Affected

All FAISS operations that receive a numpy array through `swig_ptr()`:
- `index.add()` — **the crash site**
- `index.search()` — (`is_duplicate` would also crash if reached)
- `index.train()`
- `faiss.write_index()` — involves internal numpy usage
- `faiss.read_index()` — reads from file path, unaffected by this bug

### Why It Only Fails Sometimes

The bug is **deterministic** — *every* video that reaches `add_embedding()` will crash.
The pipeline just hasn't processed many videos that pass the fine filter yet
(10 affected so far). The next such video will also crash.

---

## Verified Fix (in Dev)

Upgrade `faiss-cpu` to a version that supports NumPy 2.x.

**Confirmed working configuration:**
| Package | Version | Notes |
|---------|---------|-------|
| `faiss-cpu` | **1.13.2** | Latest stable; confirmed compatible |
| `numpy` | **2.2.6** | Current installed version — no downgrade needed |
| `opencv-python` | **4.13.0** | Works with numpy 2.x, unchanged |
| `onnxruntime-gpu` | **1.23.2** | Unchanged |

Full end-to-end test passed:
- `ArcFaceEmbedder.embed_crops_batch()` → ONNX session → normalized embeddings
- `V2ContentFilter._mean_embedding()` → mean + L2 normalization
- `ArcFaceEmbedder.add_embedding()` → FAISS `index.add()`
- `ArcFaceEmbedder.is_duplicate()` → FAISS `index.search()`
- `ArcFaceEmbedder.flush()` → FAISS `write_index()`
- FAISS index save/load round-trip

---

## Fix Options

### Option A: Upgrade faiss-cpu (RECOMMENDED)

```bash
# In the afl conda environment:
pip install "faiss-cpu>=1.10.0"
```

- **No code changes required.** All FAISS APIs remain identical.
- FAISS index files are binary compatible (no re-indexing needed).
- Removes the duplicate `faiss` (Facebook) package if present.
- **Estimated scope:** ~2 minutes (one pip command + verify).
- **Risk:** Low. API is stable across 1.9.x → 1.13.x.

### Option B: Pin to faiss-cpu 1.13.x (SAFEST)

```bash
pip install "faiss-cpu==1.13.2"
```

Same as Option A but version-pinned. Slightly more predictable across env rebuilds.

### Option C: Lock dependencies (for Docker/CI)

Add to `requirements.txt` or conda `environment.yml`:
```
faiss-cpu>=1.10.0,<2
numpy>=2,<3        # required by opencv-python 4.13.x
```

---

## Code-Level Workaround (Not Recommended)

A code-level fix is **not possible** for FAISS 1.9.0 because the bug is in the
compiled C extension (`_swigfaiss.so`), not in the Python wrapper. Adding
`embedding.copy()` or `np.ascontiguousarray()` before passing to FAISS would
not help — the C extension still rejects numpy 2.x `__array_interface__`.

---

## Potential Surprises

1. **Duplicate `faiss` package**: The production env may have both `faiss` (Facebook)
   and `faiss-cpu` installed. They share the same `faiss/` namespace. Clean up by
   uninstalling the one not used: `pip uninstall -y faiss` (the Facebook package)
   before installing `faiss-cpu>=1.10.0`.

2. **Index file compatibility**: FAISS `IndexFlatL2` binary format is stable
   across versions. Existing `.faiss` index files will load correctly after upgrade.

3. **SQLite-stored embeddings**: Stored as raw bytes via `.tobytes()` and loaded
   via `np.frombuffer(...).copy().reshape(512,)`. This is numpy-version-independent.

4. **Thread safety check**: The pipeline runs in a single thread (the `_run_loop`
   daemon thread). No multiprocessing issues to worry about.

---

## Steps to Apply Fix

```bash
# 1. Activate the conda environment
source /root/miniconda3/envs/afl/bin/activate

# 2. Remove the duplicate 'faiss' package (if present)
pip uninstall -y faiss

# 3. Install the fixed faiss-cpu
pip install "faiss-cpu>=1.10.0"

# 4. Verify
python -c "
import faiss; print('faiss:', faiss.__version__)
import numpy as np
idx = faiss.IndexFlatL2(512)
idx.add(np.random.randn(1, 512).astype(np.float32))
print('FAISS add: OK')
"

# 5. Run production pipeline normally
# (the fix takes effect immediately, no source code changes needed)
```

---

## Investigation Details

### Location
- Development: `/root/codex_workspace/AutoFaceLabeler/`
- Production: `/root/AutoFaceLabeler/` (DO NOT MODIFY without approval)

### Files Read
- `src/validators/v2_models/arcface_embedder.py` (full source)
- `src/validators/validator.py` (full source, specifically `_fine_filter`, `_process_single_video`, `_mean_embedding`)
- `src/core/storage/video_store.py` (embedding persist/load APIs)

### Confirmed Not the Cause
- Multiprocessing serialization (pipeline is single-threaded)
- None embeddings (guarded by shape checks)
- PyTorch tensor vs numpy (ONNX Runtime always returns numpy arrays)
- `_mean_embedding` edge cases (handles zeros, all-same, single inputs)

### Proposed By
Subagent investigation, 2026-05-16
