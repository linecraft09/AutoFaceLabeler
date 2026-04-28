# AFL Integration Test Report
Date: 2026-04-28
Branch: fix/p0-p1-batch
Test Config: config_test.yaml

## Run 1: Codex Sandbox (no network)

### Result: TIMEOUT (1213s)

## Pipeline Stages Exercised
| Stage | Executed | Errors |
|-------|----------|--------|
| Search | ✓ | 36 network/proxy failures during search attempts |
| V1 Pre-filter | ✗ | 0 |
| Download | ✗ | 0 |
| V2 Coarse (YOLO) | ✗ | 0 |
| V2 Fine (ArcFace, Pose, Speech) | ✗ | 0 |

## Runtime Stats
- Total elapsed: 1212.75s
- Peak CPU: 45.9%
- Peak GPU mem: N/A (no NVIDIA driver/device detected in environment)
- Videos searched: 0 successful result sets (36 search attempts)
- Videos downloaded: 0
- Videos qualified: 0

## Errors/Warnings Categorization
- `Environment/Dependency`:
  - `ModuleNotFoundError: addict` via eager `speaker_detector` import (fixed).
  - CUDA requested but unavailable in host (no NVIDIA driver), caused V2 init failure before fallback fix (fixed).
- `Network/Infra`:
  - Repeated yt-dlp proxy/network failures: `127.0.0.1:7890` unreachable (`Operation not permitted`), causing search retries and eventual timeout.
- `Config Schema Warnings` (pre-existing):
  - `ConfigLoader` warns on multiple `validator2.*` keys as unexpected; did not crash run.
- `Non-blocking environment warnings`:
  - matplotlib cache directory not writable (`/root/.config/matplotlib`).
  - albumentations online version check blocked by restricted network.

## Bugs Found & Fixed
| # | File | Issue | Fix | Test Added |
|---|------|-------|-----|------------|
| 1 | src/validators/validator.py | Optional speech dependency imported eagerly, crashing startup when `modelscope/addict` is absent even with `speech_required=false`. | Moved `SpeakerDetector` import inside the `if self.speech_required` block (lazy import). | tests/test_validator_optional_speech.py |
| 2 | src/validators/v2_models/yolo_detector.py, src/validators/v2_models/arcface_embedder.py | `device: cuda` hard-fails on machines without NVIDIA driver/GPU. | Added CUDA availability checks and automatic CPU fallback with warning logs. | tests/test_device_fallbacks.py |

## Run 2: Host (WSL2, with proxy & GPU)

### Result: TIMEOUT (1215s) — but 4/5 pipeline stages exercised

### Pipeline Stages Exercised
| Stage | Executed | Notes |
|-------|----------|-------|
| Search | ✓ (16x) | yt-dlp works via proxy; 3 terms cycled |
| V1 Pre-filter | ✓ (8x) | Correctly filters by metadata |
| Download | ✓ (8x) | Successful after V1 filtering |
| V2 Coarse (YOLO) | ✓ (3x) | First batch only; thread died silently |
| V2 Fine (ArcFace, Pose) | ✗ | Thread crash prevented fine from running |

### Bug Found & Fixed (Run 2)
| # | File | Issue | Fix | Test Added |
|---|------|-------|-----|------------|
| 3 | src/validators/validator.py `_run_loop()` | V2 filter daemon thread had no `try/except` around `_process_video`. Unhandled exception → entire thread exits silently → all subsequently downloaded videos left unprocessed. (Root cause: V2 coarse only ran 3x for 8 downloads) | Wrapped `_process_video` call in try/except; on exception log error and mark video as `v2_failed`, then continue processing remaining videos. | tests/test_v2_thread_resilience.py (2 tests) |

### Unit Test Results
```
49 passed, 0 failed (19.27s)
```

## Recommendations
1. **V2 fine never triggered** even on host — likely because V2 coarse threshold (single_person_threshold=0.5 in test config) still rejected all real-world videos. For smoke testing, consider temporarily lowering to 0.1 or using a synthetic test video with known single-person content.
2. For sandbox/CI integration tests, force `device=cpu` in config and use offline mock search results.
3. Add an offline integration mode with mocked/fixed search results so all stages can always be exercised even under network restrictions.
4. V2 filter thread resilience is now proven — but the coarse filter acceptance rate in test is 0%, suggesting either YOLO model issues or video content mismatch. Investigate with a controlled test video.

### Commit History
```
91fdb1b fix(validators): V2 filter thread daemon crash on per-video exception
292317b fix(integration): CUDA fallback, lazy speech import, integration harness
b9004fd fix(p1-batch3): ABC interface, config schema, ArcFace reuse, docstrings
80d9b6c fix(p1-batch2): YOLO batching, FAISS flush, thread safety, error handling
f371498 fix(p1-batch1): thread safety, config validation, error handling
a868ae1 fix(p0): critical bug fixes for pipeline, security, cross-platform
c50b203 fix(wsl): WSL2/Linux compatibility fixes + new tests
```
