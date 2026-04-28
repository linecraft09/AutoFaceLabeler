# AFL Integration Test Report
Date: 2026-04-28
Branch: fix/p0-p1-batch
Test Config: config_test.yaml

## Result: TIMEOUT

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

## Recommendations
1. For sandbox/CI integration tests, force test config devices to CPU (`validator2.coarse_filter.device=cpu`, `validator2.fine_filter.device=cpu`) to avoid GPU dependency variance.
2. In test/runtime environment, unset broken proxy env vars (or provide reachable proxy) so yt-dlp can return data; current timeout was infra-bound, not logic-bound.
3. Add an offline integration mode with mocked/fixed search results so V1/Download/V2 stages can always be exercised even under network restrictions.
