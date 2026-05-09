import os
import sys

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from core.storage.video_store import VideoStore


@pytest.fixture
def video_store(tmp_path):
    return VideoStore(str(tmp_path / "videos.db"))
