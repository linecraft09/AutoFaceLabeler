import os
import sys

import pytest
from dotenv import load_dotenv


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Load .env so DASHSCOPE_API_KEY etc. are available in all test sessions
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"), override=True)
load_dotenv(override=True)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from core.storage.video_store import VideoStore


@pytest.fixture
def video_store(tmp_path):
    return VideoStore(str(tmp_path / "videos.db"))
