#!/usr/bin/env python3
import builtins
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from core.storage.video_store import VideoStore
from validators.validator import V2ContentFilter


class TestValidatorOptionalSpeech(unittest.TestCase):
    def test_no_speaker_import_when_speech_disabled(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        db_path = os.path.join(temp_dir.name, "videos.db")
        store = VideoStore(db_path=db_path)

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.endswith('speaker_detector'):
                raise ModuleNotFoundError("blocked speaker_detector import")
            return real_import(name, globals, locals, fromlist, level)

        with patch('validators.validator.YOLODetector') as mock_yolo, \
             patch('validators.validator.ArcFaceEmbedder') as mock_arc, \
             patch('builtins.__import__', side_effect=guarded_import):
            mock_yolo.return_value = object()
            mock_arc.return_value = object()
            cfg = {
                'coarse_filter': {'device': 'cpu', 'model_path': 'yolo11n.pt'},
                'fine_filter': {'device': 'cpu', 'speech_required': False},
            }
            filt = V2ContentFilter(cfg, store)
            self.assertFalse(hasattr(filt, 'speaker_detector'))


if __name__ == '__main__':
    unittest.main()
