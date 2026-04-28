#!/usr/bin/env python3
import os
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from validators.v2_models.arcface_embedder import ArcFaceEmbedder
from validators.v2_models.yolo_detector import YOLODetector


class TestDeviceFallbacks(unittest.TestCase):
    def test_yolo_falls_back_to_cpu_when_cuda_unavailable(self):
        with patch('validators.v2_models.yolo_detector.torch.cuda.is_available', return_value=False), \
             patch('validators.v2_models.yolo_detector.YOLO') as mock_yolo_cls:
            mock_model = mock_yolo_cls.return_value
            detector = YOLODetector(model_path='dummy.pt', device='cuda')
            self.assertEqual(detector.device, 'cpu')
            mock_model.to.assert_not_called()

    def test_arcface_falls_back_to_cpu_when_cuda_unavailable(self):
        with patch('validators.v2_models.arcface_embedder.torch.cuda.is_available', return_value=False), \
             patch('validators.v2_models.arcface_embedder.FaceAnalysis') as mock_face_analysis, \
             patch.object(ArcFaceEmbedder, '_load_index', autospec=True) as mock_load_index:
            mock_instance = mock_face_analysis.return_value
            emb = ArcFaceEmbedder(device='cuda', db_path=':memory:')
            self.assertEqual(emb.device, 'cpu')
            mock_face_analysis.assert_called_once()
            _, kwargs = mock_face_analysis.call_args
            self.assertEqual(kwargs['providers'], ['CPUExecutionProvider'])
            mock_instance.prepare.assert_called_once()
            self.assertEqual(mock_instance.prepare.call_args.kwargs['ctx_id'], -1)
            mock_load_index.assert_called_once()


if __name__ == '__main__':
    unittest.main()
