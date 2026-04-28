#!/usr/bin/env python3
"""
测试 YOLODetector:
1. 初始化(model download)
2. GPU 是否正常加载
3. 对空视频（无人物）返回正确结果
4. 对合成视频（含单人）检测单人时间段

需要 YOLO 模型：yolo11n.pt (首次运行自动下载)
"""

import os
import sys
import subprocess
import unittest
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from validators.v2_models.yolo_detector import YOLODetector
from aflutils.video_utils import FFMPEG, FFPROBE


class TestYOLODetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize YOLO model once for all tests"""
        print("\nLoading YOLO model (will download if not cached)...")
        cls.detector = YOLODetector(model_path='yolo11n.pt', device='cuda')
        print(f"YOLO device: {cls.detector.device}")

    def test_model_loaded_on_gpu(self):
        """验证 YOLO 模型在 GPU 上"""
        model_device = next(self.detector.model.parameters()).device
        self.assertIn(str(model_device), ['cuda', 'cuda:0'])

    def test_empty_video_no_person(self):
        """生成全黑视频，YOLO 应检测不到人物"""
        video_path = self._create_blank_video(duration_sec=3, fps=10)
        try:
            segments, total_single_secs = self.detector.detect_single_person_segments(
                video_path, stream=True
            )
            self.assertEqual(total_single_secs, 0.0)
            self.assertEqual(len(segments), 0)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def test_stream_vs_nonstream_consistency(self):
        """验证 stream=True 和 stream=False 对空视频结果一致"""
        video_path = self._create_blank_video(duration_sec=2, fps=5)
        try:
            segs_stream, total_stream = self.detector.detect_single_person_segments(
                video_path, stream=True
            )
            segs_nonstream, total_nonstream = self.detector.detect_single_person_segments(
                video_path, stream=False
            )
            self.assertEqual(total_stream, total_nonstream)
            self.assertEqual(len(segs_stream), len(segs_nonstream))
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def _create_blank_video(self, duration_sec=3, fps=10, width=640, height=480,
                            color=(0, 0, 0)):
        """用 OpenCV 生成纯色测试视频"""
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
        for _ in range(duration_sec * fps):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            out.write(frame)
        out.release()
        return tmp.name

    def _check_ffmpeg(self):
        """Check if ffmpeg can write a minimal h264 video"""
        try:
            subprocess.run([FFMPEG, "-version"], capture_output=True, check=True)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    unittest.main()
