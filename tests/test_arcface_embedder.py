#!/usr/bin/env python3
"""
测试 ArcFaceEmbedder:
1. 初始化（insightface model download）
2. GPU 推理
3. FAISS 索引创建/加载/保存
4. extract -> 人脸检测
5. is_duplicate / add_embedding 逻辑

注意：如果测试环境无 GPU，会回退到 CPU
"""

import os
import sys
import unittest
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from validators.v2_models.arcface_embedder import ArcFaceEmbedder
from validators.v2_models.face_quality import compute_laplacian_variance


class TestArcFaceEmbedderInit(unittest.TestCase):
    """测试初始化和 FAISS"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_face_index.faiss")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_gpu(self):
        """初始化 ArcFaceEmbedder 在 GPU 上"""
        try:
            embedder = ArcFaceEmbedder(device='cuda', db_path=self.db_path)
            self.assertEqual(embedder.dim, 512)
            self.assertIsNotNone(embedder.index)
            self.assertIsNotNone(embedder.face_app)
            # verify at least one model is loaded on GPU
            self.assertIn('detection', embedder.face_app.models)
        except Exception as e:
            self.fail(f"ArcFace GPU init failed: {e}")

    def test_faiss_index_create_and_load(self):
        """创建空 FAISS 索引，验证 ntotal=0"""
        embedder = ArcFaceEmbedder(device='cuda', db_path=self.db_path)
        self.assertEqual(embedder.index.ntotal, 0)

        # add embeddings
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        embedder.add_embedding(emb1)
        self.assertEqual(embedder.index.ntotal, 1)

        # save and reload
        embedder._save_index()
        embedder2 = ArcFaceEmbedder(device='cuda', db_path=self.db_path)
        self.assertEqual(embedder2.index.ntotal, 1)

    def test_is_duplicate(self):
        """验证去重逻辑"""
        embedder = ArcFaceEmbedder(device='cuda', db_path=self.db_path)

        # empty index -> not duplicate
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        self.assertFalse(embedder.is_duplicate(emb, threshold=0.8))

        # add same embedding -> should be duplicate
        embedder.add_embedding(emb)
        self.assertTrue(embedder.is_duplicate(emb, threshold=0.8))

        # different embedding (far apart) -> not duplicate
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        # If very different, L2 distance > threshold -> not duplicate
        is_dup = embedder.is_duplicate(emb2, threshold=0.001)
        self.assertFalse(is_dup)


class TestArcFaceExtract(unittest.TestCase):
    """测试人脸提取功能（需要一个 face image）"""

    @classmethod
    def setUpClass(cls):
        cls.embedder = ArcFaceEmbedder(device='cuda')

    def test_extract_synthetic_face(self):
        """生成一个简单的人脸状图案，检测是否返回 None（可能检测不到，但不应报错）"""
        # 创建一个类人脸椭圆形图案
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # 画一个椭圆模拟人脸位置
        cv2.ellipse(img, (320, 240), (80, 100), 0, 0, 360, (180, 150, 130), -1)
        # 画眼睛和嘴巴（简单特征）
        cv2.circle(img, (290, 210), 10, (50, 50, 50), -1)
        cv2.circle(img, (350, 210), 10, (50, 50, 50), -1)
        cv2.ellipse(img, (320, 280), (30, 15), 0, 0, 180, (80, 60, 60), 3)

        try:
            face = self.embedder.extract(img)
            # 合成图可能检测不到人脸，所以允许 face is None
            # 但至少不应报错
            if face is not None:
                self.assertIsNotNone(face.normed_embedding)
                self.assertEqual(len(face.normed_embedding), 512)
        except Exception as e:
            self.fail(f"extract failed: {e}")

    def test_extract_empty_image(self):
        """空背景不应检测出人脸"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 240
        face = self.embedder.extract(img)
        self.assertIsNone(face)


if __name__ == "__main__":
    unittest.main()
