#!/usr/bin/env python3
"""
测试 face_quality 模块 - compute_laplacian_variance
"""

import os
import sys
import unittest

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from validators.v2_models.face_quality import compute_laplacian_variance


class TestFaceQuality(unittest.TestCase):
    def test_sharp_image_high_variance(self):
        """清晰图像应产生较高方差"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # 添加高频纹理
        img[::2, ::2] = 255
        img[1::2, 1::2] = 0
        variance = compute_laplacian_variance(img)
        self.assertGreater(variance, 0)

    def test_blank_image_low_variance(self):
        """纯色图像应产生很低方差"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        variance = compute_laplacian_variance(img)
        self.assertLess(variance, 1.0)

    def test_gradient_image(self):
        """渐变图像"""
        img = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
        img = cv2.merge([img, img, img])
        variance = compute_laplacian_variance(img)
        self.assertGreaterEqual(variance, 0)

    def test_grayscale_input(self):
        """单通道输入：转换为3通道后应正常工作"""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        # 函数内部会调用 cv2.cvtColor(..., COLOR_BGR2GRAY)，需要3通道输入
        bgr = cv2.merge([gray, gray, gray])
        variance = compute_laplacian_variance(bgr)
        self.assertGreater(variance, 0)

    def test_laplacian_threshold_comparison(self):
        """验证阈值逻辑：模糊图 < 清晰图"""
        blurry = cv2.GaussianBlur(
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),
            (15, 15), 5
        )
        sharp = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        blur_var = compute_laplacian_variance(blurry)
        sharp_var = compute_laplacian_variance(sharp)
        self.assertLess(blur_var, sharp_var)


if __name__ == "__main__":
    unittest.main()
