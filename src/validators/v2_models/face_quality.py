import cv2
import numpy as np


def compute_laplacian_variance(face_img: np.ndarray) -> float:
    """计算拉普拉斯方差作为清晰度指标"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()
