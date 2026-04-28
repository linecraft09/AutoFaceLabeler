import os
import atexit
from typing import Any, Optional, Sequence, Tuple

import faiss
import numpy as np
from insightface.app import FaceAnalysis

from aflutils.logger import get_logger

logger = get_logger(__name__)


class ArcFaceEmbedder:
    """ArcFace 人脸特征提取与向量库去重"""

    def __init__(self, model_name: str = 'buffalo_m', device: str = 'cpu', db_path: str = 'data/face_index.faiss'):
        self.device = device
        self.db_path = db_path
        # 初始化 FaceAnalysis
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(name=model_name, providers=providers)
        self.face_app.prepare(ctx_id=0 if device == 'cuda' else -1, det_thresh=0.65, det_size=(320, 320))

        # FAISS 索引维度
        self.dim = 512
        self.index = None
        self._last_saved_ntotal = 0
        self._flush_lock = False
        self._load_index()
        atexit.register(self.flush)

    def _load_index(self):
        if os.path.exists(self.db_path):
            self.index = faiss.read_index(self.db_path)
            logger.info(f"Loaded FAISS index from {self.db_path}, size={self.index.ntotal}")
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            logger.info("Created new FAISS index")
        self._last_saved_ntotal = self.index.ntotal

    def _save_index(self):
        faiss.write_index(self.index, self.db_path)
        self._last_saved_ntotal = self.index.ntotal
        logger.info(f"Saved FAISS index to {self.db_path}, size={self.index.ntotal}")

    def flush(self):
        """写回未持久化的向量，确保增量写入不会在异常退出时丢失。"""
        if self.index is None:
            return
        if self._flush_lock:
            return
        if self.index.ntotal == self._last_saved_ntotal:
            return
        try:
            self._flush_lock = True
            self._save_index()
        except Exception as e:
            logger.error(f"Failed to flush FAISS index: {e}", exc_info=True)
        finally:
            self._flush_lock = False

    def is_duplicate(self, embedding: np.ndarray, threshold: float = 0.8) -> bool:
        """检查是否重复（L2距离阈值，归一化特征下0.8对应余弦相似度约0.68）"""
        if self.index.ntotal == 0:
            return False
        distances, _ = self.index.search(embedding.reshape(1, -1), 1)
        min_dist = distances[0][0]
        # 归一化特征 L2 距离范围 [0,2]，阈值0.8经验值
        return min_dist < threshold

    def add_embedding(self, embedding: np.ndarray):
        self.index.add(embedding.reshape(1, -1))
        if self.index.ntotal % 10 == 0:
            self._save_index()

    def __del__(self):
        try:
            self.flush()
        except Exception:
            # Avoid destructor-time hard failures.
            pass

    @staticmethod
    def _normalize_box(box: Any, frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Normalize an external face box into [x1, y1, x2, y2] int ndarray in frame bounds.
        """
        if box is None:
            return None
        if hasattr(box, "bbox"):
            box = box.bbox
        arr = np.asarray(box).reshape(-1)
        if arr.shape[0] < 4:
            return None
        x1, y1, x2, y2 = arr[:4].astype(int)
        h, w = frame_shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.array([x1, y1, x2, y2], dtype=int)

    def extract(self, face_img: np.ndarray, face_boxes: Optional[Sequence[Any]] = None):
        """
        Extract a face embedding result.
        If face_boxes is provided, skips full-frame face detection and runs ArcFace on cropped boxes.
        """
        if face_boxes:
            for box in face_boxes:
                bbox = self._normalize_box(box, face_img.shape)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox.tolist()
                crop = face_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                faces = self.face_app.get(crop, max_num=1)
                if len(faces) == 0:
                    continue
                face = faces[0]
                face.bbox = bbox.astype(np.float32)
                return face
            return None

        faces = self.face_app.get(face_img, max_num=1)
        if len(faces) == 0:
            return None
        return faces[0]
