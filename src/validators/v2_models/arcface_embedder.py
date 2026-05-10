import os
import atexit
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import faiss
import numpy as np
import torch
from insightface.app import FaceAnalysis

from aflutils.logger import get_logger
from .ort_utils import DEFAULT_ARCFACE_BATCH_MODEL, create_session, resolve_model_path

logger = get_logger(__name__)


class ArcFaceEmbedder:
    """ArcFace 人脸特征提取与向量库去重"""

    def __init__(
        self,
        model_name: str = 'buffalo_m',
        device: str = 'cpu',
        db_path: str = 'data/face_index.faiss',
        batch_model_path: str | Path = DEFAULT_ARCFACE_BATCH_MODEL,
        batch_size: int = 16,
    ):
        requested_device = (device or 'cpu').lower()
        # 'cuda' → try GPU (warn on fallback); 'auto' → use GPU if available, else CPU silently
        if requested_device in ('cuda', 'auto'):
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
                if requested_device == 'cuda':
                    logger.warning("CUDA requested for ArcFace but unavailable; falling back to CPU")
        else:
            self.device = 'cpu'
        self.db_path = db_path
        self.batch_model_path = resolve_model_path(batch_model_path)
        self.batch_size = int(batch_size)
        self._batch_session = None
        self._batch_input_name = None
        self._batch_output_names = None
        # 初始化 FaceAnalysis
        providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(name=model_name, providers=providers)
        self.face_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_thresh=0.65, det_size=(320, 320))

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

    @classmethod
    def rebuild_from_db(cls, video_store, dim=512) -> faiss.Index:
        """Rebuild a FAISS index from SQLite-persisted embeddings."""
        _, embeddings = video_store.load_all_embeddings()
        index = faiss.IndexFlatL2(dim)
        if not embeddings:
            return index

        embeddings_array = np.vstack([
            np.asarray(embedding, dtype=np.float32).reshape(dim,)
            for embedding in embeddings
        ]).astype(np.float32)
        index.add(embeddings_array)
        return index

    def _save_index(self, log: bool = True):
        if self.db_path == ':memory:':
            return
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        faiss.write_index(self.index, self.db_path)
        self._last_saved_ntotal = self.index.ntotal
        if log:
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
            self._save_index(log=False)
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

    def _get_batch_session(self):
        if self._batch_session is None:
            self._batch_session = create_session(self.batch_model_path)
            self._batch_input_name = self._batch_session.get_inputs()[0].name
            self._batch_output_names = [item.name for item in self._batch_session.get_outputs()]
        return self._batch_session

    def embed_crops_batch(self, face_crops: list[np.ndarray]) -> np.ndarray:
        """
        Extract normalized ArcFace embeddings from a batch of aligned or cropped faces.

        Returns:
            A float32 array with shape [N, 512].
        """
        if not face_crops:
            return np.empty((0, self.dim), dtype=np.float32)

        session = self._get_batch_session()
        embeddings = []
        for start in range(0, len(face_crops), self.batch_size):
            crops = face_crops[start : start + self.batch_size]
            prepared = []
            for crop in crops:
                if crop is None or crop.size == 0:
                    raise ValueError("face_crops contains an empty crop")
                prepared.append(cv2.resize(crop, (112, 112)))
            blob = cv2.dnn.blobFromImages(
                prepared,
                1.0 / 127.5,
                (112, 112),
                (127.5, 127.5, 127.5),
                swapRB=True,
            )
            output = session.run(self._batch_output_names, {self._batch_input_name: blob})[0]
            if output.shape != (len(prepared), self.dim):
                raise RuntimeError(f"ArcFace output shape mismatch: expected {(len(prepared), self.dim)}, got {output.shape}")
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            embeddings.append((output / np.maximum(norms, 1e-12)).astype(np.float32))
        return np.vstack(embeddings)

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
