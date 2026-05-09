import os
import sqlite3
import sys

import faiss
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from core.storage.video_store import VideoStore
from validators.v2_models.arcface_embedder import ArcFaceEmbedder


def _make_store(tmp_path):
    return VideoStore(db_path=str(tmp_path / "videos.db"))


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(512, dtype=np.float32)


def test_embeddings_table_creation(tmp_path):
    store = _make_store(tmp_path)

    with sqlite3.connect(store.db_path) as conn:
        cursor = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'embeddings'
        """)
        assert cursor.fetchone() is not None


def test_save_and_load_embeddings(tmp_path):
    store = _make_store(tmp_path)
    embeddings = [_embedding(1), _embedding(2), _embedding(3)]

    inserted = store.save_embeddings("video-1", "youtube", embeddings)
    labels, loaded = store.load_all_embeddings()

    assert inserted == 3
    assert len(labels) == 3
    assert len(loaded) == 3
    for expected, actual in zip(embeddings, loaded):
        assert actual.shape == (512,)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected)


def test_save_duplicate_embedding(tmp_path):
    store = _make_store(tmp_path)

    assert store.save_embeddings("video-1", "youtube", [_embedding(1)]) == 1
    assert store.save_embeddings("video-1", "youtube", [_embedding(2)]) == 0
    assert store.get_embedding_count() == 1


def test_rebuild_from_db_empty(tmp_path):
    store = _make_store(tmp_path)

    index = ArcFaceEmbedder.rebuild_from_db(store)

    assert isinstance(index, faiss.Index)
    assert index.ntotal == 0


def test_rebuild_from_db_restores_all(tmp_path):
    store = _make_store(tmp_path)
    embeddings = [_embedding(seed) for seed in range(5)]
    for index, embedding in enumerate(embeddings):
        store.save_embeddings(f"video-{index}", "youtube", [embedding])

    index = ArcFaceEmbedder.rebuild_from_db(store)
    distances, indices = index.search(embeddings[0].reshape(1, -1).astype(np.float32), 1)

    assert index.ntotal == 5
    assert indices[0][0] == 0
    assert distances[0][0] == 0


def test_rebuild_from_db_matches_in_memory(tmp_path):
    store = _make_store(tmp_path)
    embeddings = [_embedding(seed) for seed in range(3)]

    in_memory = faiss.IndexFlatL2(512)
    in_memory.add(np.vstack(embeddings).astype(np.float32))
    for index, embedding in enumerate(embeddings):
        store.save_embeddings(f"video-{index}", "youtube", [embedding])

    rebuilt = ArcFaceEmbedder.rebuild_from_db(store)

    assert rebuilt.ntotal == in_memory.ntotal
