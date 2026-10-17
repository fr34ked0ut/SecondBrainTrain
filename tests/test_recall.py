from dataclasses import dataclass

import numpy as np

from sbt.memory.vector_store import VectorStore


@dataclass
class FakeChunk:
    note_path: str
    title: str
    idx: int
    text: str
    tags: list


def test_search_ranks_by_similarity(tmp_path):
    store = VectorStore(tmp_path / "m.db", dim=3)
    chunks = [
        FakeChunk("a.md", "a", 0, "apartment lease", ["home"]),
        FakeChunk("b.md", "b", 0, "thesis outline", ["school"]),
    ]
    vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    store.upsert(chunks, vecs)
    hits = store.search(np.array([0.9, 0.1, 0], dtype=np.float32), k=1)
    assert hits[0]["note_path"] == "a.md"


def test_upsert_replaces(tmp_path):
    store = VectorStore(tmp_path / "m.db", dim=2)
    c = FakeChunk("a.md", "a", 0, "v1", [])
    store.upsert([c], np.array([[1, 0]], dtype=np.float32))
    c.text = "v2"
    store.upsert([c], np.array([[1, 0]], dtype=np.float32))
    assert store.count() == 1