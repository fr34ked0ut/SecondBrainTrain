import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    note_path TEXT NOT NULL,
    title TEXT,
    idx INTEGER,
    text TEXT NOT NULL,
    tags TEXT,
    vec BLOB NOT NULL,
    UNIQUE(note_path, idx)
);
CREATE INDEX IF NOT EXISTS ix_note ON chunks(note_path);
"""


class VectorStore:
    def __init__(self, path: Path, dim: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.dim = dim
        self._cache: tuple[np.ndarray, list[int]] | None = None

    def upsert(self, chunks, vecs: np.ndarray):
        rows = [
            (c.note_path, c.title, c.idx, c.text, json.dumps(c.tags), v.tobytes())
            for c, v in zip(chunks, vecs)
        ]
        self.conn.executemany(
            """INSERT INTO chunks(note_path, title, idx, text, tags, vec)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(note_path, idx) DO UPDATE SET
               title=excluded.title, text=excluded.text, tags=excluded.tags, vec=excluded.vec""",
            rows,
        )
        self.conn.commit()
        self._cache = None

    def delete_note(self, note_path: str):
        self.conn.execute("DELETE FROM chunks WHERE note_path = ?", (note_path,))
        self.conn.commit()
        self._cache = None

    def _matrix(self):
        if self._cache is None:
            cur = self.conn.execute("SELECT id, vec FROM chunks")
            ids, vecs = [], []
            for row_id, blob in cur:
                ids.append(row_id)
                vecs.append(np.frombuffer(blob, dtype=np.float32))
            mat = np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)
            self._cache = (mat, ids)
        return self._cache

    def search(self, query_vec: np.ndarray, k: int = 6, min_score: float = 0.0):
        mat, ids = self._matrix()
        if not ids:
            return []
        scores = mat @ query_vec
        top = np.argsort(-scores)[:k]
        hits = []
        for i in top:
            if scores[i] < min_score:
                continue
            row = self.conn.execute(
                "SELECT note_path, title, text, tags FROM chunks WHERE id = ?", (ids[i],)
            ).fetchone()
            hits.append(
                {
                    "score": float(scores[i]),
                    "note_path": row[0],
                    "title": row[1],
                    "text": row[2],
                    "tags": json.loads(row[3]),
                }
            )
        return hits

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]