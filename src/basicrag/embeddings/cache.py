"""A small persistent embedding cache backed by SQLite.

Avoids re-embedding identical chunks across indexing runs without adding a
dependency on a hosted cache or a specific vector database vendor.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings


def _key(namespace: str, text: str) -> str:
    payload = f"{namespace}\x1f{text}".encode()
    return hashlib.sha256(payload).hexdigest()


class SQLiteEmbeddingCache:
    """Thread-safe key/value store for embedding vectors."""

    def __init__(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / "embeddings.sqlite3"
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                " key TEXT PRIMARY KEY,"
                " vector TEXT NOT NULL"
                ")"
            )

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT vector FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_many(self, keys: Sequence[str]) -> list[list[float] | None]:
        return [self.get(key) for key in keys]

    def put(self, key: str, vector: Sequence[float]) -> None:
        self.put_many({key: list(vector)})

    def put_many(self, vectors: dict[str, list[float]]) -> None:
        rows = [(key, json.dumps(vector)) for key, vector in vectors.items()]
        if not rows:
            return
        with self._lock, self._connection:
            self._connection.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                rows,
            )

    def clear(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM embeddings")

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class CachedEmbeddings(Embeddings):
    """Wrap an embedding model with SQLite-backed caching and batching."""

    def __init__(
        self,
        inner: Embeddings,
        cache: SQLiteEmbeddingCache,
        *,
        namespace: str = "",
        batch_size: int = 32,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.namespace = namespace
        self.batch_size = max(1, batch_size)

    def _cache_key(self, text: str) -> str:
        return _key(self.namespace, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._cache_key(text) for text in texts]
        cached = self.cache.get_many(keys)
        missing = [
            (index, text)
            for index, (text, vector) in enumerate(zip(texts, cached, strict=True))
            if vector is None
        ]

        if missing:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                vectors = self.inner.embed_documents([text for _, text in batch])
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        "Embedding provider returned a different number of vectors "
                        f"({len(vectors)}) than inputs ({len(batch)})"
                    )
                for (index, _), vector in zip(batch, vectors, strict=True):
                    cached[index] = vector
                    self.cache.put(keys[index], vector)

        return [vector or [] for vector in cached]

    def embed_query(self, text: str) -> list[float]:
        key = self._cache_key(text)
        vector = self.cache.get(key)
        if vector is None:
            vector = self.inner.embed_query(text)
            self.cache.put(key, vector)
        return vector


class NormalizedEmbeddings(Embeddings):
    """L2-normalize vectors so cosine and inner-product search agree."""

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner

    @staticmethod
    def _normalize(vector: Sequence[float]) -> list[float]:
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            return array.tolist()
        return (array / norm).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._normalize(vector) for vector in self.inner.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._normalize(self.inner.embed_query(text))
