"""Embedding factory with optional normalization and caching."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from ..config import Settings, settings
from .cache import CachedEmbeddings, NormalizedEmbeddings, SQLiteEmbeddingCache
from .providers import EMBEDDING_PROVIDERS


def get_embeddings(
    cfg: Settings | None = None,
    provider: str | None = None,
    *,
    cache: bool | None = None,
    **overrides: Any,
) -> Embeddings:
    """Build an embedding model from config or explicit overrides."""
    cfg = cfg or settings
    provider = provider or cfg.embeddings.provider
    embeddings = EMBEDDING_PROVIDERS.create(provider, cfg.embeddings, **overrides)

    if cfg.embeddings.normalize:
        embeddings = NormalizedEmbeddings(embeddings)

    use_cache = cfg.embeddings.cache if cache is None else cache
    if use_cache:
        store = SQLiteEmbeddingCache(cfg.embeddings.cache_directory)
        embeddings = CachedEmbeddings(
            embeddings,
            store,
            namespace=f"{provider}:{cfg.embeddings.model}",
            batch_size=cfg.embeddings.batch_size,
        )
    return embeddings


__all__ = ["get_embeddings", "EMBEDDING_PROVIDERS"]
