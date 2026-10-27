"""Vector store factory."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..config import Settings, settings
from ..embeddings import get_embeddings
from .providers import VECTOR_STORE_PROVIDERS, VectorStoreProvider


def _provider(cfg: Settings, **overrides: Any) -> VectorStoreProvider:
    store_cfg = cfg.vector_store
    if overrides:
        store_cfg = store_cfg.model_copy(update=overrides)
    return VECTOR_STORE_PROVIDERS.create(store_cfg.provider, store_cfg)


def get_vector_store(
    embeddings: Embeddings | None = None,
    cfg: Settings | None = None,
    documents: list[Document] | None = None,
    *,
    load: bool = False,
    **overrides: Any,
) -> VectorStore:
    """Create (or load) a vector store from configuration."""
    cfg = cfg or settings
    embeddings = embeddings or get_embeddings(cfg)
    provider = _provider(cfg, **overrides)

    if load:
        store = provider.load(embeddings)
        if store is None:
            raise FileNotFoundError(
                f"No persisted {cfg.vector_store.provider} index found for "
                f"collection '{cfg.vector_store.collection}'. Run indexing first."
            )
        return store

    return provider.create(embeddings, documents)


def create_vector_store(
    embeddings: Embeddings | None = None,
    cfg: Settings | None = None,
    documents: list[Document] | None = None,
    **overrides: Any,
) -> VectorStore:
    return get_vector_store(embeddings, cfg, documents, **overrides)


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings | None = None,
    cfg: Settings | None = None,
    **overrides: Any,
) -> VectorStore:
    """Build and (optionally) persist an index from documents."""
    cfg = cfg or settings
    store = get_vector_store(embeddings, cfg, documents, **overrides)
    if cfg.vector_store.persist:
        persist_vector_store(store, cfg, **overrides)
    return store


def persist_vector_store(store: VectorStore, cfg: Settings | None = None, **overrides: Any) -> None:
    cfg = cfg or settings
    if not cfg.vector_store.persist:
        return
    _provider(cfg, **overrides).persist(store)


def load_vector_store(
    embeddings: Embeddings | None = None, cfg: Settings | None = None, **overrides: Any
) -> VectorStore:
    return get_vector_store(embeddings, cfg, load=True, **overrides)


def delete_vector_store(cfg: Settings | None = None, **overrides: Any) -> None:
    cfg = cfg or settings
    _provider(cfg, **overrides).delete()


def vector_store_exists(cfg: Settings | None = None, **overrides: Any) -> bool:
    cfg = cfg or settings
    return _provider(cfg, **overrides).exists()


__all__ = [
    "get_vector_store",
    "create_vector_store",
    "build_vector_store",
    "load_vector_store",
    "persist_vector_store",
    "delete_vector_store",
    "vector_store_exists",
    "VECTOR_STORE_PROVIDERS",
]
