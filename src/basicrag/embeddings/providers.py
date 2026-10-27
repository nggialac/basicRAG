"""Embedding provider implementations.

Register a new provider by adding a function decorated with
``@EMBEDDING_PROVIDERS.register("name")``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from ..config import EmbeddingConfig
from ..core.registry import Registry

EMBEDDING_PROVIDERS: Registry[Any] = Registry("embedding provider")


@EMBEDDING_PROVIDERS.register("ollama")
def _ollama(cfg: EmbeddingConfig, **overrides: Any) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    params: dict[str, Any] = {"model": cfg.model}
    if cfg.base_url:
        params["base_url"] = cfg.base_url
    if cfg.dimensions:
        params["dimensions"] = cfg.dimensions
    params.update(cfg.extra)
    params.update(overrides)
    return OllamaEmbeddings(**params)


@EMBEDDING_PROVIDERS.register("openai", aliases=("openai_compat",))
def _openai(cfg: EmbeddingConfig, **overrides: Any) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    params: dict[str, Any] = {"model": cfg.model}
    if cfg.base_url:
        params["base_url"] = cfg.base_url
    if cfg.api_key:
        params["api_key"] = cfg.api_key.get_secret_value()
    if cfg.dimensions:
        params["dimensions"] = cfg.dimensions
    params.update(cfg.extra)
    params.update(overrides)
    return OpenAIEmbeddings(**params)


@EMBEDDING_PROVIDERS.register("huggingface", aliases=("sentence_transformers",))
def _huggingface(cfg: EmbeddingConfig, **overrides: Any) -> Embeddings:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface' embedding provider requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    model_name = overrides.pop("model", cfg.model)
    device = overrides.pop("device", None)
    normalize = overrides.pop("normalize_embeddings", cfg.normalize)
    model = SentenceTransformer(model_name, device=device, **cfg.extra, **overrides)

    class _SentenceTransformerEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            vectors = model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)
            return vectors.tolist()

        def embed_query(self, text: str) -> list[float]:
            vector = model.encode([text], normalize_embeddings=normalize, convert_to_numpy=True)
            return vector[0].tolist()

    return _SentenceTransformerEmbeddings()


@EMBEDDING_PROVIDERS.register("fake")
def _fake(cfg: EmbeddingConfig, **overrides: Any) -> Embeddings:
    from langchain_core.embeddings import FakeEmbeddings

    size = overrides.pop("size", cfg.dimensions or 32)
    return FakeEmbeddings(size=size, **overrides)
