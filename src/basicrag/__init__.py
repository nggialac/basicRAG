"""BasicRAG: a modular, production-oriented RAG toolkit.

Pipeline overview::

    ingestion (load -> clean -> chunk) -> embeddings -> vector store
        -> search (dense / sparse / hybrid / multi-query)
        -> parent expansion -> query rewrite / HyDE -> reranking
        -> grounded answer + citations

Each stage is a registry-backed factory. Add a provider or strategy by
decorating a function in the corresponding module and selecting it in
``configs/*.yaml`` or via ``BASICRAG_*`` environment variables.
"""

from __future__ import annotations

from typing import Any

from langchain_core.retrievers import BaseRetriever

from .config import Settings, load_settings, settings
from .embeddings import EMBEDDING_PROVIDERS, get_embeddings
from .evaluation import EvalCase, EvalReport, evaluate, load_cases
from .ingestion import (
    CHUNKERS,
    ChunkResult,
    IndexBundle,
    Indexer,
    build_index,
    chunk_documents,
    clean_documents,
    get_loader,
    get_splitter,
    load_documents,
    split_documents,
)
from .llm import LLM_PROVIDERS, get_llm
from .pipeline import (
    AdvancedRAG,
    RAGResult,
    Source,
    build_chain,
    build_from_bundle,
    build_rag,
    chat,
)
from .prompts import TEMPLATE, format_context, get_prompt
from .query import HydeRetriever, RewriteRetriever
from .reranking import (
    RERANKERS,
    CrossEncoderReranker,
    LLMReranker,
    NoOpReranker,
    RerankRetriever,
    get_reranker,
)
from .search import (
    SEARCH_STRATEGIES,
    BM25Index,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
    build_retriever,
    search_documents,
    weighted_rrf,
)
from .vectorstores import (
    VECTOR_STORE_PROVIDERS,
    build_vector_store,
    create_vector_store,
    delete_vector_store,
    get_vector_store,
    load_vector_store,
    persist_vector_store,
    vector_store_exists,
)

__version__ = "0.2.0"


def get_retriever(
    vector_store: Any,
    cfg: Settings | None = None,
    *,
    k: int | None = None,
    search_type: str | None = None,
    score_threshold: float | None = None,
    **search_kwargs: Any,
) -> BaseRetriever:
    """Backward-compatible dense retriever factory.

    New code should use :func:`basicrag.build_retriever` for the full advanced
    stack (hybrid search, reranking, query transforms).
    """
    cfg = cfg or settings
    search_type = search_type or cfg.search.dense_search_type
    kwargs: dict[str, Any] = {"k": cfg.search.k if k is None else k}
    if search_type == "similarity_score_threshold":
        kwargs["score_threshold"] = (
            cfg.search.score_threshold if score_threshold is None else score_threshold
        )
    kwargs.update(search_kwargs)
    return vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)


def main() -> None:
    from .cli import main as cli_main

    raise SystemExit(cli_main())


__all__ = [
    "__version__",
    "Settings",
    "settings",
    "load_settings",
    # LLM / embeddings
    "get_llm",
    "get_embeddings",
    "LLM_PROVIDERS",
    "EMBEDDING_PROVIDERS",
    # Ingestion
    "load_documents",
    "get_loader",
    "clean_documents",
    "chunk_documents",
    "split_documents",
    "get_splitter",
    "ChunkResult",
    "CHUNKERS",
    "Indexer",
    "IndexBundle",
    "build_index",
    # Vector stores
    "get_vector_store",
    "create_vector_store",
    "build_vector_store",
    "load_vector_store",
    "persist_vector_store",
    "delete_vector_store",
    "vector_store_exists",
    "VECTOR_STORE_PROVIDERS",
    # Retrieval
    "get_retriever",
    "build_retriever",
    "search_documents",
    "SEARCH_STRATEGIES",
    "DenseRetriever",
    "BM25Index",
    "BM25Retriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "ParentDocumentRetriever",
    "weighted_rrf",
    "RewriteRetriever",
    "HydeRetriever",
    # Reranking
    "get_reranker",
    "RERANKERS",
    "LLMReranker",
    "CrossEncoderReranker",
    "NoOpReranker",
    "RerankRetriever",
    # Answers
    "get_prompt",
    "format_context",
    "TEMPLATE",
    "AdvancedRAG",
    "RAGResult",
    "Source",
    "build_rag",
    "build_chain",
    "build_from_bundle",
    "chat",
    # Evaluation
    "EvalCase",
    "EvalReport",
    "load_cases",
    "evaluate",
    "main",
]
