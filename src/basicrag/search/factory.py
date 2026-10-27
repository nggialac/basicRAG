"""Build the retrieval stack from configuration.

The factory layers components in this order:

    base strategy -> parent expansion -> HyDE -> query rewrite -> reranking

Each layer is optional and independently configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from ..config import Settings, settings
from ..core.registry import Registry
from ..llm import get_llm
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .multi_query import MultiQueryRetriever
from .parent_document import ParentDocumentRetriever
from .sparse import BM25Index, BM25Retriever

SEARCH_STRATEGIES: Registry[Any] = Registry("search strategy")


@dataclass
class SearchContext:
    """Everything a retrieval strategy needs, built once per pipeline."""

    vector_store: VectorStore
    chunks: list[Document]
    embeddings: Embeddings
    docstore: Any
    cfg: Settings
    lexical_index: BM25Index | None = None

    def get_lexical_index(self) -> BM25Index:
        if self.lexical_index is None:
            self.lexical_index = BM25Index.from_documents(self.chunks)
        return self.lexical_index


def _dense_retriever(context: SearchContext, k: int) -> DenseRetriever:
    cfg = context.cfg.search
    search_type = cfg.dense_search_type
    if search_type not in {"similarity", "mmr", "similarity_score_threshold"}:
        raise ValueError(
            f"Unknown dense search type '{search_type}'. Use similarity, mmr, "
            "or similarity_score_threshold."
        )
    threshold = cfg.score_threshold if search_type == "similarity_score_threshold" else None
    return DenseRetriever(
        vector_store=context.vector_store,
        k=k,
        search_type=search_type,
        score_threshold=threshold,
        fetch_k=max(k, cfg.fetch_k),
        lambda_mult=cfg.mmr_lambda,
        filter=cfg.filter,
    )


def _sparse_retriever(context: SearchContext, k: int) -> BM25Retriever:
    return BM25Retriever(
        index=context.get_lexical_index(),
        k=k,
        filter=context.cfg.search.filter,
    )


@SEARCH_STRATEGIES.register("dense", aliases=("semantic", "vector"))
def _build_dense(context: SearchContext, k: int, llm: BaseChatModel | None) -> BaseRetriever:
    return _dense_retriever(context, k)


@SEARCH_STRATEGIES.register("sparse", aliases=("keyword", "bm25", "lexical"))
def _build_sparse(context: SearchContext, k: int, llm: BaseChatModel | None) -> BaseRetriever:
    return _sparse_retriever(context, k)


@SEARCH_STRATEGIES.register("hybrid", aliases=("ensemble",))
def _build_hybrid(context: SearchContext, k: int, llm: BaseChatModel | None) -> BaseRetriever:
    cfg = context.cfg.search
    return HybridRetriever(
        dense=_dense_retriever(context, k),
        sparse=_sparse_retriever(context, k),
        dense_weight=cfg.dense_weight,
        sparse_weight=cfg.sparse_weight,
        rrf_k=cfg.rrf_k,
        k=k,
    )


@SEARCH_STRATEGIES.register("multi_query", aliases=("multiquery", "expansion"))
def _build_multi_query(context: SearchContext, k: int, llm: BaseChatModel | None) -> BaseRetriever:
    if llm is None:
        raise ValueError("The multi_query search strategy requires an LLM")
    cfg = context.cfg.search
    base_name = cfg.multi_query_base
    if base_name not in {"dense", "hybrid"}:
        raise ValueError("search.multi_query_base must be 'dense' or 'hybrid'")
    base = SEARCH_STRATEGIES.get(base_name)(context, max(k, cfg.fetch_k), llm)
    return MultiQueryRetriever(
        base_retriever=base,
        llm=llm,
        count=cfg.multi_query_count,
        mode=cfg.multi_query_mode,
        rrf_k=cfg.rrf_k,
        k=k,
    )


def build_retriever(
    bundle: Any,
    cfg: Settings | None = None,
    *,
    rerank: bool | None = None,
    llm: BaseChatModel | None = None,
) -> BaseRetriever:
    """Assemble a retriever from an index bundle (see ``ingestion.IndexBundle``)."""
    cfg = cfg or getattr(bundle, "cfg", settings)
    use_rerank = cfg.rerank.enabled if rerank is None else rerank
    final_k = max(1, cfg.search.k)
    candidate_k = max(final_k, cfg.search.fetch_k) if use_rerank else final_k

    llm = llm or get_llm(cfg)
    context = SearchContext(
        vector_store=bundle.vector_store,
        chunks=bundle.chunks,
        embeddings=bundle.embeddings,
        docstore=bundle.docstore,
        cfg=cfg,
    )

    strategy = SEARCH_STRATEGIES.get(cfg.search.strategy)
    retriever = strategy(context, candidate_k, llm)

    if cfg.search.parent_document and context.docstore is not None:
        retriever = ParentDocumentRetriever(
            base_retriever=retriever,
            docstore=context.docstore,
            k=candidate_k,
        )

    if cfg.query.hyde:
        from ..query.transforms import HydeRetriever

        retriever = HydeRetriever(
            base_retriever=retriever,
            llm=llm,
            rrf_k=cfg.search.rrf_k,
            max_chars=cfg.query.hyde_max_chars,
            k=candidate_k,
        )

    if cfg.query.rewrite:
        from ..query.transforms import RewriteRetriever

        retriever = RewriteRetriever(
            base_retriever=retriever,
            llm=llm,
            max_chars=cfg.query.rewrite_max_chars,
        )

    if use_rerank:
        from ..reranking.factory import get_reranker
        from ..reranking.wrapper import RerankRetriever

        top_n = min(cfg.rerank.top_n or final_k, candidate_k)
        retriever = RerankRetriever(
            base_retriever=retriever,
            reranker=get_reranker(cfg, llm=llm),
            top_n=top_n,
        )

    return retriever


def search_documents(
    bundle: Any,
    query: str,
    cfg: Settings | None = None,
    **kwargs: Any,
) -> list[Document]:
    cfg = cfg or getattr(bundle, "cfg", settings)
    return build_retriever(bundle, cfg, **kwargs).invoke(query)


__all__ = [
    "build_retriever",
    "search_documents",
    "SearchContext",
    "SEARCH_STRATEGIES",
]
