"""Reciprocal Rank Fusion (RRF) for combining multiple result lists."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.documents import Document

from ..core.ids import document_id, stable_id


def document_key(document: Document) -> str:
    """Use stable ids before falling back to content."""
    if document.id:
        return str(document.id)
    metadata = document.metadata or {}
    for key in ("chunk_id", "parent_id", "doc_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return document_id(document)


@dataclass
class _Fused:
    document: Document
    score: float = 0.0
    retrievers: list[str] = field(default_factory=list)
    best_rank: int = 10**9


def weighted_rrf(
    result_lists: Sequence[Sequence[Document]],
    weights: Sequence[float] | None = None,
    *,
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[Document]:
    """Fuse ranked lists. Scores are ``weight / (rrf_k + rank + 1)``."""
    if not result_lists:
        return []
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")

    fused: dict[str, _Fused] = {}
    for list_index, (documents, weight) in enumerate(zip(result_lists, weights, strict=True)):
        label = f"list{list_index}"
        for rank, document in enumerate(documents):
            key = document_key(document)
            entry = fused.get(key)
            if entry is None:
                entry = _Fused(document=document)
                fused[key] = entry
            entry.score += weight / (rrf_k + rank + 1)
            entry.best_rank = min(entry.best_rank, rank)
            if label not in entry.retrievers:
                entry.retrievers.append(label)

    ordered = sorted(fused.values(), key=lambda item: (-item.score, item.best_rank))
    if limit is not None:
        ordered = ordered[:limit]

    documents: list[Document] = []
    for item in ordered:
        metadata = {
            **(item.document.metadata or {}),
            "rrf_score": round(item.score, 8),
            "rrf_sources": list(item.retrievers),
        }
        documents.append(item.document.model_copy(update={"metadata": metadata}))
    return documents


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Document]],
    *,
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[Document]:
    """Unweighted RRF convenience wrapper."""
    return weighted_rrf(result_lists, rrf_k=rrf_k, limit=limit)


def merge_unique(documents: Sequence[Document]) -> list[Document]:
    """De-duplicate while preserving order (useful outside RRF)."""
    seen: set[str] = set()
    merged: list[Document] = []
    for document in documents:
        key = document_key(document) or stable_id(document.page_content)
        if key in seen:
            continue
        seen.add(key)
        merged.append(document)
    return merged
