"""No-op reranker for latency-sensitive or offline deployments."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document


class NoOpReranker:
    """Return the base retriever order unchanged."""

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        return list(documents)[:top_n] if top_n else list(documents)
