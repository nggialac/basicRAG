"""Dense (embedding) retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import Field


def with_score(document: Document, score: float, key: str = "retrieval_score") -> Document:
    metadata = {**(document.metadata or {}), key: round(float(score), 6)}
    return document.model_copy(update={"metadata": metadata})


class DenseRetriever(BaseRetriever):
    """Thin retriever with similarity / MMR / score-threshold support."""

    vector_store: VectorStore
    k: int = 4
    search_type: str = "similarity"
    score_threshold: float | None = None
    fetch_k: int = 20
    lambda_mult: float = 0.5
    filter: dict[str, Any] | None = Field(default=None)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        kwargs: dict[str, Any] = {"k": self.k}
        if self.filter:
            kwargs["filter"] = self.filter

        if self.search_type == "mmr":
            return self.vector_store.max_marginal_relevance_search(
                query,
                fetch_k=max(self.fetch_k, self.k),
                lambda_mult=self.lambda_mult,
                **kwargs,
            )

        if self.search_type == "similarity_score_threshold":
            threshold = 0.0 if self.score_threshold is None else self.score_threshold
            pairs = self.vector_store.similarity_search_with_relevance_scores(query, **kwargs)
            return [with_score(document, score) for document, score in pairs if score >= threshold]

        return self.vector_store.similarity_search(query, **kwargs)
