"""Hybrid dense + lexical retrieval fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .fusion import weighted_rrf


class HybridRetriever(BaseRetriever):
    """Combine semantic recall with keyword precision."""

    dense: BaseRetriever
    sparse: BaseRetriever
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60
    k: int = 8

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        dense_documents = self.dense.invoke(query)
        sparse_documents = self.sparse.invoke(query)
        return weighted_rrf(
            [dense_documents, sparse_documents],
            [self.dense_weight, self.sparse_weight],
            rrf_k=self.rrf_k,
            limit=self.k,
        )
