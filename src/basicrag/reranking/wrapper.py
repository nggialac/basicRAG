"""Retriever wrapper that applies a reranker to a candidate pool."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class RerankRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    reranker: Any
    top_n: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        return self.reranker.rerank(query, candidates, top_n=self.top_n)
