"""Expand small child chunks into their larger parent documents."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..core.ids import document_id


class ParentDocumentRetriever(BaseRetriever):
    """Retrieve child chunks, then return their stored parent documents."""

    base_retriever: BaseRetriever
    docstore: Any
    k: int = 8
    include_child_if_missing: bool = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        children = self.base_retriever.invoke(query)
        seen: set[str] = set()
        parents: list[Document] = []

        for child in children:
            metadata = child.metadata or {}
            parent_id = metadata.get("parent_id")
            parent = self.docstore.get(str(parent_id)) if parent_id else None
            document = parent or (child if self.include_child_if_missing else None)
            if document is None:
                continue
            key = document_id(document)
            if key in seen:
                continue
            seen.add(key)
            parents.append(document)
            if len(parents) >= self.k:
                break

        return parents
