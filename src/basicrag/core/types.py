"""Protocols and lightweight shared types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class Reranker(Protocol):
    """Anything that can reorder a candidate set for a query."""

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]: ...


@runtime_checkable
class DocumentStore(Protocol):
    """Minimal document store used by parent-document retrieval."""

    def add(self, documents: Sequence[Document]) -> None: ...

    def get(self, document_id: str) -> Document | None: ...

    def get_many(self, document_ids: Sequence[str]) -> list[Document]: ...

    def __len__(self) -> int: ...
