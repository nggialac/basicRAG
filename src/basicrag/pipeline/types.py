"""Result objects returned by the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class Source:
    index: int
    content: str
    source: str | None = None
    file_name: str | None = None
    page: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "source": self.source,
            "file_name": self.file_name,
            "page": self.page,
            "score": self.score,
        }
        if include_content:
            payload["content"] = self.content
        return payload


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    contexts: list[Document] = field(default_factory=list)

    def as_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.as_dict(include_content=include_content) for source in self.sources],
        }


__all__ = ["Source", "RAGResult"]
