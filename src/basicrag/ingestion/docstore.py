"""JSONL-backed document store used for parent chunks and offline inspection."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from langchain_core.documents import Document

from ..core.ids import document_id


class JsonDocumentStore:
    """Simple document store that persists as JSON Lines.

    Good enough for local/business deployments with up to a few hundred
    thousand chunks. Swap it for Redis/Postgres by implementing the same
    ``add`` / ``get`` / ``get_many`` methods.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._documents: dict[str, Document] = {}

    def add(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self._documents[document_id(document)] = document

    def get(self, document_id_value: str) -> Document | None:
        return self._documents.get(document_id_value)

    def get_many(self, document_ids: Sequence[str]) -> list[Document]:
        return [
            document
            for document_id_value in document_ids
            if (document := self._documents.get(document_id_value)) is not None
        ]

    def all(self) -> list[Document]:
        return list(self._documents.values())

    def clear(self) -> None:
        self._documents.clear()

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, document_id_value: object) -> bool:
        return document_id_value in self._documents

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("No path supplied for JsonDocumentStore.save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for key, document in self._documents.items():
                payload = {
                    "id": key,
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, default=str))
                handle.write("\n")
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: str | Path) -> JsonDocumentStore:
        source = Path(path)
        store = cls(source)
        if not source.exists():
            return store
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                document = Document(
                    page_content=payload["page_content"],
                    metadata=payload.get("metadata") or {},
                    id=payload.get("id"),
                )
                store._documents[payload.get("id") or document_id(document)] = document
        return store


def save_documents_jsonl(documents: Iterable[Document], path: str | Path) -> None:
    store = JsonDocumentStore(path)
    store.add(documents)
    store.save()


def load_documents_jsonl(path: str | Path) -> list[Document]:
    return JsonDocumentStore.load(path).all()
