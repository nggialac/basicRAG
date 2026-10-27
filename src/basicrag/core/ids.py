"""Stable, content-based identifiers for documents and chunks."""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document


def sanitize_text(value: str) -> str:
    """Repair lone UTF-16 surrogates that UTF-8 cannot encode.

    PDF extractors occasionally emit surrogate code units (``\\ud835``) instead
    of real code points. Valid pairs are recombined; invalid ones become U+FFFD.
    """
    if not value:
        return value
    return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def stable_id(*parts: Any, length: int = 16) -> str:
    """Return a short deterministic id for arbitrary values."""
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha1(sanitize_text(payload).encode("utf-8")).hexdigest()
    return digest[:length]


def source_of(document: Document) -> str:
    metadata = document.metadata or {}
    return str(metadata.get("source") or metadata.get("file_name") or metadata.get("path") or "")


def base_document_id(document: Document) -> str:
    """Identify a source document independently of its content."""
    metadata = document.metadata or {}
    explicit = metadata.get("doc_id")
    if explicit:
        return str(explicit)
    return stable_id(source_of(document), metadata.get("page"))


def document_id(document: Document) -> str:
    """Return the best available stable id for a document/chunk."""
    if document.id:
        return str(document.id)
    metadata = document.metadata or {}
    explicit = metadata.get("doc_id") or metadata.get("chunk_id")
    if explicit:
        return str(explicit)
    return stable_id(source_of(document), metadata.get("page"), document.page_content)


def document_key(document: Document) -> str:
    """A key that can be matched against evaluation labels."""
    metadata = document.metadata or {}
    for key in ("doc_id", "chunk_id", "parent_id", "source", "file_name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return document_id(document)
