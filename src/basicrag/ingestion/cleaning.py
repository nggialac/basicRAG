"""Text normalization, metadata enrichment, and de-duplication."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

from ..config import LoaderConfig
from ..core.ids import base_document_id, sanitize_text, stable_id

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    text = sanitize_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _NEWLINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def clean_documents(documents: list[Document], cfg: LoaderConfig | None = None) -> list[Document]:
    """Clean, enrich, and de-duplicate loaded documents."""
    cfg = cfg or LoaderConfig()
    seen: set[str] = set()
    cleaned: list[Document] = []
    ingested_at = datetime.now(UTC).isoformat()

    for document in documents:
        text = clean_text(document.page_content or "")
        if len(text) < cfg.min_chars:
            continue

        metadata = dict(document.metadata or {})
        source = str(metadata.get("source") or metadata.get("path") or "")
        path = Path(source) if source else None
        metadata.setdefault("source", source)
        if path and path.name:
            metadata.setdefault("file_name", path.name)
            metadata.setdefault("file_type", path.suffix.lower())
        metadata.setdefault("ingested_at", ingested_at)
        metadata.setdefault("doc_id", base_document_id(document))

        page = metadata.get("page")
        if isinstance(page, int):
            metadata.setdefault("page_number", page + 1)

        fingerprint = stable_id(source, metadata.get("page"), text, length=32)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        metadata["content_hash"] = fingerprint

        cleaned.append(Document(page_content=text, metadata=metadata, id=metadata["doc_id"]))

    return cleaned


def deduplicate(documents: list[Document]) -> list[Document]:
    """Drop exact duplicate page contents while preserving order."""
    seen: set[str] = set()
    unique: list[Document] = []
    for document in documents:
        key = stable_id(document.page_content, length=32)
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique
