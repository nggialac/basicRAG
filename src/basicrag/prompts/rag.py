"""Prompt construction and citation-friendly context rendering."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from ..config import PromptConfig, Settings, settings


def source_label(document: Document, index: int) -> str:
    metadata = document.metadata or {}
    name = metadata.get("file_name") or metadata.get("source") or metadata.get("path") or "document"
    page = metadata.get("page_number") or metadata.get("page")
    if isinstance(page, int):
        page = page + 1 if "page_number" not in metadata else page
    heading = metadata.get("h1") or metadata.get("h2") or metadata.get("h3")
    label = f"{name}"
    if isinstance(page, int):
        label += f", page {page}"
    if heading:
        label += f" — {heading}"
    return label


def format_context(
    documents: Sequence[Document],
    cfg: PromptConfig | Settings | None = None,
) -> str:
    """Render numbered passages with source labels, bounded by a char budget."""
    if isinstance(cfg, Settings):
        cfg = cfg.prompt
    cfg = cfg or settings.prompt

    blocks: list[str] = []
    used = 0
    for index, document in enumerate(documents, start=1):
        text = (document.page_content or "").strip()
        if not text:
            continue
        block = cfg.source_template.format(
            index=index,
            label=source_label(document, index),
            content=text,
        )
        remaining = cfg.max_context_chars - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining]
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def get_prompt(
    cfg: Settings | PromptConfig | str | None = None,
) -> ChatPromptTemplate:
    """Build the answer prompt (system + human template).

    Passing a string preserves the original API and uses it as a standalone
    template containing ``{context}`` and ``{question}``.
    """
    if isinstance(cfg, str):
        return ChatPromptTemplate.from_template(cfg)
    if isinstance(cfg, Settings):
        cfg = cfg.prompt
    cfg = cfg or settings.prompt
    return ChatPromptTemplate.from_messages(
        [
            ("system", cfg.system),
            ("human", cfg.answer_template),
        ]
    )


__all__ = ["get_prompt", "format_context", "source_label"]
