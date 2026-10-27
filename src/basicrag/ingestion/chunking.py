"""Chunking strategies, including optional parent/child indexing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config import ChunkingConfig, Settings, settings
from ..core.ids import base_document_id
from ..core.registry import Registry
from ..observability import get_logger

logger = get_logger(__name__)

CHUNKERS: Registry[Any] = Registry("chunking strategy")


@dataclass
class ChunkResult:
    """Chunks that are indexed plus optional parent documents."""

    chunks: list[Document] = field(default_factory=list)
    parents: list[Document] = field(default_factory=list)


def _recursive_splitter(
    cfg: ChunkingConfig,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    separators = cfg.separators or ["\n\n", "\n", ". ", " ", ""]
    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size or cfg.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap,
        keep_separator=cfg.keep_separator,
        add_start_index=cfg.add_start_index,
        strip_whitespace=True,
    )


def _recursive(
    documents: list[Document], cfg: ChunkingConfig, embeddings: Embeddings | None
) -> list[Document]:
    return _recursive_splitter(cfg).split_documents(documents)


@CHUNKERS.register("recursive", aliases=("default",))
def _build_recursive(
    documents: list[Document], cfg: ChunkingConfig, embeddings: Embeddings | None
) -> list[Document]:
    return _recursive(documents, cfg, embeddings)


@CHUNKERS.register("fixed")
def _build_fixed(
    documents: list[Document], cfg: ChunkingConfig, embeddings: Embeddings | None
) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=[""],
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        keep_separator=False,
        add_start_index=cfg.add_start_index,
        strip_whitespace=True,
    )
    return splitter.split_documents(documents)


@CHUNKERS.register("token")
def _build_token(
    documents: list[Document], cfg: ChunkingConfig, embeddings: Embeddings | None
) -> list[Document]:
    from langchain_text_splitters import TokenTextSplitter

    splitter = TokenTextSplitter(
        encoding_name=cfg.token_encoding,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        add_start_index=cfg.add_start_index,
    )
    return splitter.split_documents(documents)


@CHUNKERS.register("markdown")
def _build_markdown(
    documents: list[Document], cfg: ChunkingConfig, embeddings: Embeddings | None
) -> list[Document]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=cfg.markdown_headers,
        strip_headers=False,
    )
    sections: list[Document] = []
    for document in documents:
        for section in header_splitter.split_text(document.page_content):
            section.metadata = {**document.metadata, **(section.metadata or {})}
            sections.append(section)

    recursive = _recursive_splitter(cfg)
    return [chunk for section in sections for chunk in recursive.split_documents([section])]


@CHUNKERS.register("semantic")
def _build_semantic(
    documents: list[Document], cfg: ChunkingConfig, embeddings: Embeddings | None
) -> list[Document]:
    if embeddings is None:
        raise ValueError("Semantic chunking requires an embedding model")
    from langchain_experimental.text_splitter import SemanticChunker

    chunker = SemanticChunker(
        embeddings=embeddings,
        buffer_size=cfg.semantic_buffer_size,
        breakpoint_threshold_type=cfg.breakpoint_threshold_type,  # type: ignore[arg-type]
        breakpoint_threshold_amount=cfg.breakpoint_threshold_amount,
        min_chunk_size=cfg.min_chunk_size or None,
    )
    return chunker.split_documents(documents)


def _with_source_ids(documents: list[Document]) -> list[Document]:
    for document in documents:
        metadata = document.metadata or {}
        metadata.setdefault("source_doc_id", base_document_id(document))
        document.metadata = metadata
    return documents


def _index_chunks(chunks: list[Document], cfg: ChunkingConfig) -> list[Document]:
    counters: dict[str, int] = defaultdict(int)
    indexed: list[Document] = []
    for chunk in chunks:
        text = (chunk.page_content or "").strip()
        if not text:
            continue
        metadata = dict(chunk.metadata or {})
        source_doc_id = str(metadata.get("source_doc_id") or base_document_id(chunk))
        index = counters[source_doc_id]
        counters[source_doc_id] += 1
        chunk_id = f"{source_doc_id}:c{index}"
        metadata.update(
            {
                "source_doc_id": source_doc_id,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "doc_id": chunk_id,
            }
        )
        if "parent_id" in metadata:
            metadata["doc_id"] = chunk_id
        indexed.append(Document(page_content=text, metadata=metadata, id=chunk_id))
    return indexed


def chunk_documents(
    documents: list[Document],
    embeddings: Embeddings | None = None,
    cfg: Settings | None = None,
    **overrides: Any,
) -> ChunkResult:
    """Split documents using the configured strategy.

    When ``chunking.parent_child`` is enabled, small child chunks are indexed
    and their larger parents are stored for parent-document retrieval.
    """
    cfg = cfg or settings
    chunk_cfg = cfg.chunking
    if overrides:
        chunk_cfg = chunk_cfg.model_copy(update=overrides)

    if chunk_cfg.strategy == "semantic" and embeddings is None:
        from ..embeddings import get_embeddings

        embeddings = get_embeddings(cfg)

    documents = _with_source_ids(documents)
    splitter: Callable[..., list[Document]] = CHUNKERS.get(chunk_cfg.strategy)

    if not chunk_cfg.parent_child:
        chunks = splitter(documents, chunk_cfg, embeddings)
        return ChunkResult(chunks=_index_chunks(chunks, chunk_cfg), parents=[])

    parent_splitter = _recursive_splitter(
        chunk_cfg,
        chunk_size=chunk_cfg.parent_chunk_size,
        chunk_overlap=chunk_cfg.parent_chunk_overlap,
    )
    parents = _with_source_ids(parent_splitter.split_documents(documents))

    parent_counters: dict[str, int] = defaultdict(int)
    named_parents: list[Document] = []
    for parent in parents:
        metadata = dict(parent.metadata or {})
        source_doc_id = str(metadata.get("source_doc_id") or base_document_id(parent))
        index = parent_counters[source_doc_id]
        parent_counters[source_doc_id] += 1
        parent_id = f"{source_doc_id}:p{index}"
        metadata.update(
            {
                "source_doc_id": source_doc_id,
                "parent_id": parent_id,
                "doc_id": parent_id,
            }
        )
        named_parents.append(
            Document(
                page_content=parent.page_content,
                metadata=metadata,
                id=parent_id,
            )
        )

    child_chunks = splitter(named_parents, chunk_cfg, embeddings)
    return ChunkResult(chunks=_index_chunks(child_chunks, chunk_cfg), parents=named_parents)


class _Splitter:
    """Compatibility wrapper matching the original ``get_splitter`` API."""

    def __init__(self, cfg: Settings, embeddings: Embeddings | None) -> None:
        self.cfg = cfg
        self.embeddings = embeddings

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return chunk_documents(documents, self.embeddings, self.cfg).chunks


def get_splitter(
    embeddings: Embeddings | None = None,
    cfg: Settings | None = None,
    **overrides: Any,
) -> _Splitter:
    cfg = cfg or settings
    if overrides:
        cfg = cfg.model_copy(update={"chunking": cfg.chunking.model_copy(update=overrides)})
    return _Splitter(cfg, embeddings)


def split_documents(
    documents: list[Document],
    embeddings: Embeddings | None = None,
    cfg: Settings | None = None,
    **overrides: Any,
) -> list[Document]:
    """Backward-compatible helper returning only the indexed chunks."""
    return chunk_documents(documents, embeddings, cfg, **overrides).chunks
