"""Deprecated compatibility module. Use :mod:`basicrag.ingestion.chunking`."""

from .ingestion import CHUNKERS, ChunkResult, chunk_documents, get_splitter, split_documents

__all__ = [
    "get_splitter",
    "split_documents",
    "chunk_documents",
    "ChunkResult",
    "CHUNKERS",
]
