"""Deprecated compatibility module.

Use :func:`basicrag.build_retriever` (advanced stack) or
:func:`basicrag.get_retriever` (simple dense wrapper).
"""

from . import get_retriever
from .search import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
)

__all__ = [
    "get_retriever",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "ParentDocumentRetriever",
]
