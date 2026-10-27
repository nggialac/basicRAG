"""Deprecated compatibility module. Use :mod:`basicrag.pipeline`."""

from .pipeline import AdvancedRAG, RAGResult, build_chain, build_rag, chat, print_result

__all__ = [
    "chat",
    "build_chain",
    "build_rag",
    "AdvancedRAG",
    "RAGResult",
    "print_result",
]
