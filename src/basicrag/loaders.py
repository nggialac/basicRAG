"""Deprecated compatibility module. Use :mod:`basicrag.ingestion.loaders`."""

from .ingestion import DocumentCollectionLoader, get_loader, load_documents

__all__ = ["get_loader", "load_documents", "DocumentCollectionLoader"]
