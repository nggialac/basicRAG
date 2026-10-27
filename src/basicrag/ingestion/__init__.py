"""Document ingestion: loading, cleaning, chunking, and indexing."""

from .chunking import CHUNKERS, ChunkResult, chunk_documents, get_splitter, split_documents
from .cleaning import clean_documents, clean_text, deduplicate
from .docstore import JsonDocumentStore, load_documents_jsonl, save_documents_jsonl
from .indexer import IndexBundle, Indexer, build_index
from .loaders import DocumentCollectionLoader, get_loader, load_documents

__all__ = [
    "load_documents",
    "get_loader",
    "DocumentCollectionLoader",
    "clean_documents",
    "clean_text",
    "deduplicate",
    "chunk_documents",
    "split_documents",
    "get_splitter",
    "ChunkResult",
    "CHUNKERS",
    "Indexer",
    "IndexBundle",
    "build_index",
    "JsonDocumentStore",
    "save_documents_jsonl",
    "load_documents_jsonl",
]
