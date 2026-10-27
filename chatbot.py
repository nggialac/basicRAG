"""Interactive entry point.

Every stage is independently configurable:

- ``configs/*.yaml`` or ``BASICRAG_*`` environment variables
- ``src/basicrag/ingestion``  loading, cleaning, chunking, indexing
- ``src/basicrag/embeddings`` embedding providers and caching
- ``src/basicrag/vectorstores`` memory / FAISS / Chroma / Qdrant
- ``src/basicrag/search`` dense, BM25, hybrid RRF, multi-query, parent docs
- ``src/basicrag/reranking`` LLM / cross-encoder reranking
- ``src/basicrag/query`` query rewriting and HyDE
- ``src/basicrag/pipeline`` answer generation and citations
"""

from basicrag.pipeline import chat

if __name__ == "__main__":
    chat()
