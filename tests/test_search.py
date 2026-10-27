from __future__ import annotations

import re
import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from basicrag import load_settings
from basicrag.ingestion import IndexBundle, chunk_documents, clean_documents
from basicrag.ingestion.docstore import JsonDocumentStore
from basicrag.reranking.llm import LLMReranker
from basicrag.search import (
    BM25Index,
    MultiQueryRetriever,
    build_retriever,
    weighted_rrf,
)
from basicrag.vectorstores import create_vector_store


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding for offline tests."""

    def __init__(self, dimensions: int = 12) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"\w+", text.lower()):
            index = sum(ord(char) for char in word) % self.dimensions
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_settings(**overrides):
    base = {
        "llm": {"provider": "fake"},
        "embeddings": {"provider": "fake", "cache": False, "dimensions": 12},
        "vector_store": {"provider": "memory", "persist": False},
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 120,
            "chunk_overlap": 20,
            "parent_child": False,
        },
        "search": {
            "strategy": "sparse",
            "k": 2,
            "fetch_k": 5,
            "parent_document": False,
        },
        "rerank": {"enabled": False, "provider": "none"},
        "query": {"rewrite": False, "hyde": False},
        "pipeline": {"verbose": False},
    }
    base.update(overrides)
    return load_settings(**base)


def make_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "The transformer architecture uses self attention to handle very "
                "long documents and efficient long context retrieval."
            ),
            metadata={"source": "transformer.txt", "file_name": "transformer.txt"},
        ),
        Document(
            page_content=(
                "Cats sleep on warm sofas during the afternoon and ignore every "
                "request to move elsewhere."
            ),
            metadata={"source": "cats.txt", "file_name": "cats.txt"},
        ),
    ]


def show(title: str, items) -> None:
    print(f"\n=== {title}: {len(items)} result(s) ===")
    for rank, item in enumerate(items, start=1):
        if isinstance(item, tuple):
            document, score = item
            metadata = document.metadata or {}
            print(
                f"  {rank}. score={score:.4f} metadata={metadata} "
                f"content={document.page_content[:80]!r}"
            )
        else:
            metadata = item.metadata or {}
            print(f"  {rank}. metadata={metadata} content={item.page_content[:80]!r}")


def make_bundle(cfg, documents: list[Document] | None = None) -> IndexBundle:
    embeddings = KeywordEmbeddings()
    docs = clean_documents(documents or make_documents(), cfg.loaders)
    result = chunk_documents(docs, embeddings, cfg)
    store = create_vector_store(embeddings, cfg, result.chunks)
    docstore = JsonDocumentStore()
    docstore.add(result.parents)
    return IndexBundle(
        documents=docs,
        chunks=result.chunks,
        vector_store=store,
        embeddings=embeddings,
        docstore=docstore,
        cfg=cfg,
    )


class SearchTests(unittest.TestCase):
    def test_bm25_prefers_keyword_match(self) -> None:
        documents = make_documents()
        print(f"\n[bm25] corpus={len(documents)} docs, query='attention long documents', k=2")
        index = BM25Index.from_documents(documents)
        results = index.search("attention long documents", k=2)
        show("bm25 results", results)
        self.assertTrue(results)
        self.assertIn("transformer", results[0][0].page_content.lower())

    def test_hybrid_retriever(self) -> None:
        cfg = make_settings(
            search={
                "strategy": "hybrid",
                "k": 2,
                "fetch_k": 5,
                "parent_document": False,
                "dense_weight": 0.0,
                "sparse_weight": 1.0,
            }
        )
        bundle = make_bundle(cfg)
        print(f"\n[hybrid] dense_weight=0.0 sparse_weight=1.0 chunks={len(bundle.chunks)}")
        retriever = build_retriever(bundle, cfg, rerank=False)
        results = retriever.invoke("attention long documents")
        show("hybrid results", results)
        self.assertTrue(results)
        self.assertIn("transformer", results[0].page_content.lower())

    def test_weighted_rrf(self) -> None:
        a = Document(page_content="a", metadata={"doc_id": "a"})
        b = Document(page_content="b", metadata={"doc_id": "b"})
        c = Document(page_content="c", metadata={"doc_id": "c"})
        print("\n[rrf] rankers=[a,b] and [b,c], rrf_k=60")
        fused = weighted_rrf([[a, b], [b, c]], rrf_k=60)
        show("weighted_rrf results", fused)
        self.assertEqual([doc.metadata["doc_id"] for doc in fused], ["b", "a", "c"])

    def test_parent_child_expansion(self) -> None:
        text = (
            "Retrieval augmented generation combines a retriever with a language model. " * 8
            + " Hybrid search adds lexical BM25 scoring to dense embeddings. " * 8
        )
        cfg = make_settings(
            chunking={
                "strategy": "recursive",
                "chunk_size": 80,
                "chunk_overlap": 10,
                "parent_child": True,
                "parent_chunk_size": 400,
                "parent_chunk_overlap": 50,
            },
            search={
                "strategy": "sparse",
                "k": 2,
                "fetch_k": 5,
                "parent_document": True,
            },
        )
        bundle = make_bundle(
            cfg,
            [Document(page_content=text, metadata={"source": "rag.txt", "file_name": "rag.txt"})],
        )
        print(f"\n[parent_child] chunks={len(bundle.chunks)} parents={len(bundle.docstore)}")
        self.assertTrue(bundle.docstore)
        retriever = build_retriever(bundle, cfg, rerank=False)
        results = retriever.invoke("lexical BM25 scoring")
        show("parent-child results", results)
        self.assertTrue(results)
        self.assertTrue(any("parent_id" in (doc.metadata or {}) for doc in results))

    def test_llm_reranker_reorders(self) -> None:
        documents = [
            Document(page_content="first candidate", metadata={"doc_id": "1"}),
            Document(page_content="second candidate", metadata={"doc_id": "2"}),
        ]
        llm = FakeListChatModel(responses=['[{"index": 2, "score": 9}, {"index": 1, "score": 1}]'])
        reranker = LLMReranker(llm, max_candidates=5, top_n=2)
        print(f"\n[rerank] query='query' candidates={len(documents)}")
        results = reranker.rerank("query", documents, top_n=2)
        show("reranked results", results)
        self.assertEqual(results[0].metadata["doc_id"], "2")
        self.assertEqual(results[0].metadata["rerank_score"], 9.0)

    def test_multi_query_retriever(self) -> None:
        cfg = make_settings()
        bundle = make_bundle(cfg)
        base = build_retriever(bundle, cfg, rerank=False)
        llm = FakeListChatModel(
            responses=['["self attention long documents", "transformer context"]']
        )
        retriever = MultiQueryRetriever(base_retriever=base, llm=llm, count=2, k=2)
        print("\n[multi_query] query='attention' generated=['self attention long documents', 'transformer context']")
        results = retriever.invoke("attention")
        show("multi-query results", results)
        self.assertTrue(results)
        self.assertTrue(any("transformer" in doc.page_content.lower() for doc in results))


if __name__ == "__main__":
    unittest.main()
