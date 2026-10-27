"""Vector store backends.

Each backend exposes ``create`` / ``load`` / ``persist`` / ``delete`` /
``exists`` so the rest of the system never needs backend-specific logic.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from ..config import VectorStoreConfig
from ..core.registry import Registry
from ..observability import get_logger

logger = get_logger(__name__)

VECTOR_STORE_PROVIDERS: Registry[Any] = Registry("vector store provider")


class VectorStoreProvider(ABC):
    """Base class for a pluggable vector store backend."""

    def __init__(self, cfg: VectorStoreConfig) -> None:
        self.cfg = cfg

    @property
    def path(self) -> Path:
        return Path(self.cfg.persist_directory) / self.cfg.collection

    @abstractmethod
    def create(
        self, embeddings: Embeddings, documents: list[Document] | None = None
    ) -> VectorStore: ...

    @abstractmethod
    def load(self, embeddings: Embeddings) -> VectorStore | None: ...

    @abstractmethod
    def persist(self, store: VectorStore) -> None: ...

    @abstractmethod
    def delete(self) -> None: ...

    @abstractmethod
    def exists(self) -> bool: ...


@VECTOR_STORE_PROVIDERS.register("memory", aliases=("inmemory",))
class MemoryProvider(VectorStoreProvider):
    """In-process store. Fast for tests and small knowledge bases."""

    def create(
        self, embeddings: Embeddings, documents: list[Document] | None = None
    ) -> VectorStore:
        store = InMemoryVectorStore(embedding=embeddings)
        if documents:
            store.add_documents(documents)
        return store

    def load(self, embeddings: Embeddings) -> VectorStore | None:
        logger.warning("The in-memory vector store cannot be reloaded from disk")
        return None

    def persist(self, store: VectorStore) -> None:
        return None

    def delete(self) -> None:
        return None

    def exists(self) -> bool:
        return False


@VECTOR_STORE_PROVIDERS.register("faiss")
class FaissProvider(VectorStoreProvider):
    """Local, persistent FAISS index (recommended default)."""

    def _distance_strategy(self) -> Any:
        from langchain_community.vectorstores.utils import DistanceStrategy

        name = self.cfg.distance_strategy.lower().replace("-", "_")
        mapping = {
            "cosine": DistanceStrategy.COSINE,
            "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
            "l2": DistanceStrategy.EUCLIDEAN_DISTANCE,
            "max_inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
            "inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
            "ip": DistanceStrategy.MAX_INNER_PRODUCT,
            "dot": DistanceStrategy.DOT_PRODUCT,
            "dot_product": DistanceStrategy.DOT_PRODUCT,
        }
        if name not in mapping:
            raise ValueError(
                f"Unknown FAISS distance strategy '{self.cfg.distance_strategy}'. "
                "Use cosine, euclidean, max_inner_product, or dot_product."
            )
        return mapping[name]

    def create(
        self, embeddings: Embeddings, documents: list[Document] | None = None
    ) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        distance = self._distance_strategy()
        if documents:
            return FAISS.from_documents(
                documents,
                embedding=embeddings,
                distance_strategy=distance,
                normalize_L2=self.cfg.normalize_l2,
            )

        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore

        dimension = len(embeddings.embed_query("dimension probe"))
        if distance.name in {"COSINE", "MAX_INNER_PRODUCT", "DOT_PRODUCT"}:
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            distance_strategy=distance,
            normalize_L2=self.cfg.normalize_l2,
        )

    def load(self, embeddings: Embeddings) -> VectorStore | None:
        from langchain_community.vectorstores import FAISS

        if not (self.path / "index.faiss").exists():
            return None
        return FAISS.load_local(
            str(self.path),
            embeddings,
            index_name="index",
            allow_dangerous_deserialization=True,
            distance_strategy=self._distance_strategy(),
            normalize_L2=self.cfg.normalize_l2,
        )

    def persist(self, store: VectorStore) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.path), index_name="index")  # type: ignore[attr-defined]

    def delete(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)

    def exists(self) -> bool:
        return (self.path / "index.faiss").exists()


@VECTOR_STORE_PROVIDERS.register("chroma")
class ChromaProvider(VectorStoreProvider):
    """Chroma backend. Requires ``pip install chromadb``."""

    def _build(self, embeddings: Embeddings) -> VectorStore:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError as exc:
            raise ImportError(
                "The 'chroma' vector store provider requires chromadb. "
                "Install it with: pip install chromadb"
            ) from exc
        return Chroma(
            collection_name=self.cfg.collection,
            embedding_function=embeddings,
            persist_directory=str(self.path),
            **self.cfg.extra,
        )

    def create(
        self, embeddings: Embeddings, documents: list[Document] | None = None
    ) -> VectorStore:
        store = self._build(embeddings)
        if documents:
            store.add_documents(documents)
        return store

    def load(self, embeddings: Embeddings) -> VectorStore | None:
        if not self.path.exists():
            return None
        return self._build(embeddings)

    def persist(self, store: VectorStore) -> None:
        persist = getattr(store, "persist", None)
        if callable(persist):
            persist()

    def delete(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)

    def exists(self) -> bool:
        return self.path.exists()


@VECTOR_STORE_PROVIDERS.register("qdrant")
class QdrantProvider(VectorStoreProvider):
    """Qdrant backend. Requires ``langchain-qdrant`` and ``qdrant-client``."""

    def _client(self) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError(
                "The 'qdrant' vector store provider requires qdrant-client. "
                "Install it with: pip install qdrant-client"
            ) from exc
        if self.cfg.url:
            return QdrantClient(url=self.cfg.url)
        return QdrantClient(host=self.cfg.host or "localhost", port=self.cfg.port or 6333)

    def _build(self, embeddings: Embeddings, client: Any) -> VectorStore:
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError as exc:
            raise ImportError(
                "The 'qdrant' vector store provider requires langchain-qdrant. "
                "Install it with: pip install langchain-qdrant"
            ) from exc
        return QdrantVectorStore(
            client=client,
            collection_name=self.cfg.collection,
            embedding=embeddings,
            **self.cfg.extra,
        )

    def create(
        self, embeddings: Embeddings, documents: list[Document] | None = None
    ) -> VectorStore:
        client = self._client()
        store = self._build(embeddings, client)
        if documents:
            try:
                store.add_documents(documents)
            except Exception:
                # Collection may not exist yet; let langchain-qdrant create it.
                from langchain_qdrant import QdrantVectorStore

                store = QdrantVectorStore.from_documents(
                    documents,
                    embeddings,
                    client=client,
                    collection_name=self.cfg.collection,
                    **self.cfg.extra,
                )
        return store

    def load(self, embeddings: Embeddings) -> VectorStore | None:
        client = self._client()
        if not client.collection_exists(self.cfg.collection):
            return None
        return self._build(embeddings, client)

    def persist(self, store: VectorStore) -> None:
        return None

    def delete(self) -> None:
        client = self._client()
        if client.collection_exists(self.cfg.collection):
            client.delete_collection(self.cfg.collection)

    def exists(self) -> bool:
        return self._client().collection_exists(self.cfg.collection)
