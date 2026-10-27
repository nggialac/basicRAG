"""End-to-end ingestion: load -> clean -> chunk -> embed -> persist."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..config import Settings, settings
from ..embeddings import get_embeddings
from ..observability import Timer, get_logger, setup_logging
from ..vectorstores import (
    create_vector_store,
    delete_vector_store,
    load_vector_store,
    persist_vector_store,
    vector_store_exists,
)
from .chunking import chunk_documents
from .cleaning import clean_documents
from .docstore import JsonDocumentStore, load_documents_jsonl, save_documents_jsonl
from .loaders import load_documents

logger = get_logger(__name__)


@dataclass
class IndexBundle:
    """Everything needed to answer questions without touching the source files."""

    documents: list[Document]
    chunks: list[Document]
    vector_store: VectorStore
    embeddings: Embeddings
    docstore: JsonDocumentStore
    cfg: Settings
    metadata: dict[str, object] = field(default_factory=dict)


class Indexer:
    """Build, persist, reload, and delete the document index."""

    def __init__(
        self,
        cfg: Settings | None = None,
        embeddings: Embeddings | None = None,
        *,
        verbose: bool | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.embeddings = embeddings or get_embeddings(self.cfg)
        self.verbose = self.cfg.pipeline.verbose if verbose is None else verbose
        if self.verbose:
            setup_logging(self.cfg.pipeline.log_level)

    @property
    def collection_dir(self) -> Path:
        return Path(self.cfg.vector_store.persist_directory) / self.cfg.vector_store.collection

    @property
    def chunks_path(self) -> Path:
        return self.collection_dir / "chunks.jsonl"

    @property
    def docstore_path(self) -> Path:
        return self.collection_dir / "docstore.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.collection_dir / "manifest.json"

    def exists(self) -> bool:
        return vector_store_exists(self.cfg) and self.chunks_path.exists()

    def _write_manifest(self, bundle: IndexBundle) -> None:
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "documents": len(bundle.documents),
            "chunks": len(bundle.chunks),
            "parents": len(bundle.docstore),
            "embedding_provider": self.cfg.embeddings.provider,
            "embedding_model": self.cfg.embeddings.model,
            "vector_store": self.cfg.vector_store.provider,
            "collection": self.cfg.vector_store.collection,
            "chunking": self.cfg.chunking.strategy,
            "parent_child": self.cfg.chunking.parent_child,
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def is_compatible(self) -> bool:
        """Return whether the persisted index matches the current settings."""
        if not self.manifest_path.exists():
            return True
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        expected = {
            "embedding_provider": self.cfg.embeddings.provider,
            "embedding_model": self.cfg.embeddings.model,
            "vector_store": self.cfg.vector_store.provider,
            "collection": self.cfg.vector_store.collection,
            "chunking": self.cfg.chunking.strategy,
            "parent_child": self.cfg.chunking.parent_child,
        }
        return all(manifest.get(key) == value for key, value in expected.items())

    def build(
        self,
        documents: list[Document] | None = None,
        *,
        force: bool = False,
    ) -> IndexBundle:
        """Build a fresh index, or load an existing one unless ``force``."""
        if not force and self.exists():
            if self.is_compatible():
                logger.info(
                    "Loading existing index for collection %s",
                    self.cfg.vector_store.collection,
                )
                return self.load()
            logger.warning(
                "Index settings changed for collection %s; rebuilding from source",
                self.cfg.vector_store.collection,
            )
            force = True

        timer = Timer()
        with timer:
            if documents is None:
                documents = load_documents(self.cfg)
            documents = clean_documents(documents, self.cfg.loaders)
            if not documents:
                raise ValueError(
                    "No documents were found. Check loaders.paths and loaders.extensions."
                )

            chunk_result = chunk_documents(documents, self.embeddings, self.cfg)
            if not chunk_result.chunks:
                raise ValueError("Chunking produced no chunks; check chunking settings.")

            vector_store = create_vector_store(self.embeddings, self.cfg, chunk_result.chunks)
            docstore = JsonDocumentStore(self.docstore_path)
            docstore.add(chunk_result.parents)

            bundle = IndexBundle(
                documents=documents,
                chunks=chunk_result.chunks,
                vector_store=vector_store,
                embeddings=self.embeddings,
                docstore=docstore,
                cfg=self.cfg,
                metadata={
                    "parents": len(chunk_result.parents),
                    "chunking": self.cfg.chunking.strategy,
                },
            )

            if self.cfg.vector_store.persist:
                persist_vector_store(vector_store, self.cfg)
                save_documents_jsonl(chunk_result.chunks, self.chunks_path)
                docstore.save()
                self._write_manifest(bundle)

        logger.info("Index build completed in %.1f ms", timer.elapsed_ms)
        return bundle

    def load(self) -> IndexBundle:
        """Load a persisted index and its chunk/doc metadata."""
        vector_store = load_vector_store(self.embeddings, self.cfg)
        chunks = load_documents_jsonl(self.chunks_path)
        docstore = JsonDocumentStore.load(self.docstore_path)
        return IndexBundle(
            documents=[],
            chunks=chunks,
            vector_store=vector_store,
            embeddings=self.embeddings,
            docstore=docstore,
            cfg=self.cfg,
            metadata={"loaded": True},
        )

    def get_or_build(self, *, force: bool = False) -> IndexBundle:
        return self.build(force=force)

    def delete(self) -> None:
        """Remove the persisted index and metadata."""
        delete_vector_store(self.cfg)
        if self.collection_dir.exists():
            shutil.rmtree(self.collection_dir)

    def stats(self) -> dict[str, object]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        bundle = self.load()
        return {
            "chunks": len(bundle.chunks),
            "parents": len(bundle.docstore),
        }


def build_index(cfg: Settings | None = None, *, force: bool = False) -> IndexBundle:
    """Convenience helper used by the CLI and pipeline builder."""
    return Indexer(cfg).build(force=force)
