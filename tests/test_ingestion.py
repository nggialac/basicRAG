from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import FakeEmbeddings

from basicrag import load_settings
from basicrag.ingestion import Indexer, clean_documents, load_documents


class IngestionTests(unittest.TestCase):
    def test_loads_text_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.txt").write_text(
                "This is a sufficiently long text document about retrieval.",
                encoding="utf-8",
            )
            (root / "b.md").write_text(
                "# Heading\n\nMarkdown content about reranking and search.",
                encoding="utf-8",
            )
            cfg = load_settings(
                loaders={
                    "paths": [str(root)],
                    "min_chars": 1,
                    "continue_on_error": True,
                }
            )
            documents = load_documents(cfg)
            self.assertEqual(len(documents), 2)
            names = {document.metadata.get("file_name") for document in documents}
            self.assertEqual(names, {"a.txt", "b.md"})

    def test_cleaning_deduplicates_and_enriches(self) -> None:
        from langchain_core.documents import Document

        cfg = load_settings(loaders={"min_chars": 1})
        documents = [
            Document(
                page_content="Same   content\n\n\nwith   whitespace.",
                metadata={"source": "x.txt"},
            ),
            Document(
                page_content="Same content\n\nwith whitespace.",
                metadata={"source": "x.txt"},
            ),
        ]
        cleaned = clean_documents(documents, cfg.loaders)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].page_content, "Same content\n\nwith whitespace.")
        self.assertIn("doc_id", cleaned[0].metadata)
        self.assertIn("content_hash", cleaned[0].metadata)

    def test_surrogates_are_repaired_and_stripped(self) -> None:
        from langchain_core.documents import Document

        from basicrag.core.ids import stable_id

        self.assertTrue(stable_id("lone \ud835 surrogate"))
        cfg = load_settings(loaders={"min_chars": 1})
        cleaned = clean_documents(
            [Document(page_content="Math \ud835\udc50 and lone \ud835 symbol", metadata={})],
            cfg.loaders,
        )
        self.assertEqual(len(cleaned), 1)
        text = cleaned[0].page_content
        self.assertIn("\U0001d450", text)
        self.assertIn("\ufffd", text)
        text.encode("utf-8")

    def test_manifest_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = load_settings(
                embeddings={"model": "model-a"},
                vector_store={
                    "provider": "faiss",
                    "persist_directory": temporary,
                    "collection": "test",
                },
            )
            indexer = Indexer(cfg, FakeEmbeddings(size=4), verbose=False)
            indexer.collection_dir.mkdir(parents=True)
            indexer.manifest_path.write_text(
                json.dumps(
                    {
                        "embedding_provider": cfg.embeddings.provider,
                        "embedding_model": cfg.embeddings.model,
                        "vector_store": cfg.vector_store.provider,
                        "collection": cfg.vector_store.collection,
                        "chunking": cfg.chunking.strategy,
                        "parent_child": cfg.chunking.parent_child,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(indexer.is_compatible())

            changed = cfg.model_copy(
                update={"embeddings": cfg.embeddings.model_copy(update={"model": "model-b"})}
            )
            changed_indexer = Indexer(changed, FakeEmbeddings(size=4), verbose=False)
            self.assertFalse(changed_indexer.is_compatible())


if __name__ == "__main__":
    unittest.main()
