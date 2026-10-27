from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from basicrag import load_settings


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = load_settings()
        self.assertEqual(cfg.llm.model, "qwen3.8-fn")
        self.assertEqual(cfg.embeddings.provider, "ollama")
        self.assertEqual(cfg.vector_store.provider, "faiss")
        self.assertEqual(cfg.search.strategy, "hybrid")

    def test_explicit_override_wins(self) -> None:
        cfg = load_settings(search={"k": 3, "strategy": "dense"})
        self.assertEqual(cfg.search.k, 3)
        self.assertEqual(cfg.search.strategy, "dense")

    def test_nested_environment_variable(self) -> None:
        with patch.dict(os.environ, {"BASICRAG_SEARCH__K": "9"}, clear=False):
            cfg = load_settings()
        self.assertEqual(cfg.search.k, 9)

    def test_legacy_environment_variable(self) -> None:
        with patch.dict(os.environ, {"BASICRAG_RETRIEVER_K": "7"}, clear=False):
            cfg = load_settings()
        self.assertEqual(cfg.search.k, 7)

    def test_legacy_path_is_wrapped_in_list(self) -> None:
        with patch.dict(os.environ, {"BASICRAG_PAPERS_DIR": "./custom"}, clear=False):
            cfg = load_settings()
        self.assertEqual(cfg.loaders.paths, ["./custom"])

    def test_missing_explicit_config_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_settings(config_file="does-not-exist.yaml")


if __name__ == "__main__":
    unittest.main()
