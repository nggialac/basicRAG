from __future__ import annotations

import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from test_search import make_bundle, make_settings

from basicrag import AdvancedRAG
from basicrag.evaluation import EvalCase, evaluate


class PipelineTests(unittest.TestCase):
    def test_ask_returns_answer_and_sources(self) -> None:
        cfg = make_settings()
        bundle = make_bundle(cfg)
        llm = FakeListChatModel(responses=["The answer is grounded [1]."])
        rag = AdvancedRAG(bundle, cfg, llm=llm)
        result = rag.ask("What handles long documents?")
        self.assertEqual(result.answer, "The answer is grounded [1].")
        self.assertTrue(result.sources)
        self.assertIn("doc_id", result.sources[0].metadata)

    def test_evaluation_reports_metrics(self) -> None:
        cfg = make_settings()
        bundle = make_bundle(cfg)
        cases = [
            EvalCase(
                question="attention long documents",
                relevant=["transformer.txt"],
            )
        ]
        report = evaluate(bundle, cases, cfg, k=2, rerank=False)
        self.assertIn("hit_rate@2", report.metrics)
        self.assertEqual(report.metrics["hit_rate@2"], 1.0)
        self.assertEqual(len(report.cases), 1)


if __name__ == "__main__":
    unittest.main()
