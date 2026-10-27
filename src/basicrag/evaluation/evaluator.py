"""Evaluate retrieval quality against a labelled question set."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ..config import Settings, settings
from ..observability import get_logger
from ..search import build_retriever
from .metrics import average, relevance_metrics, token_f1

logger = get_logger(__name__)


@dataclass
class EvalCase:
    question: str
    relevant: list[str]
    reference_answer: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalCase:
        question = str(payload.get("question") or "").strip()
        relevant = (
            payload.get("relevant_ids")
            or payload.get("relevant")
            or payload.get("relevant_sources")
            or payload.get("sources")
            or []
        )
        if isinstance(relevant, str):
            relevant = [relevant]
        return cls(
            question=question,
            relevant=[str(item) for item in relevant],
            reference_answer=payload.get("reference_answer") or payload.get("answer"),
        )


@dataclass
class CaseResult:
    question: str
    retrieved: list[str]
    relevant: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    answer_f1: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "retrieved": self.retrieved,
            "relevant": self.relevant,
            "metrics": self.metrics,
        }
        if self.answer_f1 is not None:
            payload["answer_f1"] = self.answer_f1
        return payload


@dataclass
class EvalReport:
    cases: list[CaseResult]
    metrics: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "cases": [case.as_dict() for case in self.cases],
        }


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load JSON (array), JSONL, or a mapping with a ``cases`` key."""
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    payload: Any
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("cases") or payload.get("questions") or []
    return [
        EvalCase.from_dict(item)
        for item in payload
        if isinstance(item, dict) and item.get("question")
    ]


def _candidate_keys(document: Document) -> set[str]:
    metadata = document.metadata or {}
    keys: set[str] = set()
    for value in (
        document.id,
        metadata.get("doc_id"),
        metadata.get("chunk_id"),
        metadata.get("parent_id"),
        metadata.get("source_doc_id"),
        metadata.get("source"),
        metadata.get("file_name"),
        metadata.get("path"),
    ):
        if not value:
            continue
        text = str(value)
        keys.add(text.lower())
        name = Path(text).name
        if name:
            keys.add(name.lower())
    return keys


def _identity(document: Document) -> str:
    metadata = document.metadata or {}
    for key in ("chunk_id", "parent_id", "doc_id", "source_doc_id"):
        value = metadata.get(key)
        if value:
            return str(value).lower()
    if document.id:
        return str(document.id).lower()
    for key in ("source", "file_name", "path"):
        value = metadata.get(key)
        if value:
            return str(value).lower()
    return str(hash(document.page_content))


def _document_key_sets(documents: list[Document]) -> list[set[str]]:
    key_sets: list[set[str]] = []
    seen: set[str] = set()
    for document in documents:
        identity = _identity(document)
        if identity in seen:
            continue
        seen.add(identity)
        key_sets.append(_candidate_keys(document))
    return key_sets


def evaluate(
    bundle: Any,
    cases: list[EvalCase],
    cfg: Settings | None = None,
    *,
    k: int | None = None,
    rerank: bool | None = None,
    rag: Any | None = None,
) -> EvalReport:
    """Run retrieval (and optional answer) evaluation."""
    cfg = cfg or getattr(bundle, "cfg", settings)
    if k is not None:
        cfg = cfg.model_copy(
            update={
                "search": cfg.search.model_copy(update={"k": k}),
                "rerank": cfg.rerank.model_copy(update={"top_n": k}),
            }
        )
    retriever = build_retriever(bundle, cfg, rerank=rerank)
    top_k = k or cfg.search.k

    results: list[CaseResult] = []
    for case in cases:
        documents = retriever.invoke(case.question)
        key_sets = _document_key_sets(documents)
        relevant = {item.lower() for item in case.relevant}
        metrics = relevance_metrics(key_sets, relevant, top_k)

        answer_f1: float | None = None
        if case.reference_answer and rag is not None:
            answer_f1 = token_f1(rag.ask(case.question).answer, case.reference_answer)

        results.append(
            CaseResult(
                question=case.question,
                retrieved=[sorted(keys)[0] if keys else "" for keys in key_sets[:top_k]],
                relevant=case.relevant,
                metrics=metrics,
                answer_f1=answer_f1,
            )
        )

    metrics = average([result.metrics for result in results])
    if results and all(result.answer_f1 is not None for result in results):
        metrics["answer_f1"] = sum(result.answer_f1 or 0.0 for result in results) / len(results)
    return EvalReport(cases=results, metrics=metrics)


__all__ = [
    "EvalCase",
    "CaseResult",
    "EvalReport",
    "load_cases",
    "evaluate",
]
