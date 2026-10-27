"""Offline evaluation helpers."""

from .evaluator import EvalCase, EvalReport, evaluate, load_cases
from .metrics import (
    average,
    exact_match,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevance_metrics,
    retrieval_metrics,
    token_f1,
)

__all__ = [
    "EvalCase",
    "EvalReport",
    "load_cases",
    "evaluate",
    "retrieval_metrics",
    "relevance_metrics",
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "token_f1",
    "exact_match",
    "average",
]
