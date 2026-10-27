"""Offline retrieval and answer metrics (no external dependencies)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence


def _unique(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(_unique(retrieved)[:k]) & relevant else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    top = _unique(retrieved)[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(_unique(retrieved)[:k]) & relevant) / len(relevant)


def mrr_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(_unique(retrieved)[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    top = _unique(retrieved)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, item in enumerate(top, start=1) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(retrieved: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    return {
        f"hit_rate@{k}": hit_rate_at_k(retrieved, relevant, k),
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"mrr@{k}": mrr_at_k(retrieved, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }


def relevance_metrics(
    document_keys: Sequence[set[str]], relevant: set[str], k: int
) -> dict[str, float]:
    """Document-level metrics where one document can match any gold label."""
    top = list(document_keys)[:k]
    relevance = [1 if keys & relevant else 0 for keys in top]
    matched = set()
    for keys in top:
        matched |= keys & relevant

    precision = sum(relevance) / len(top) if top else 0.0
    recall = len(matched) / len(relevant) if relevant else 0.0
    hit = 1.0 if any(relevance) else 0.0
    reciprocal = next((1.0 / rank for rank, flag in enumerate(relevance, start=1) if flag), 0.0)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, flag in enumerate(relevance, start=1) if flag)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        f"hit_rate@{k}": hit,
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"mrr@{k}": reciprocal,
        f"ndcg@{k}": ndcg,
    }


_WORD = re.compile(r"\w+", re.UNICODE)


def token_f1(prediction: str, reference: str) -> float:
    predicted = _WORD.findall(prediction.lower())
    expected = _WORD.findall(reference.lower())
    if not predicted or not expected:
        return 0.0
    common: dict[str, int] = {}
    for token in expected:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in predicted:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def average(metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: sum(item.get(key, 0.0) for item in metrics) / len(metrics) for key in keys}


__all__ = [
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "retrieval_metrics",
    "relevance_metrics",
    "token_f1",
    "exact_match",
    "average",
]
