"""Cross-encoder reranking (optional dependency)."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from ..observability import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Score (query, passage) pairs with a sentence-transformers model."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
        max_candidates: int = 30,
        top_n: int = 5,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._cross_encoder_cls = CrossEncoder
        self.model_name = model
        self.device = device
        self.max_candidates = max(1, max_candidates)
        self.top_n = top_n
        self.batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading cross-encoder %s", self.model_name)
                    self._model = self._cross_encoder_cls(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        limit = top_n or self.top_n
        candidates = list(documents)[: self.max_candidates]
        if not candidates:
            return []

        model = self._load()
        pairs = [(query, document.page_content or "") for document in candidates]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        results: list[Document] = []
        for position, (document, score) in enumerate(ranked[:limit]):
            metadata = {
                **(document.metadata or {}),
                "rerank_score": round(float(score), 6),
                "rerank_position": position,
            }
            results.append(document.model_copy(update={"metadata": metadata}))
        return results


__all__ = ["CrossEncoderReranker"]
