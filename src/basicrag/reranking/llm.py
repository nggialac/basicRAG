"""LLM-based listwise reranking.

One extra chat call reorders the candidate pool. It works with any chat model,
including local OpenAI-compatible servers and Ollama, without a cross-encoder.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..observability import get_logger
from ..search.multi_query import message_text

logger = get_logger(__name__)

_RERANK_PROMPT = """You are a relevance judge for a retrieval system.
Rank the passages by how well they answer the question.
Use ONLY the supplied passages. Return ONLY a JSON array of objects, ordered
from most to least relevant, each shaped like:
[{{"index": 1, "score": 9}}, {{"index": 3, "score": 4}}]
"index" is the passage number shown below (1-based). Include every passage.

Question: {question}

Passages:
{passages}
"""


def _parse_ranking(text: str, count: int) -> list[tuple[int, float | None]]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # Some models add prose or single quotes; retry with a loose pattern.
        matches = re.findall(r"\{[^{}]*\}", text[start : end + 1])
        payload = []
        for match in matches:
            try:
                payload.append(json.loads(match))
            except json.JSONDecodeError:
                continue
    if not isinstance(payload, list):
        return []

    ranking: list[tuple[int, float | None]] = []
    for _position, item in enumerate(payload):
        index: int | None = None
        score: float | None = None
        if isinstance(item, int):
            index = item
        elif isinstance(item, dict):
            for key in ("index", "passage", "id", "rank"):
                value = item.get(key)
                if isinstance(value, int):
                    index = value
                    break
                if isinstance(value, str) and value.strip().isdigit():
                    index = int(value.strip())
                    break
            if index is None and isinstance(item.get("position"), int):
                index = item["position"]
            value = item.get("score")
            if isinstance(value, (int, float)):
                score = float(value)
        elif isinstance(item, str) and item.strip().isdigit():
            index = int(item.strip())
        if index is None:
            continue
        # Prompt uses 1-based indices.
        zero_based = index - 1 if index >= 1 else index
        if 0 <= zero_based < count:
            ranking.append((zero_based, score))
    return ranking


class LLMReranker:
    """Listwise LLM reranker with graceful fallback on malformed output."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        max_candidates: int = 30,
        max_chars: int = 1200,
        top_n: int = 5,
    ) -> None:
        self.llm = llm
        self.max_candidates = max(1, max_candidates)
        self.max_chars = max(200, max_chars)
        self.top_n = top_n
        self._prompt = ChatPromptTemplate.from_template(_RERANK_PROMPT)

    def _render_passages(self, documents: Sequence[Document]) -> str:
        blocks: list[str] = []
        for index, document in enumerate(documents, start=1):
            text = (document.page_content or "")[: self.max_chars]
            blocks.append(f"[{index}] {text}")
        return "\n\n".join(blocks)

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        limit = top_n or self.top_n
        candidates = list(documents)[: self.max_candidates]
        if len(candidates) <= 1:
            return candidates[:limit]

        try:
            messages = self._prompt.format_messages(
                question=query,
                passages=self._render_passages(candidates),
            )
            response = self.llm.invoke(messages)
            ranking = _parse_ranking(message_text(response), len(candidates))
        except Exception as exc:
            logger.warning("LLM reranking failed, keeping base order: %s", exc)
            return candidates[:limit]

        if not ranking:
            logger.debug("LLM reranker returned no usable ranking; keeping base order")
            return candidates[:limit]

        seen: set[int] = set()
        reranked: list[Document] = []
        for index, score in ranking:
            if index in seen:
                continue
            seen.add(index)
            metadata = {**(candidates[index].metadata or {}), "rerank_position": len(reranked)}
            if score is not None:
                metadata["rerank_score"] = score
            reranked.append(candidates[index].model_copy(update={"metadata": metadata}))
            if len(reranked) >= limit:
                break

        # Append candidates the model omitted, preserving base order.
        if len(reranked) < limit:
            for index, document in enumerate(candidates):
                if index in seen:
                    continue
                reranked.append(document)
                if len(reranked) >= limit:
                    break
        return reranked


__all__: list[Any] = ["LLMReranker"]
