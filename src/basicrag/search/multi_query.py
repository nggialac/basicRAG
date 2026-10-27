"""LLM-driven query expansion (multi-query and sub-question decomposition)."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from ..observability import get_logger
from .fusion import weighted_rrf

logger = get_logger(__name__)

_VARIANTS_PROMPT = """You expand a search query for a retrieval system.
Generate {count} diverse, self-contained search queries that would find relevant
passages for the user's question. Preserve names, numbers, and domain terms.
Return ONLY a JSON array of strings, for example: ["query one", "query two"].

Question: {question}
"""

_SUBQUESTION_PROMPT = """You decompose a complex question for a retrieval system.
Write {count} standalone sub-questions that together cover every part of the
user's question. Each must be understandable on its own.
Return ONLY a JSON array of strings.

Question: {question}
"""


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _extract_strings(text: str) -> list[str]:
    """Parse a JSON array, an object with ``queries``, or bullet lines."""
    candidates: list[Any] = []
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            candidates = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            candidates = []
    if not candidates:
        brace_start, brace_end = text.find("{"), text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                payload = json.loads(text[brace_start : brace_end + 1])
                candidates = payload.get("queries") or payload.get("questions") or []
            except json.JSONDecodeError:
                candidates = []
    if not candidates:
        candidates = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip(" \"'") for line in text.splitlines()
        ]
    if not isinstance(candidates, list):
        return []
    queries: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            value = candidate.strip().strip("\"'")
            if value:
                queries.append(value)
    return queries


class MultiQueryRetriever(BaseRetriever):
    """Run several generated queries and fuse their results with RRF."""

    base_retriever: BaseRetriever
    llm: BaseChatModel
    count: int = 3
    mode: str = "variants"
    rrf_k: int = 60
    k: int = 8

    def _generate_queries(self, question: str) -> list[str]:
        template = _SUBQUESTION_PROMPT if self.mode == "subquestions" else _VARIANTS_PROMPT
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(question=question, count=self.count)
        response: BaseMessage = self.llm.invoke(messages)
        return _extract_strings(message_text(response))[: self.count]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        queries = [query]
        try:
            queries.extend(self._generate_queries(query))
        except Exception as exc:
            logger.warning("Query expansion failed, using original query: %s", exc)

        seen: set[str] = set()
        unique_queries: list[str] = []
        for item in queries:
            normalized = item.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_queries.append(normalized)

        result_lists: list[list[Document]] = []
        for item in unique_queries:
            try:
                result_lists.append(self.base_retriever.invoke(item))
            except Exception as exc:
                logger.warning("Retrieval failed for expanded query %r: %s", item, exc)

        if not result_lists:
            return []
        return weighted_rrf(result_lists, rrf_k=self.rrf_k, limit=self.k)
