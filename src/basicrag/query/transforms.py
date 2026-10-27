"""Query understanding: rewriting and hypothetical-document retrieval."""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from ..observability import get_logger
from ..search.fusion import weighted_rrf
from ..search.multi_query import message_text

logger = get_logger(__name__)

_REWRITE_PROMPT = """Rewrite the user's question as a clear, standalone search query.
Keep every name, number, and constraint. Do not answer the question.
Return only the rewritten query, without quotes or explanation.

Question: {question}
"""

_HYDE_PROMPT = """Write a short, factual passage that would answer the question.
This passage is only used to search a document index, so include likely keywords
and domain terms. Do not mention that it is hypothetical.

Question: {question}
"""


class RewriteRetriever(BaseRetriever):
    """Rewrite the query with an LLM before retrieval."""

    base_retriever: BaseRetriever
    llm: BaseChatModel
    max_chars: int = 512

    def _rewrite(self, question: str) -> str:
        try:
            prompt = ChatPromptTemplate.from_template(_REWRITE_PROMPT)
            response = self.llm.invoke(prompt.format_messages(question=question))
            rewritten = message_text(response).strip().strip("\"'")
            rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
            return rewritten[: self.max_chars] or question
        except Exception as exc:
            logger.warning("Query rewrite failed, using original query: %s", exc)
            return question

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return self.base_retriever.invoke(self._rewrite(query))


class HydeRetriever(BaseRetriever):
    """Hypothetical Document Embeddings: search with a generated answer."""

    base_retriever: BaseRetriever
    llm: BaseChatModel
    rrf_k: int = 60
    max_chars: int = 800
    k: int = 8

    def _hypothetical_document(self, question: str) -> str:
        try:
            prompt = ChatPromptTemplate.from_template(_HYDE_PROMPT)
            response = self.llm.invoke(prompt.format_messages(question=question))
            return message_text(response).strip()[: self.max_chars]
        except Exception as exc:
            logger.warning("HyDE generation failed, using original query: %s", exc)
            return ""

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        hypothetical = self._hypothetical_document(query)
        if not hypothetical:
            return self.base_retriever.invoke(query)
        original_documents = self.base_retriever.invoke(query)
        hypothetical_documents = self.base_retriever.invoke(hypothetical)
        return weighted_rrf(
            [original_documents, hypothetical_documents],
            [1.0, 1.0],
            rrf_k=self.rrf_k,
            limit=self.k,
        )
