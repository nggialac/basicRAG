"""The assembled advanced RAG pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from ..config import Settings, settings
from ..llm import get_llm
from ..observability import get_logger, setup_logging
from ..prompts import format_context, get_prompt
from ..search import build_retriever
from .types import RAGResult, Source

logger = get_logger(__name__)

NO_ANSWER = "I don't know based on the provided documents."


def _score_of(document: Document) -> float | None:
    metadata = document.metadata or {}
    for key in ("rerank_score", "retrieval_score", "rrf_score", "bm25_score"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _to_source(index: int, document: Document) -> Source:
    metadata = document.metadata or {}
    page = metadata.get("page_number")
    if not isinstance(page, int):
        page = metadata.get("page")
        if isinstance(page, int):
            page += 1
    return Source(
        index=index,
        content=document.page_content,
        source=metadata.get("source"),
        file_name=metadata.get("file_name"),
        page=page if isinstance(page, int) else None,
        score=_score_of(document),
        metadata=metadata,
    )


class AdvancedRAG:
    """Retrieval, reranking, and grounded generation in one object."""

    def __init__(
        self,
        bundle: Any,
        cfg: Settings | None = None,
        *,
        llm: BaseChatModel | None = None,
        retriever: BaseRetriever | None = None,
    ) -> None:
        self.bundle = bundle
        self.cfg = cfg or getattr(bundle, "cfg", settings)
        self.llm = llm or get_llm(self.cfg)
        self._retriever = retriever
        self._retrievers: dict[int, BaseRetriever] = {}
        if self.cfg.pipeline.verbose:
            setup_logging(self.cfg.pipeline.log_level)

    @property
    def retriever(self) -> BaseRetriever:
        if self._retriever is None:
            self._retriever = build_retriever(self.bundle, self.cfg, llm=self.llm)
        return self._retriever

    def _retriever_for(self, k: int | None) -> BaseRetriever:
        if k is None or k == self.cfg.search.k:
            return self.retriever
        if k not in self._retrievers:
            search_cfg = self.cfg.search.model_copy(update={"k": k})
            rerank_cfg = self.cfg.rerank.model_copy(update={"top_n": k})
            cfg = self.cfg.model_copy(update={"search": search_cfg, "rerank": rerank_cfg})
            self._retrievers[k] = build_retriever(self.bundle, cfg, llm=self.llm)
        return self._retrievers[k]

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        """Return reranked source documents for a question."""
        return self._retriever_for(k).invoke(question)

    def search(self, question: str, k: int | None = None) -> list[Document]:
        return self.retrieve(question, k)

    def ask(self, question: str, k: int | None = None) -> RAGResult:
        """Answer a question and return the grounded citations."""
        documents = self.retrieve(question, k)
        if not documents:
            return RAGResult(question=question, answer=NO_ANSWER, sources=[])

        context = format_context(documents, self.cfg)
        chain: Runnable = get_prompt(self.cfg) | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        sources = [_to_source(index, document) for index, document in enumerate(documents, start=1)]
        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            contexts=documents,
        )

    def stream(self, question: str, k: int | None = None) -> Iterator[str]:
        """Stream answer tokens; call :meth:`retrieve` for citations."""
        documents = self.retrieve(question, k)
        if not documents:
            yield NO_ANSWER
            return

        context = format_context(documents, self.cfg)
        prompt = get_prompt(self.cfg)
        messages = prompt.format_messages(context=context, question=question)
        for chunk in self.llm.stream(messages):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                yield content


def build_chain(cfg: Settings | None = None, *, verbose: bool = True) -> Runnable:
    """Backward-compatible question -> answer ``Runnable``."""
    from .builder import build_rag

    rag = build_rag(cfg, verbose=verbose)
    return RunnableLambda(lambda question: rag.ask(str(question)).answer)


__all__ = ["AdvancedRAG", "build_chain", "NO_ANSWER"]
