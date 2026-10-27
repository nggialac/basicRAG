"""Reranker factory."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from ..config import Settings, settings
from ..core.registry import Registry
from .cross_encoder import CrossEncoderReranker
from .llm import LLMReranker
from .none import NoOpReranker

RERANKERS: Registry[Any] = Registry("reranker")


@RERANKERS.register("none", aliases=("off", "noop"))
def _none(cfg: Any, llm: BaseChatModel | None) -> NoOpReranker:
    return NoOpReranker()


@RERANKERS.register("llm", aliases=("listwise", "model"))
def _llm(cfg: Any, llm: BaseChatModel | None) -> LLMReranker:
    if llm is None:
        raise ValueError("LLM reranking requires a chat model")
    return LLMReranker(
        llm,
        max_candidates=cfg.max_candidates,
        max_chars=cfg.max_chars,
        top_n=cfg.top_n,
    )


@RERANKERS.register("cross_encoder", aliases=("crossencoder", "sentence_transformers"))
def _cross_encoder(cfg: Any, llm: BaseChatModel | None) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        cfg.model,
        device=cfg.device,
        max_candidates=cfg.max_candidates,
        top_n=cfg.top_n,
    )


def get_reranker(
    cfg: Settings | None = None,
    llm: BaseChatModel | None = None,
    provider: str | None = None,
    **overrides: Any,
):
    """Build a reranker from config (``none``, ``llm``, or ``cross_encoder``)."""
    cfg = cfg or settings
    rerank_cfg = cfg.rerank
    if overrides:
        rerank_cfg = rerank_cfg.model_copy(update=overrides)
    provider = provider or rerank_cfg.provider
    return RERANKERS.create(provider, rerank_cfg, llm)


__all__ = ["get_reranker", "RERANKERS"]
