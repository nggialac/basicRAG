"""Factories that assemble the full pipeline from configuration."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from ..config import Settings, settings
from ..ingestion import IndexBundle, Indexer
from .rag import AdvancedRAG


def build_indexer(cfg: Settings | None = None, **kwargs: object) -> Indexer:
    return Indexer(cfg, **kwargs)  # type: ignore[arg-type]


def build_rag(
    cfg: Settings | None = None,
    *,
    rebuild: bool = False,
    llm: BaseChatModel | None = None,
    verbose: bool | None = None,
) -> AdvancedRAG:
    """Load or build the index and return a ready-to-use RAG pipeline."""
    cfg = cfg or settings
    if verbose is not None:
        cfg = cfg.model_copy(
            update={"pipeline": cfg.pipeline.model_copy(update={"verbose": verbose})}
        )
    indexer = Indexer(cfg, verbose=verbose)
    bundle = indexer.build(force=rebuild)
    return AdvancedRAG(bundle, cfg, llm=llm)


def build_from_bundle(
    bundle: IndexBundle,
    cfg: Settings | None = None,
    *,
    llm: BaseChatModel | None = None,
) -> AdvancedRAG:
    return AdvancedRAG(bundle, cfg or bundle.cfg, llm=llm)


__all__ = ["build_rag", "build_indexer", "build_from_bundle"]
