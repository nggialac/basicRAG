"""Chat model factory."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from ..config import Settings, settings
from .providers import LLM_PROVIDERS


def get_llm(
    cfg: Settings | None = None,
    provider: str | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """Build a chat model from config or explicit overrides."""
    cfg = cfg or settings
    provider = provider or cfg.llm.provider
    return LLM_PROVIDERS.create(provider, cfg.llm, **overrides)


__all__ = ["get_llm", "LLM_PROVIDERS"]
