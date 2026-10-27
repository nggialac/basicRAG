"""Chat model provider implementations."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from ..config import LLMConfig
from ..core.registry import Registry

LLM_PROVIDERS: Registry[Any] = Registry("llm provider")


def _common(cfg: LLMConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
    }
    if cfg.max_tokens is not None:
        params["max_tokens"] = cfg.max_tokens
    if cfg.timeout:
        params["timeout"] = cfg.timeout
    if cfg.max_retries:
        params["max_retries"] = cfg.max_retries
    return params


@LLM_PROVIDERS.register("openai_compat", aliases=("openai",))
def _openai(cfg: LLMConfig, **overrides: Any) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    params = _common(cfg)
    params["api_key"] = cfg.api_key.get_secret_value()
    if cfg.base_url:
        params["base_url"] = cfg.base_url
    params.update(cfg.extra)
    params.update(overrides)
    return ChatOpenAI(**params)


@LLM_PROVIDERS.register("ollama")
def _ollama(cfg: LLMConfig, **overrides: Any) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    params: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
    }
    if cfg.base_url:
        params["base_url"] = cfg.base_url
    if cfg.max_tokens is not None:
        params["num_predict"] = cfg.max_tokens
    params.update(cfg.extra)
    params.update(overrides)
    return ChatOllama(**params)


@LLM_PROVIDERS.register("anthropic", aliases=("claude",))
def _anthropic(cfg: LLMConfig, **overrides: Any) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' LLM provider requires langchain-anthropic. "
            "Install it with: pip install langchain-anthropic"
        ) from exc

    params = _common(cfg)
    params["api_key"] = cfg.api_key.get_secret_value()
    params.update(cfg.extra)
    params.update(overrides)
    return ChatAnthropic(**params)


@LLM_PROVIDERS.register("fake")
def _fake(cfg: LLMConfig, **overrides: Any) -> BaseChatModel:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    responses = overrides.pop(
        "responses",
        cfg.extra.get(
            "responses",
            ['{"queries": []}'],
        ),
    )
    return FakeListChatModel(responses=list(responses), **overrides)
