"""Prompt templates and context formatting."""

from ..config import DEFAULT_ANSWER_TEMPLATE, DEFAULT_SYSTEM_PROMPT
from .rag import format_context, get_prompt, source_label

TEMPLATE = DEFAULT_ANSWER_TEMPLATE

__all__ = [
    "get_prompt",
    "format_context",
    "source_label",
    "TEMPLATE",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_ANSWER_TEMPLATE",
]
