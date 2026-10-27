"""Deprecated compatibility module. Use :mod:`basicrag.prompts`."""

from .prompts import TEMPLATE, format_context, get_prompt, source_label

__all__ = ["get_prompt", "format_context", "source_label", "TEMPLATE"]
