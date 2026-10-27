"""Lightweight logging and timing helpers (no external dependencies)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_CONFIGURED = False


def setup_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Configure a readable root logger once per process."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class Timer:
    """Context manager that records elapsed milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


@contextmanager
def timed(logger: logging.Logger, label: str, **fields: Any) -> Iterator[Timer]:
    """Log the duration of a block at DEBUG level."""
    timer = Timer()
    with timer:
        yield timer
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.debug("%s finished in %.1f ms %s", label, timer.elapsed_ms, suffix)


def mask_secret(value: str | None) -> str:
    """Return a safe representation of a secret for logs/config output."""
    if not value:
        return ""
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}***{value[-2:]}"
