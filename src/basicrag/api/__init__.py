"""Optional HTTP API (requires the ``api`` extra)."""

from .app import create_app

__all__ = ["create_app"]
