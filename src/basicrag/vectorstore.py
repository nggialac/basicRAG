"""Deprecated compatibility module. Use :mod:`basicrag.vectorstores`."""

from .vectorstores import (
    VECTOR_STORE_PROVIDERS,
    build_vector_store,
    create_vector_store,
    delete_vector_store,
    get_vector_store,
    load_vector_store,
    persist_vector_store,
    vector_store_exists,
)

__all__ = [
    "get_vector_store",
    "create_vector_store",
    "build_vector_store",
    "load_vector_store",
    "persist_vector_store",
    "delete_vector_store",
    "vector_store_exists",
    "VECTOR_STORE_PROVIDERS",
]
