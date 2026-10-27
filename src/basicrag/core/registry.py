"""A tiny typed registry that powers every pluggable component.

Adding a provider or strategy is intentionally boring: write a function/class,
decorate it with ``@REGISTRY.register("name")`` and select it from config.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> object mapping with helpful errors and alias support."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str, *, aliases: tuple[str, ...] = ()) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            self._items[key] = item
            for alias in aliases:
                self._items[alias] = item
            return item

        return decorator

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.name} '{key}'. Available: {available}") from None

    def create(self, key: str, *args: object, **kwargs: object) -> T:
        factory = self.get(key)
        if not callable(factory):
            raise TypeError(f"{self.name} '{key}' is not callable")
        return factory(*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._items)
