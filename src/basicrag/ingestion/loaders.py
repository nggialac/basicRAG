"""Multi-format document loading with per-file error isolation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ..config import LoaderConfig, Settings, settings
from ..observability import get_logger, timed

logger = get_logger(__name__)


def _pdf(path: Path, cfg: LoaderConfig) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def _text(path: Path, cfg: LoaderConfig) -> list[Document]:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding=cfg.encoding).load()


def _csv(path: Path, cfg: LoaderConfig) -> list[Document]:
    from langchain_community.document_loaders import CSVLoader

    return CSVLoader(str(path), encoding=cfg.encoding).load()


def _docx(path: Path, cfg: LoaderConfig) -> list[Document]:
    from langchain_community.document_loaders import Docx2txtLoader

    return Docx2txtLoader(str(path)).load()


def _html(path: Path, cfg: LoaderConfig) -> list[Document]:
    from langchain_community.document_loaders import UnstructuredHTMLLoader

    return UnstructuredHTMLLoader(str(path)).load()


def _unstructured(path: Path, cfg: LoaderConfig) -> list[Document]:
    from langchain_community.document_loaders import UnstructuredFileLoader

    return UnstructuredFileLoader(str(path)).load()


LOADERS: dict[str, Callable[[Path, LoaderConfig], list[Document]]] = {
    ".pdf": _pdf,
    ".txt": _text,
    ".md": _text,
    ".markdown": _text,
    ".csv": _csv,
    ".docx": _docx,
    ".html": _html,
    ".htm": _html,
}


class DocumentCollectionLoader:
    """Load every configured file and attach consistent metadata."""

    def __init__(self, cfg: LoaderConfig | None = None) -> None:
        self.cfg = cfg or settings.loaders

    def _is_allowed(self, path: Path, root: Path) -> bool:
        if path.suffix.lower() not in {ext.lower() for ext in self.cfg.extensions}:
            return False
        if not self.cfg.include_hidden:
            try:
                parts = path.relative_to(root).parts
            except ValueError:
                parts = path.parts
            if any(part.startswith(".") for part in parts):
                return False
        return True

    def _iter_files(self) -> Iterator[Path]:
        seen: set[Path] = set()
        for raw in self.cfg.paths:
            path = Path(raw).expanduser()
            if not path.exists():
                logger.warning("Input path does not exist: %s", path)
                continue
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen and self._is_allowed(path, path.parent):
                    seen.add(resolved)
                    yield path
                continue

            glob = self.cfg.glob
            if self.cfg.recursive:
                candidates = path.rglob(glob)
            else:
                pattern = glob
                while pattern.startswith("**/"):
                    pattern = pattern[3:]
                candidates = path.glob(pattern)
            for file in sorted(candidates):
                if not file.is_file() or not self._is_allowed(file, path):
                    continue
                resolved = file.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield file

    def _load_one(self, path: Path) -> list[Document]:
        primary = LOADERS.get(path.suffix.lower(), _unstructured)
        try:
            documents = primary(path, self.cfg)
        except Exception as exc:  # optional parser missing or corrupt file
            if primary is _unstructured:
                raise
            logger.warning("Falling back to UnstructuredFileLoader for %s (%s)", path, exc)
            documents = _unstructured(path, self.cfg)

        for document in documents:
            metadata = dict(document.metadata or {})
            metadata.setdefault("source", str(path))
            metadata.setdefault("file_name", path.name)
            metadata.setdefault("file_type", path.suffix.lower())
            document.metadata = metadata
        return documents

    def load(self) -> list[Document]:
        documents: list[Document] = []
        for path in self._iter_files():
            try:
                with timed(logger, "load document", path=path):
                    loaded = self._load_one(path)
                documents.extend(loaded)
                logger.info("Loaded %s (%d page(s))", path.name, len(loaded))
            except Exception as exc:
                if not self.cfg.continue_on_error:
                    raise
                logger.error("Skipping %s: %s", path, exc)
        logger.info("Loaded %d document page(s)", len(documents))
        return documents


def get_loader(cfg: Settings | None = None, **overrides: Any) -> DocumentCollectionLoader:
    """Return a loader object with a ``load()`` method."""
    cfg = cfg or settings
    loader_cfg = cfg.loaders
    if overrides:
        loader_cfg = loader_cfg.model_copy(update=overrides)
    return DocumentCollectionLoader(loader_cfg)


def load_documents(cfg: Settings | None = None, **overrides: Any) -> list[Document]:
    """Load and return all configured documents."""
    return get_loader(cfg, **overrides).load()
