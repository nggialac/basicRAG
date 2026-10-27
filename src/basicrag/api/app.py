"""Optional FastAPI service.

Install extras first::

    pip install -e ".[api]"

Run with::

    basicrag serve --port 8000
    # or: uvicorn basicrag.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("The API requires FastAPI. Install it with: pip install -e '.[api]'") from exc

from ..config import Settings
from ..config import settings as default_settings
from ..ingestion import Indexer
from ..pipeline import AdvancedRAG


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(default=None, ge=1, le=50)
    include_content: bool = False


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(default=None, ge=1, le=50)


def create_app(cfg: Settings | None = None, *, rebuild: bool = False) -> FastAPI:
    cfg = cfg or default_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        indexer = Indexer(cfg)
        bundle = indexer.build(force=rebuild)
        app.state.rag = AdvancedRAG(bundle, cfg)
        app.state.indexer = indexer
        yield

    app = FastAPI(
        title="BasicRAG API",
        version="0.2.0",
        description="Advanced RAG service: hybrid search, reranking, citations.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        rag: AdvancedRAG = app.state.rag
        return {
            "status": "ok",
            "collection": cfg.vector_store.collection,
            "chunks": len(rag.bundle.chunks),
            "documents": len(rag.bundle.documents),
        }

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        try:
            documents = app.state.rag.retrieve(request.question, request.k)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "question": request.question,
            "results": [
                {
                    "index": index,
                    "content": document.page_content,
                    "source": (document.metadata or {}).get("source"),
                    "file_name": (document.metadata or {}).get("file_name"),
                    "page": (document.metadata or {}).get("page_number")
                    or (document.metadata or {}).get("page"),
                    "metadata": document.metadata,
                }
                for index, document in enumerate(documents, start=1)
            ],
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        try:
            result = app.state.rag.ask(request.question, request.k)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result.as_dict(include_content=request.include_content)

    @app.post("/index")
    def rebuild_index() -> dict[str, Any]:
        try:
            indexer = app.state.indexer
            bundle = indexer.build(force=True)
            app.state.rag = AdvancedRAG(bundle, cfg)
            return {"status": "rebuilt", "chunks": len(bundle.chunks)}
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


__all__ = ["app", "create_app", "AskRequest", "SearchRequest"]
