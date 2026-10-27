"""Interactive terminal chat."""

from __future__ import annotations

from ..config import Settings
from .builder import build_rag
from .types import RAGResult


def print_result(result: RAGResult, *, show_sources: bool = True) -> None:
    print("\n" + result.answer.strip() + "\n")
    if show_sources and result.sources:
        print("Sources:")
        for source in result.sources:
            location = source.file_name or source.source or "document"
            page = f", page {source.page}" if source.page else ""
            score = f" (score {source.score:.3f})" if source.score is not None else ""
            print(f"  [{source.index}] {location}{page}{score}")


def chat(cfg: Settings | None = None, *, verbose: bool = True, rebuild: bool = False) -> None:
    """Build the pipeline and answer questions until the user exits."""
    rag = build_rag(cfg, verbose=verbose, rebuild=rebuild)
    show_sources = True
    print("Advanced RAG ready. Commands: /sources, /search <query>, /help, exit")

    while True:
        try:
            user_input = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            print("Exiting.")
            break
        if user_input.lower() == "/help":
            print(
                "/sources  toggle source list\n/search <query>  show retrieved chunks\nexit  quit"
            )
            continue
        if user_input.lower() == "/sources":
            show_sources = not show_sources
            print(f"Sources {'shown' if show_sources else 'hidden'}.")
            continue
        if user_input.lower().startswith("/search "):
            query = user_input[len("/search ") :].strip()
            for index, document in enumerate(rag.retrieve(query), start=1):
                metadata = document.metadata or {}
                name = metadata.get("file_name") or metadata.get("source") or "?"
                print(f"[{index}] {name}\n{document.page_content[:400]}\n")
            continue

        result = rag.ask(user_input)
        print_result(result, show_sources=show_sources)


__all__ = ["chat", "print_result"]
