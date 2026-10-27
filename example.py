"""Python API example.

Run from the project root after configuring embeddings/LLM in ``.env`` or a
``configs/*.yaml`` profile::

    python example.py
"""

from basicrag import AdvancedRAG, Indexer, load_settings


def main() -> None:
    cfg = load_settings()  # or load_settings(profile="business")
    indexer = Indexer(cfg)
    bundle = indexer.build()  # loads the persisted index if it exists
    rag = AdvancedRAG(bundle, cfg)

    question = "What is the main contribution of the RAG empirical study?"
    result = rag.ask(question)
    print(result.answer)

    print("\nSources:")
    for source in result.sources:
        name = source.file_name or source.source or "document"
        page = f", page {source.page}" if source.page else ""
        print(f"[{source.index}] {name}{page} score={source.score}")


if __name__ == "__main__":
    main()
