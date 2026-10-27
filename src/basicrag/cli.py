"""Command-line interface: index, search, ask, chat, eval, config, serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import Settings, load_settings
from .evaluation import evaluate, load_cases
from .ingestion import Indexer
from .pipeline import build_rag
from .pipeline.chat import chat as interactive_chat
from .pipeline.types import RAGResult


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Path to a YAML config file (default: configs/<profile>.yaml)",
    )
    parser.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        help="Config profile name, e.g. default or business",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Suppress informational logging",
    )


def _resolve_settings(args: argparse.Namespace, **overrides: Any) -> Settings:
    cfg = load_settings(
        config_file=getattr(args, "config", None),
        profile=getattr(args, "profile", None),
    )
    quiet = bool(getattr(args, "quiet", False))
    verbose = bool(getattr(args, "verbose", False))
    if quiet or verbose:
        cfg = cfg.model_copy(
            update={"pipeline": cfg.pipeline.model_copy(update={"verbose": not quiet})}
        )
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg


def _print_sources(result: RAGResult) -> None:
    if not result.sources:
        return
    print("\nSources:")
    for source in result.sources:
        location = source.file_name or source.source or "document"
        page = f", page {source.page}" if source.page else ""
        score = f" (score {source.score:.3f})" if source.score is not None else ""
        print(f"  [{source.index}] {location}{page}{score}")


def _cmd_index(args: argparse.Namespace) -> int:
    cfg = _resolve_settings(args)
    if getattr(args, "paths", None):
        cfg = cfg.model_copy(
            update={"loaders": cfg.loaders.model_copy(update={"paths": args.paths})}
        )
    indexer = Indexer(cfg)
    bundle = indexer.build(force=args.rebuild)
    stats = (
        indexer.stats()
        if cfg.vector_store.persist
        else {
            "chunks": len(bundle.chunks),
            "parents": len(bundle.docstore),
        }
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    cfg = _resolve_settings(args)
    rag = build_rag(cfg, verbose=cfg.pipeline.verbose)
    documents = rag.retrieve(" ".join(args.question), k=args.k)
    if args.json:
        payload = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata or {}
            payload.append(
                {
                    "index": index,
                    "source": metadata.get("source"),
                    "file_name": metadata.get("file_name"),
                    "page": metadata.get("page_number") or metadata.get("page"),
                    "content": document.page_content,
                    "metadata": {
                        key: metadata.get(key)
                        for key in (
                            "rrf_score",
                            "retrieval_score",
                            "bm25_score",
                            "rerank_score",
                            "parent_id",
                        )
                        if metadata.get(key) is not None
                    },
                }
            )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        name = metadata.get("file_name") or metadata.get("source") or "document"
        print(f"\n[{index}] {name}")
        print(document.page_content[:1000])
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    cfg = _resolve_settings(args)
    rag = build_rag(cfg, verbose=cfg.pipeline.verbose)
    result = rag.ask(" ".join(args.question), k=args.k)
    if args.json:
        print(
            json.dumps(
                result.as_dict(include_content=args.content),
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(result.answer)
        _print_sources(result)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    cfg = _resolve_settings(args)
    interactive_chat(cfg, verbose=cfg.pipeline.verbose, rebuild=args.rebuild)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    cfg = _resolve_settings(args)
    rag = build_rag(cfg, verbose=cfg.pipeline.verbose)
    cases = load_cases(args.dataset)
    report = evaluate(
        rag.bundle,
        cases,
        rag.cfg,
        k=args.k,
        rerank=False if args.no_rerank else None,
        rag=rag if args.answers else None,
    )
    payload = report.as_dict()
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = _resolve_settings(args)
    print(cfg.model_dump_json(indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "Serving requires the API extras: pip install -e '.[api]'",
            file=sys.stderr,
        )
        return 1
    from .api.app import create_app

    cfg = _resolve_settings(args)
    app = create_app(cfg, rebuild=args.rebuild)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_common(common)

    parser = argparse.ArgumentParser(
        prog="basicrag",
        description="Advanced, modular RAG system for business knowledge bases.",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser(
        "index", parents=[common], help="Build or load the document index"
    )
    index.add_argument("--rebuild", action="store_true", help="Ignore existing index")
    index.add_argument(
        "--path",
        dest="paths",
        action="append",
        help="Override loaders.paths (can be repeated)",
    )
    index.set_defaults(func=_cmd_index)

    search = subparsers.add_parser("search", parents=[common], help="Inspect retrieved chunks")
    search.add_argument("question", nargs="+")
    search.add_argument("-k", type=int, default=None, help="Number of results")
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=_cmd_search)

    ask = subparsers.add_parser("ask", parents=[common], help="Ask one question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("-k", type=int, default=None, help="Number of sources")
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--content", action="store_true", help="Include source text in JSON output")
    ask.set_defaults(func=_cmd_ask)

    chat = subparsers.add_parser("chat", parents=[common], help="Interactive chat")
    chat.add_argument("--rebuild", action="store_true")
    chat.set_defaults(func=_cmd_chat)

    evaluation = subparsers.add_parser(
        "eval", parents=[common], help="Evaluate retrieval against a JSON/JSONL set"
    )
    evaluation.add_argument("dataset", help="Path to evaluation cases")
    evaluation.add_argument("-k", type=int, default=None)
    evaluation.add_argument("--no-rerank", action="store_true")
    evaluation.add_argument("--answers", action="store_true", help="Also score reference answers")
    evaluation.add_argument("--output", help="Write the full report to this path")
    evaluation.set_defaults(func=_cmd_eval)

    config = subparsers.add_parser(
        "config", parents=[common], help="Print the effective configuration"
    )
    config.set_defaults(func=_cmd_config)

    serve = subparsers.add_parser("serve", parents=[common], help="Serve the optional FastAPI app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--rebuild", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
