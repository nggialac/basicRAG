# BasicRAG — Advanced, Modular RAG

A business-oriented RAG toolkit with pluggable ingestion, embeddings, vector
stores, hybrid search, reranking, query transformation, citations, evaluation,
and an optional HTTP API. Every stage is an independent, registry-backed
extension point, so changing a model or strategy is a config edit rather than a
pipeline rewrite.

## Pipeline

```
documents ──▶ load ──▶ clean ──▶ chunk ──▶ embed ──▶ vector store
                                                        │
question ──▶ rewrite / HyDE ──▶ dense + BM25 hybrid ─────┘
                                        │
                                        ▼
                              parent expansion ──▶ rerank ──▶ grounded answer + sources
```

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An LLM and embedding backend. The default profile uses a local
  [Ollama](https://ollama.com) server at `http://127.0.0.1:11434`:

  ```powershell
  ollama pull llama3.2
  ollama pull embeddinggemma
  ```

  To use an OpenAI-compatible endpoint instead, set
  `BASICRAG_LLM__PROVIDER=openai_compat` together with
  `BASICRAG_LLM__BASE_URL` and `BASICRAG_LLM__API_KEY` (see `.env.example`).

## Quick start

```powershell
# from BasicRAG/
uv sync
Copy-Item .env.example .env    # optional: adjust models or endpoint

# Commands below assume the virtualenv is activated. Without activation,
# prefix them with `uv run` (for example: `uv run basicrag index`).

# Build the persistent index (reads ./papers)
basicrag index

# One question
basicrag ask "What is the main contribution of the RAG empirical study?"

# Interactive chat
basicrag chat

# Inspect retrieved chunks
basicrag search "efficient transformer long documents" --json

# Evaluate against labelled questions
basicrag eval tests/data/eval_cases.json --output eval-report.json
```

The repository ships two small sample papers in `papers/` so the quick start
works out of the box. Add your own PDF/TXT/MD/HTML/DOCX files there (or pass
`basicrag index --path <dir>`) and rerun with `--rebuild` to include them. The
larger ACL 2026 test corpus is deliberately not committed; download it into
`papers/` if you want to reproduce that setup.

Or use Python (see `example.py` for the runnable version):

```python
from basicrag import AdvancedRAG, Indexer, load_settings

cfg = load_settings()                  # or load_settings(profile="business")
bundle = Indexer(cfg).build()          # load persisted index if present
rag = AdvancedRAG(bundle, cfg)
result = rag.ask("How does hybrid retrieval work?")
print(result.answer)
for source in result.sources:
    print(source.file_name, source.page, source.score)
```

## Configuration

Settings are validated with Pydantic. Priority, highest first:

1. Factory keyword overrides (`get_llm(model="...")`)
2. Nested environment variables: `BASICRAG_SEARCH__STRATEGY=hybrid`
3. Legacy variables: `BASICRAG_RETRIEVER_K`, `BASICRAG_LLM_MODEL`, …
4. `.env`
5. YAML profile: `configs/<profile>.yaml`, selected with `--profile business`
6. Code defaults

```powershell
basicrag --profile business index
basicrag config --profile business     # print effective settings
```

Environment variable naming uses `__` for nesting:

```powershell
$env:BASICRAG_LLM__MODEL = "qwen3.8-fn"
$env:BASICRAG_VECTOR_STORE__PROVIDER = "faiss"
$env:BASICRAG_RERANK__PROVIDER = "cross_encoder"
```

Set `BASICRAG_CONFIG=/path/to/config.yaml` to load an explicit file.

## Project layout

```
src/basicrag/
├── config.py              validated settings + YAML/env sources
├── cli.py                 index / ask / search / chat / eval / config / serve
├── core/                  registries, protocols, stable document ids
├── ingestion/
│   ├── loaders.py         per-format loading with error isolation
│   ├── cleaning.py        normalization, metadata, de-duplication
│   ├── chunking.py        recursive / fixed / token / markdown / semantic
│   ├── docstore.py        JSONL parent-document store
│   └── indexer.py         load → clean → chunk → embed → persist
├── llm/                   Ollama / OpenAI-compatible chat models
├── embeddings/            Ollama / OpenAI / HuggingFace + SQLite cache
├── vectorstores/          memory / FAISS / Chroma / Qdrant
├── search/
│   ├── dense.py           similarity / MMR / score threshold
│   ├── sparse.py          dependency-free BM25
│   ├── hybrid.py          dense + lexical fusion
│   ├── fusion.py          Reciprocal Rank Fusion
│   ├── multi_query.py     query expansion / decomposition
│   ├── parent_document.py parent chunk expansion
│   └── factory.py         composes the full retrieval stack
├── query/                 query rewriting and HyDE
├── reranking/             LLM listwise / cross-encoder / no-op
├── prompts/               citation-aware context formatting
├── pipeline/              RAG answers, streaming, chat, builders
├── evaluation/            hit rate, precision, recall, MRR, nDCG, F1
├── observability/         logging, timing, secret masking
└── api/                   optional FastAPI service
```

Old import paths (`basicrag.retriever`, `basicrag.vectorstore`,
`basicrag.splitter`, `basicrag.prompt`, `basicrag.loaders`, `basicrag.chat`)
remain as compatibility shims.

## Ingestion options

Choose a loader source and chunking strategy in YAML:

```yaml
loaders:
  paths: ["./papers", "./contracts"]
  extensions: [".pdf", ".txt", ".md", ".docx", ".csv", ".html"]

chunking:
  strategy: recursive       # recursive | fixed | token | markdown | semantic
  chunk_size: 800
  chunk_overlap: 120
  parent_child: true        # index small children, return larger parents
  parent_chunk_size: 2400
```

`semantic` chunking uses the embedding model and is slower; `recursive` is the
recommended production default.

## Search and reranking

```yaml
search:
  strategy: hybrid          # dense | sparse | hybrid | multi_query
  dense_search_type: similarity   # similarity | mmr | similarity_score_threshold
  k: 10                     # final results
  fetch_k: 60               # candidate pool before reranking
  dense_weight: 0.55
  sparse_weight: 0.45
  parent_document: true

rerank:
  enabled: true
  provider: llm             # none | llm | cross_encoder
  top_n: 10                 # 0 = use search.k
  max_candidates: 40

query:
  rewrite: true
  hyde: false
```

`hybrid` fuses dense and BM25 results with weighted RRF. `multi_query` asks the
LLM for alternative queries or sub-questions, then fuses the results.

Cross-encoder reranking is optional:

```powershell
uv sync --extra rerank
```

## Vector stores

| Provider | Persistence | Notes |
| --- | --- | --- |
| `memory` | no | fastest for tests / small sets |
| `faiss` | yes | default, local files |
| `chroma` | yes | `uv sync --extra chroma` |
| `qdrant` | server | `uv sync --extra qdrant` |

FAISS indexes are stored under
`data/vector_store/<collection>/` together with `chunks.jsonl`,
`docstore.jsonl`, and `manifest.json`. Loading a FAISS index uses trusted local
files (`allow_dangerous_deserialization=True`).

`basicrag index` reuses an existing index only when the manifest matches the
current embedding provider/model, vector store, collection, chunking strategy,
and parent-child setting. Otherwise it rebuilds automatically. Changes to source
documents still require `basicrag index --rebuild`.

## HTTP API

```powershell
uv sync --extra api
uv run basicrag serve --port 8000
```

Endpoints: `GET /health`, `POST /search`, `POST /ask`, `POST /index`.
Interactive docs are at `http://127.0.0.1:8000/docs`.

## Evaluation

Create a JSON or JSONL file:

```json
[
  {
    "question": "What is retrieval-augmented generation?",
    "relevant_ids": ["paper1_rag_empirical_study.pdf:0:c3"],
    "reference_answer": "Optional reference answer"
  }
]
```

```powershell
basicrag eval tests/data/eval_cases.json -k 10 --answers
```

Metrics: `hit_rate@k`, `precision@k`, `recall@k`, `mrr@k`, `ndcg@k`, and optional
answer `token_f1`.

## Extending a stage

Each registry lives next to its implementations. For example, add an LLM:

```python
# src/basicrag/llm/providers.py
@LLM_PROVIDERS.register("my_provider")
def _my_provider(cfg, **overrides):
    from my_sdk import MyChatModel
    return MyChatModel(model=cfg.model, **overrides)
```

Then set `llm.provider: my_provider`. The same pattern applies to
`EMBEDDING_PROVIDERS`, `VECTOR_STORE_PROVIDERS`, `CHUNKERS`, `SEARCH_STRATEGIES`,
and `RERANKERS`.

## Tests

The test suite is fully offline (fake embeddings, in-memory vectors, fake LLM):

```powershell
python -m unittest discover -s tests -v
uv run pytest -q
uv run --with ruff ruff check src tests
uv run --with ruff ruff format --check src tests
```

`chatbot.py` is a minimal interactive entry point, equivalent to
`basicrag chat`.
