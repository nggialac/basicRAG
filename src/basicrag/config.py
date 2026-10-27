"""Central configuration for every stage of the RAG system.

Priority (highest first):

1. Explicit keyword overrides passed to a factory, e.g. ``get_llm(model=...)``.
2. ``BASICRAG_*`` environment variables (nested delimiter ``__``).
3. Legacy flat ``BASICRAG_*`` variables from the original project.
4. ``.env`` values.
5. A YAML profile from ``configs/<profile>.yaml`` or ``BASICRAG_CONFIG``.
6. The defaults defined below.

Examples::

    BASICRAG_LLM__MODEL=qwen3.8-fn
    BASICRAG_SEARCH__STRATEGY=hybrid
    BASICRAG_RERANK__PROVIDER=cross_encoder

    # original variables still work
    BASICRAG_LLM_MODEL=qwen3.8-fn
    BASICRAG_RETRIEVER_K=5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, citation-focused assistant for a private knowledge base.\n"
    "Rules:\n"
    "1. Answer ONLY from the supplied context.\n"
    "2. If the context does not contain the answer, reply exactly: "
    '"I don\'t know based on the provided documents."\n'
    "3. Never invent facts, sources, page numbers, or URLs.\n"
    "4. Prefer concise, well-structured answers.\n"
    "5. Cite sources as [1], [2], ... matching the context labels."
)

DEFAULT_ANSWER_TEMPLATE = (
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."
)


class LLMConfig(BaseModel):
    """Chat model configuration."""

    provider: str = "openai_compat"
    model: str = "qwen3.8-fn"
    base_url: str | None = "http://127.0.0.1:11434"
    api_key: SecretStr = SecretStr("not-needed")
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 120.0
    max_retries: int = 2
    extra: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: str = "ollama"
    model: str = "embeddinggemma"
    base_url: str | None = None
    api_key: SecretStr | None = None
    dimensions: int | None = None
    batch_size: int = 32
    cache: bool = True
    cache_directory: str = "./data/cache/embeddings"
    normalize: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class LoaderConfig(BaseModel):
    """Document ingestion sources."""

    paths: list[str] = Field(default_factory=lambda: ["./papers"])
    glob: str = "**/*.*"
    extensions: list[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".txt",
            ".md",
            ".markdown",
            ".docx",
            ".csv",
            ".html",
            ".htm",
        ]
    )
    encoding: str = "utf-8"
    recursive: bool = True
    include_hidden: bool = False
    continue_on_error: bool = True
    min_chars: int = 20


class ChunkingConfig(BaseModel):
    """Chunking strategy and parent/child settings."""

    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: list[str] | None = None
    keep_separator: bool = True
    add_start_index: bool = True
    min_chunk_size: int = 50
    token_encoding: str = "cl100k_base"
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 85.0
    semantic_buffer_size: int = 1
    markdown_headers: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
    )
    parent_child: bool = False
    parent_chunk_size: int = 2400
    parent_chunk_overlap: int = 200


class VectorStoreConfig(BaseModel):
    """Vector store backend and persistence."""

    provider: str = "faiss"
    collection: str = "default"
    persist_directory: str = "./data/vector_store"
    persist: bool = True
    distance_strategy: str = "cosine"
    normalize_l2: bool = False
    # Optional backend connection settings (Chroma / Qdrant).
    url: str | None = None
    host: str | None = None
    port: int | None = None
    api_key: SecretStr | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SearchConfig(BaseModel):
    """Retrieval strategy, candidate pool, and fusion settings."""

    strategy: str = "hybrid"  # dense | sparse | hybrid | multi_query
    dense_search_type: str = "similarity"  # similarity | mmr | similarity_score_threshold
    k: int = 8
    fetch_k: int = 40
    score_threshold: float | None = None
    filter: dict[str, Any] | None = None
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60
    mmr_lambda: float = 0.5
    multi_query_count: int = 3
    multi_query_mode: str = "variants"  # variants | subquestions
    multi_query_base: str = "dense"  # dense | hybrid
    parent_document: bool = True


class RerankConfig(BaseModel):
    """Optional second-stage ranking."""

    enabled: bool = True
    provider: str = "llm"  # none | llm | cross_encoder
    top_n: int = 0  # 0 means "use search.k"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_candidates: int = 30
    max_chars: int = 1200
    device: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class QueryConfig(BaseModel):
    """Query understanding / expansion."""

    rewrite: bool = False
    rewrite_max_chars: int = 512
    hyde: bool = False
    hyde_max_chars: int = 800


class PromptConfig(BaseModel):
    """Answer prompt and context rendering."""

    system: str = DEFAULT_SYSTEM_PROMPT
    answer_template: str = DEFAULT_ANSWER_TEMPLATE
    source_template: str = "[{index}] {label}\n{content}"
    max_context_chars: int = 12000
    include_sources: bool = True


class PipelineConfig(BaseModel):
    """Runtime behavior of the assembled pipeline."""

    verbose: bool = True
    log_level: str = "INFO"
    return_sources: bool = True


# ---------------------------------------------------------------------------
# Settings sources
# ---------------------------------------------------------------------------

_override_config: str | None = None
_override_profile: str | None = None


def _config_candidates(profile: str) -> list[Path]:
    candidates: list[Path] = []
    explicit = _override_config or os.getenv("BASICRAG_CONFIG")
    if explicit:
        return [Path(explicit)]
    names = [f"{profile}.yaml", f"{profile}.yml"]
    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for root in roots:
        for name in names:
            candidate = root / "configs" / name
            if candidate.exists():
                candidates.append(candidate)
    return candidates


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Load ``configs/<profile>.yaml`` as a low-priority source."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        profile = _override_profile or os.getenv("BASICRAG_PROFILE", "default")
        candidates = _config_candidates(profile)
        self.path: Path | None = candidates[0] if candidates else None

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if self.path is None:
            return {}
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {self.path} must contain a mapping")
        return data


LEGACY_ENV: dict[str, tuple[tuple[str, ...], bool]] = {
    "BASICRAG_LLM_BASE_URL": (("llm", "base_url"), False),
    "BASICRAG_LLM_MODEL": (("llm", "model"), False),
    "BASICRAG_LLM_API_KEY": (("llm", "api_key"), False),
    "BASICRAG_LLM_TEMPERATURE": (("llm", "temperature"), False),
    "BASICRAG_EMBEDDING_MODEL": (("embeddings", "model"), False),
    "BASICRAG_PAPERS_DIR": (("loaders", "paths"), True),
    "BASICRAG_PAPERS_GLOB": (("loaders", "glob"), False),
    "BASICRAG_BREAKPOINT_TYPE": (("chunking", "breakpoint_threshold_type"), False),
    "BASICRAG_BREAKPOINT_AMOUNT": (("chunking", "breakpoint_threshold_amount"), False),
    "BASICRAG_RETRIEVER_SEARCH_TYPE": (("search", "dense_search_type"), False),
    "BASICRAG_RETRIEVER_K": (("search", "k"), False),
    "BASICRAG_RETRIEVER_SCORE_THRESHOLD": (("search", "score_threshold"), False),
}


class LegacyEnvSettingsSource(PydanticBaseSettingsSource):
    """Map the original flat environment variables to nested settings."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for env_name, (path, as_list) in LEGACY_ENV.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            value: Any = [raw] if as_list else raw
            cursor = data
            for key in path[:-1]:
                cursor = cursor.setdefault(key, {})
            cursor[path[-1]] = value
        return data


class Settings(BaseSettings):
    """Validated runtime configuration for the whole system."""

    model_config = SettingsConfigDict(
        env_prefix="BASICRAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    profile: str = "default"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    loaders: LoaderConfig = Field(default_factory=LoaderConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            LegacyEnvSettingsSource(settings_cls),
            YamlSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_settings(
    config_file: str | Path | None = None,
    profile: str | None = None,
    **overrides: Any,
) -> Settings:
    """Load settings, optionally with an explicit YAML file or profile."""
    global _override_config, _override_profile
    previous = (_override_config, _override_profile)
    if config_file is not None:
        _override_config = str(config_file)
    if profile is not None:
        _override_profile = profile
    try:
        return Settings(**overrides)
    finally:
        _override_config, _override_profile = previous


settings = load_settings()

__all__ = [
    "Settings",
    "settings",
    "load_settings",
    "LLMConfig",
    "EmbeddingConfig",
    "LoaderConfig",
    "ChunkingConfig",
    "VectorStoreConfig",
    "SearchConfig",
    "RerankConfig",
    "QueryConfig",
    "PromptConfig",
    "PipelineConfig",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_ANSWER_TEMPLATE",
]
