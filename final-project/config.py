from __future__ import annotations

import logging
import sys
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "API_KEY", "api_key")
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("MODEL_NAME", "model_name"),
    )
    supervisor_model_name: str = Field(
        default="gpt-5.4-mini",
        validation_alias=AliasChoices(
            "SUPERVISOR_MODEL_NAME",
            "MODEL_NAME",
            "supervisor_model_name",
        ),
    )
    planner_model_name: str = Field(
        default="gpt-5.4-mini",
        validation_alias=AliasChoices(
            "PLANNER_MODEL_NAME",
            "MODEL_NAME",
            "planner_model_name",
        ),
    )
    research_model_name: str = Field(
        default="gpt-5.4-mini",
        validation_alias=AliasChoices(
            "RESEARCH_MODEL_NAME",
            "MODEL_NAME",
            "research_model_name",
        ),
    )
    critic_model_name: str = Field(
        default="gpt-5.4",
        validation_alias=AliasChoices(
            "CRITIC_MODEL_NAME",
            "MODEL_NAME",
            "critic_model_name",
        ),
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )
    data_dir: str = Field(default="data", validation_alias=AliasChoices("DATA_DIR"))
    index_dir: str = Field(default="index", validation_alias=AliasChoices("INDEX_DIR"))
    chroma_collection: str = Field(
        default="final-project_kb",
        validation_alias=AliasChoices("CHROMA_COLLECTION"),
    )
    ingest_rebuild_index: bool = Field(
        default=True,
        validation_alias=AliasChoices("INGEST_REBUILD_INDEX", "ingest_rebuild_index"),
        description=(
            "If true, ingest removes the existing Chroma index before writing. "
            "If false, ingest appends to the existing index."
        ),
    )
    ingest_tag_filters: str = Field(
        default="issuer_match:nbu",
        validation_alias=AliasChoices("INGEST_TAG_FILTERS", "ingest_tag_filters"),
        description=(
            "Comma-separated metadata tags to include during ingest. "
            "Empty value disables ingest tag filtering."
        ),
    )
    chunk_size: int = Field(default=1024, validation_alias=AliasChoices("CHUNK_SIZE"))
    chunk_overlap: int = Field(
        default=200, validation_alias=AliasChoices("CHUNK_OVERLAP")
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        validation_alias=AliasChoices("RERANKER_MODEL"),
    )
    retrieval_vector_k: int = Field(
        default=5, validation_alias=AliasChoices("RETRIEVAL_VECTOR_K")
    )
    retrieval_bm25_k: int = Field(
        default=5, validation_alias=AliasChoices("RETRIEVAL_BM25_K")
    )
    retrieval_fusion_top_n: int = Field(
        default=5,
        validation_alias=AliasChoices("RETRIEVAL_FUSION_TOP_N"),
    )
    rerank_top_n: int = Field(default=5, validation_alias=AliasChoices("RERANK_TOP_N"))
    max_knowledge_chars: int = Field(
        default=8000,
        validation_alias=AliasChoices("MAX_KNOWLEDGE_CHARS"),
    )
    max_search_results: int = Field(
        default=3, validation_alias=AliasChoices("MAX_SEARCH_RESULTS")
    )
    max_url_content_length: int = Field(
        default=5000, validation_alias=AliasChoices("MAX_URL_CONTENT_LENGTH")
    )
    output_dir: str = Field(default="output", validation_alias=AliasChoices("OUTPUT_DIR"))
    max_iterations: int = Field(
        default=16, validation_alias=AliasChoices("MAX_ITERATIONS")
    )
    request_timeout: int = Field(
        default=90, validation_alias=AliasChoices("REQUEST_TIMEOUT")
    )
    langfuse_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("LANGFUSE_ENABLED", "langfuse_enabled"),
    )
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "langfuse_public_key"),
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY", "langfuse_secret_key"),
    )
    langfuse_base_url: str = Field(
        default="https://us.cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "langfuse_base_url"),
    )
    langfuse_default_tags: str = Field(
        default="mas,final-project",
        validation_alias=AliasChoices(
            "LANGFUSE_DEFAULT_TAGS",
            "langfuse_default_tags",
        ),
    )
    langfuse_prompt_label: str = Field(
        default="production",
        validation_alias=AliasChoices(
            "LANGFUSE_PROMPT_LABEL",
            "langfuse_prompt_label",
        ),
    )
    supervisor_prompt_name: str = Field(
        default="final-project/supervisor-system",
        validation_alias=AliasChoices(
            "SUPERVISOR_PROMPT_NAME",
            "supervisor_prompt_name",
        ),
    )
    planner_prompt_name: str = Field(
        default="final-project/planner-system",
        validation_alias=AliasChoices(
            "PLANNER_PROMPT_NAME",
            "planner_prompt_name",
        ),
    )
    research_prompt_name: str = Field(
        default="final-project/research-system",
        validation_alias=AliasChoices(
            "RESEARCH_PROMPT_NAME",
            "research_prompt_name",
        ),
    )
    critic_prompt_name: str = Field(
        default="final-project/critic-system",
        validation_alias=AliasChoices(
            "CRITIC_PROMPT_NAME",
            "critic_prompt_name",
        ),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
        description="Python logging level name: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    log_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("LOG_ENABLED", "log_enabled"),
        description="If false, logging is effectively disabled (no console/file output).",
    )
    log_destination: str = Field(
        default="stderr",
        validation_alias=AliasChoices(
            "LOG_DESTINATION",
            "LOG_TO",
            "log_destination",
        ),
        description="Where to emit logs: stderr (screen), file, or both.",
    )
    log_file: str = Field(
        default="logs/lesson10.log",
        validation_alias=AliasChoices("LOG_FILE", "log_file"),
        description="Log file path; relative paths are under the lesson-10 project directory.",
    )

    @field_validator("log_destination")
    @classmethod
    def _normalize_log_destination(cls, value: str) -> str:
        allowed = {"stderr", "file", "both"}
        key = (value or "").strip().lower()
        if key in allowed:
            return key
        return "stderr"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def configure_logging() -> None:
    """Configure root logging from Settings (call once at process startup, e.g. from main).
    Налаштовує кореневе логування з Settings (викликати один раз на старті, напр. з main)."""

    settings = Settings()
    root = logging.getLogger()
    root.handlers.clear()

    if not settings.log_enabled:
        root.addHandler(logging.NullHandler())
        root.setLevel(logging.CRITICAL)
        return

    name = settings.log_level.strip().upper()
    level = getattr(logging, name, logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []
    dest = settings.log_destination

    if dest in {"stderr", "both"}:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    if dest in {"file", "both"}:
        raw = Path(settings.log_file.strip() or "logs/lesson10.log")
        path = raw if raw.is_absolute() else (BASE_DIR / raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if not handlers:
        fallback = logging.StreamHandler(sys.stderr)
        fallback.setFormatter(formatter)
        handlers.append(fallback)

    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    logging.getLogger(__name__).debug(
        "Logging configured level=%s destination=%s",
        name,
        dest,
    )


def preview_for_log(text: str, limit: int = 400) -> str:
    """Single-line preview for log lines (avoid huge payloads at INFO).
    Однорядковий прев’ю-текст для логів (щоб не засмічувати INFO великими даними)."""

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(limit - 3, 0)].rstrip() + "..."
