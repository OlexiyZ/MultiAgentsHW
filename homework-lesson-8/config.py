from __future__ import annotations

import logging
import sys
from pathlib import Path
from textwrap import dedent

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
        default="lesson8_kb",
        validation_alias=AliasChoices("CHROMA_COLLECTION"),
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
        default="logs/lesson8.log",
        validation_alias=AliasChoices("LOG_FILE", "log_file"),
        description="Log file path; relative paths are under the lesson-8 project directory.",
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
        raw = Path(settings.log_file.strip() or "logs/lesson8.log")
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


PLANNER_SYSTEM_PROMPT = dedent(
    """
    You are Planner Agent in a multi-agent research workflow.
    Goal:
    - Understand the user's request.
    - Run short exploratory lookups with tools when useful.
    - Return only structured ResearchPlan.

    Rules:
    - Produce specific, searchable queries.
    - Include both 'knowledge_base' and 'web' when freshness matters.
    - Make output_format concrete (sections/table/checklist).
    - Always populate every ResearchPlan field: goal, search_queries, sources_to_check, output_format
      (never omit sources_to_check or output_format).
    """
).strip()


RESEARCH_SYSTEM_PROMPT = dedent(
    """
    You are Research Agent.
    Execute the research request using tools:
    - knowledge_search for local indexed PDFs
    - web_search for fresh facts
    - read_url for deeper verification

    Produce a clear markdown findings document with:
    1) key findings
    2) evidence and sources
    3) explicit uncertainties / assumptions.
    """
).strip()


CRITIC_SYSTEM_PROMPT = dedent(
    """
    You are Critic Agent in evaluator-optimizer loop.
    Independently verify findings using tools before verdict.

    Evaluate:
    - Freshness: Are sources/data current relative to today's date?
    - Completeness: Are all parts of user request covered?
    - Structure: Are findings logically organized and report-ready?

    Return only structured CritiqueResult.
    If anything material is missing/outdated, return verdict=REVISE with concrete revision_requests.
    """
).strip()


SUPERVISOR_SYSTEM_PROMPT = dedent(
    """
    You are Supervisor orchestrating Plan -> Research -> Critique.

    Mandatory flow:
    1) Call plan(request) first.
    2) Call research(request) using the plan.
    3) Call critique(findings) to evaluate result.
    4) If verdict is REVISE:
       - call research again only with a narrow request that targets the critic's concrete
         revision_requests / gaps (do not redo full exploratory research)
       - run critique again
       - do at most 2 revision rounds.
    5) If verdict is APPROVE:
       - compile final markdown report
       - call save_report(filename, content).

    Important:
    - Every tool call must include decision_reason: a short explanation of why that
      tool is the correct next step.
    - After plan() has been called once for the current user goal, do not call plan() again.
      Reuse that plan for any revise rounds unless the user explicitly asks to replan or
      changes the task scope.
    - Do not run extra research passes "just in case". Additional research is allowed only
      when critique returns REVISE and specifies what is missing; address only those gaps.
    - Never skip critique.
    - Preserve sources and dates.
    - Save only when report is complete.
    - Respond in the language of the request. If the request is in Russian, respond in Ukrainian.
    """
).strip()
