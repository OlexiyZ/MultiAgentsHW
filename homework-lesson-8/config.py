from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from pydantic import AliasChoices, Field, SecretStr
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
        default=10, validation_alias=AliasChoices("RETRIEVAL_VECTOR_K")
    )
    retrieval_bm25_k: int = Field(
        default=10, validation_alias=AliasChoices("RETRIEVAL_BM25_K")
    )
    retrieval_fusion_top_n: int = Field(
        default=20,
        validation_alias=AliasChoices("RETRIEVAL_FUSION_TOP_N"),
    )
    rerank_top_n: int = Field(default=5, validation_alias=AliasChoices("RERANK_TOP_N"))
    max_knowledge_chars: int = Field(
        default=8000,
        validation_alias=AliasChoices("MAX_KNOWLEDGE_CHARS"),
    )
    max_search_results: int = Field(
        default=5, validation_alias=AliasChoices("MAX_SEARCH_RESULTS")
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

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


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
       - call research again with Critic feedback
       - run critique again
       - do at most 2 revision rounds.
    5) If verdict is APPROVE:
       - compile final markdown report
       - call save_report(filename, content).

    Important:
    - Never skip critique.
    - Preserve sources and dates.
    - Save only when report is complete.
    - Must use save_report to save the response to a file such as research_report.md
    """
).strip()
