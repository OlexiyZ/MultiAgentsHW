from pathlib import Path

import yaml
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


def load_system_prompt() -> str:
    path = BASE_DIR / "system_prompt.yaml"
    if not path.is_file():
        return _DEFAULT_SYSTEM_PROMPT
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    text = data.get("system_prompt")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return _DEFAULT_SYSTEM_PROMPT


_DEFAULT_SYSTEM_PROMPT = """
You are a research agent that answers questions by planning a short investigation,
using tools, and producing a structured Markdown response.
""".strip()


class Settings(BaseSettings):
    api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "API_KEY", "api_key"),
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("MODEL_NAME", "model_name"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )
    data_dir: str = Field(
        default="data",
        validation_alias=AliasChoices("DATA_DIR", "data_dir"),
    )
    index_dir: str = Field(
        default="index",
        validation_alias=AliasChoices("INDEX_DIR", "index_dir"),
    )
    chroma_collection: str = Field(
        default="lesson5_kb",
        validation_alias=AliasChoices("CHROMA_COLLECTION", "chroma_collection"),
    )
    knowledge_flavour: str = Field(
        default="langchain",
        validation_alias=AliasChoices("KNOWLEDGE_FLAVOUR", "knowledge_flavour"),
    )
    chunk_size: int = Field(
        default=1024,
        validation_alias=AliasChoices("CHUNK_SIZE", "chunk_size"),
    )
    chunk_overlap: int = Field(
        default=200,
        validation_alias=AliasChoices("CHUNK_OVERLAP", "chunk_overlap"),
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        validation_alias=AliasChoices("RERANKER_MODEL", "reranker_model"),
    )
    retrieval_vector_k: int = Field(
        default=10,
        validation_alias=AliasChoices("RETRIEVAL_VECTOR_K", "retrieval_vector_k"),
    )
    retrieval_bm25_k: int = Field(
        default=10,
        validation_alias=AliasChoices("RETRIEVAL_BM25_K", "retrieval_bm25_k"),
    )
    retrieval_fusion_top_n: int = Field(
        default=20,
        validation_alias=AliasChoices("RETRIEVAL_FUSION_TOP_N", "retrieval_fusion_top_n"),
    )
    rerank_top_n: int = Field(
        default=5,
        validation_alias=AliasChoices("RERANK_TOP_N", "rerank_top_n"),
    )
    max_knowledge_chars: int = Field(
        default=8000,
        validation_alias=AliasChoices("MAX_KNOWLEDGE_CHARS", "max_knowledge_chars"),
    )
    max_search_results: int = Field(
        default=5,
        validation_alias=AliasChoices("MAX_SEARCH_RESULTS", "max_search_results"),
    )
    max_url_content_length: int = Field(
        default=5000,
        validation_alias=AliasChoices(
            "MAX_URL_CONTENT_LENGTH",
            "max_url_content_length",
        ),
    )
    output_dir: str = Field(
        default="output",
        validation_alias=AliasChoices("OUTPUT_DIR", "output_dir"),
    )
    max_iterations: int = Field(
        default=12,
        validation_alias=AliasChoices("MAX_ITERATIONS", "max_iterations"),
    )
    request_timeout: int = Field(
        default=60,
        validation_alias=AliasChoices("REQUEST_TIMEOUT", "request_timeout"),
    )

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


SYSTEM_PROMPT = load_system_prompt()
