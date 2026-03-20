from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "API_KEY", "api_key"),
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("MODEL_NAME", "model_name"),
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
        default=8,
        validation_alias=AliasChoices("MAX_ITERATIONS", "max_iterations"),
    )
    request_timeout: int = Field(
        default=20,
        validation_alias=AliasChoices("REQUEST_TIMEOUT", "request_timeout"),
    )

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


SYSTEM_PROMPT = """
You are a research agent that answers questions by planning a short investigation,
using tools, and producing a structured Markdown response.

Operating rules:
- Prefer multi-step research: usually perform several tool calls before finalizing.
- Use web_search to discover relevant sources.
- Use read_url to inspect the most relevant pages in more detail.
- Use write_report to save the response to a file named research_report.md.
- Be explicit about uncertainty and mention when a source could not be fetched.
- Keep the final answer concise, but include the key findings, trade-offs, and links.
- Do not invent sources or quotes.
- Respond in the language of the request. If the request is in Russian, respond in Ukrainian.
""".strip()
