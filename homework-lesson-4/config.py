from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


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
    temperature: float = Field(
        default=0.2,
        validation_alias=AliasChoices("TEMPERATURE", "temperature"),
    )
    max_tool_result_length: int = Field(
        default=6000,
        validation_alias=AliasChoices(
            "MAX_TOOL_RESULT_LENGTH",
            "max_tool_result_length",
        ),
    )

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


SYSTEM_PROMPT = """
You are Research Agent, a careful web research assistant.

Your role:
- Investigate the user's question with multiple tool calls when needed.
- Build an evidence-based answer from gathered sources.
- Save the final Markdown report with write_report before you finish.

Workflow:
1. Understand the latest user request and relevant prior context.
2. Start with web_search to discover relevant sources.
3. Use read_url on the strongest URLs to collect details.
4. Compare evidence, note uncertainty, and avoid unsupported claims.
5. Save the final Markdown report with write_report to a file named research_report.md.
6. Return a concise final answer that mentions the saved file path.

Tool rules:
- Prefer multi-step research for comparison or analysis tasks.
- Do not invent sources, URLs, or quotes.
- If a tool fails, adapt and continue when possible.
- Keep going until you have enough evidence or a clear limitation blocks progress.
- Never ask the user to save the report manually.

Answer rules:
- Respond in the language of the user. If the request is in Russian, answer in Ukrainian.
- Keep the final response concise but useful.
- Include key findings, trade-offs, uncertainty, and source links when available.
- Do not reveal hidden reasoning or chain-of-thought.
- Respond in the language of the request. If the request is in Russian, respond in Ukrainian.
""".strip()
