"""LangChain tool definitions for web search, URL reading, reports, and knowledge RAG search.
Визначення інструментів LangChain: веб-пошук, читання URL, звіти та RAG-пошук по базі знань."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool

from config import BASE_DIR, Settings


settings = Settings()


def _knowledge_backend_search(query: str) -> str:
    """Dispatch the query to the LlamaIndex or LangChain hybrid retriever implementation.
    Передає запит до гібридного ретривера на базі LlamaIndex або LangChain залежно від налаштувань."""
    flavour = (settings.knowledge_flavour or "langchain").strip().lower()
    if flavour in {"llama", "llamaindex"}:
        from retriever_llama_flavour import hybrid_search_llama

        return hybrid_search_llama(query, settings)
    from retriever_langchain_flavour import hybrid_search_langchain

    return hybrid_search_langchain(query, settings)


def _clip_text(text: str, limit: int) -> str:
    """Normalize whitespace and truncate text to a maximum character length with an ellipsis.
    Нормалізує пробіли й обрізає текст до максимальної довжини з додаванням багатокрапки."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _safe_report_path(filename: str) -> Path:
    """Resolve a safe Markdown filename inside the configured output directory.
    Формує безпечну назву Markdown-файлу всередині налаштованого каталогу output."""
    candidate = Path(filename)
    safe_name = candidate.name or "report.md"
    if not safe_name.lower().endswith(".md"):
        safe_name = f"{safe_name}.md"

    output_dir = BASE_DIR / settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / safe_name

# web_search_tool = {
#     "type": "function",
#     "name": "web_search",
#     "description": (
#         "Searches the web using DuckDuckGo and returns a list of results. "
#         "Use this when you need current or factual information not in your training data."
#     ),
#     "input_schema": {
#         "type": "object",
#         "properties": {
#             "query": {
#                 "type": "string",
#                 "description": "The search query string.",
#             },
#             "max_results": {
#                 "type": "integer",
#                 "description": "Maximum number of results to return (1–20).",
#                 "default": 5,
#             },
#         },
#         "required": ["query"],
#     },
# }

@tool
def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """Search the web with DuckDuckGo and return compact title, URL, and snippet dicts.
    Шукає в Інтернеті через DuckDuckGo й повертає стислі словники з заголовком, URL і сніпетом."""
    search_limit = max_results or settings.max_search_results
    search_limit = max(1, min(search_limit, 10))

    try:
        raw_results = DDGS().text(query, max_results=search_limit)
    except Exception as exc:  # pragma: no cover - defensive path
        return [{"error": f"Web search failed: {exc}"}]

    results = []
    for item in raw_results:
        title = item.get("title") or "Untitled result"
        url = item.get("href") or ""
        snippet = item.get("body") or ""
        results.append(
            {
                "title": _clip_text(title, 200),
                "url": url,
                "snippet": _clip_text(snippet, 500),
            }
        )

    return results


@tool
def read_url(url: str) -> str:
    """Fetch a page over HTTP(S), extract readable text with trafilatura, and trim it.
    Завантажує сторінку за HTTP(S), витягує читабельний текст через trafilatura й обрізає його."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"Invalid URL: {url}"

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Unable to download content from {url}"

        extracted = trafilatura.extract(
            downloaded,
            include_links=True,
            include_formatting=False,
        )
        if not extracted:
            return f"Unable to extract readable text from {url}"
    except Exception as exc:  # pragma: no cover - defensive path
        return f"Failed to read {url}: {exc}"

    return _clip_text(extracted, settings.max_url_content_length)


@tool
def write_report(filename: str, content: str) -> str:
    """Write Markdown content to a file under the configured output directory and confirm the path.
    Записує вміст Markdown у файл у налаштованому каталозі output і повертає підтвердження зі шляхом."""
    try:
        target_path = _safe_report_path(filename)
        target_path.write_text(content, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive path
        return f"Failed to write report: {exc}"

    return f"Report saved to {target_path}"


@tool
def knowledge_search(query: str) -> str:
    """Query the local hybrid (vector + BM25 + rerank) knowledge base built from ingested PDFs.
    Запитує локальну гібридну базу (вектор + BM25 + rerank) з проіндексованих PDF і обрізає відповідь."""
    result = _knowledge_backend_search(query)
    limit = settings.max_knowledge_chars
    if len(result) <= limit:
        return result
    return result[: max(limit - 3, 0)].rstrip() + "..."


TOOLS = [web_search, read_url, write_report, knowledge_search]
# TOOLS = [write_report, knowledge_search]
