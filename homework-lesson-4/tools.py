from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from config import BASE_DIR, Settings


settings = Settings()


def _clip_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _safe_report_path(filename: str) -> Path:
    candidate = Path(filename)
    safe_name = candidate.name or "report.md"
    if not safe_name.lower().endswith(".md"):
        safe_name = f"{safe_name}.md"

    output_dir = BASE_DIR / settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / safe_name


def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """Search the web with DuckDuckGo and return compact results."""
    from ddgs import DDGS

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


def read_url(url: str) -> str:
    """Fetch the main text from a URL and return a trimmed excerpt."""
    import trafilatura

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


def write_report(filename: str, content: str) -> str:
    """Save a Markdown report into the configured output directory."""
    try:
        target_path = _safe_report_path(filename)
        target_path.write_text(content, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive path
        return f"Failed to write report: {exc}"

    return f"Report saved to {target_path}"


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current or factual information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the information to find.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Read the main text content of a specific URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "HTTP or HTTPS URL to inspect.",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": "Save the final Markdown report into the output directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Markdown file name, for example research_report.md.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete Markdown report content.",
                    },
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}


def format_tool_result(result: Any, limit: int) -> str:
    if isinstance(result, str):
        return _clip_text(result, limit)

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    return _clip_text(rendered, limit)
