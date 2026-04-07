from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool

from agent_metrics import record_supervisor_tool
from config import BASE_DIR, Settings, preview_for_log
from retriever import hybrid_search


logger = logging.getLogger(__name__)

settings = Settings()


def _clip_text(text: str, limit: int) -> str:
    """Normalizes whitespace and truncates text to a maximum length with an ellipsis.
    Нормалізує пробіли й обрізає текст до максимальної довжини з багатокрапкою."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _safe_report_path(filename: str) -> Path:
    """Resolves a basename-safe Markdown path inside the configured output directory.
    Формує безпечну назву .md у налаштованому каталозі output."""

    candidate = Path(filename)
    safe_name = candidate.name or "report.md"
    if not safe_name.lower().endswith(".md"):
        safe_name = f"{safe_name}.md"

    output_dir = BASE_DIR / settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / safe_name


@tool
def web_search(query: str) -> list[dict]:
    # def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """Searches the web via DuckDuckGo and returns compact title, URL, and snippet dicts.
    Шукає в мережі через DuckDuckGo й повертає стислі словники з заголовком, URL і сніпетом."""

    search_limit = settings.max_search_results
    # search_limit = max_results or settings.max_search_results
    search_limit = max(1, min(search_limit, 10))
    logger.info(
        "Tool web_search: query=%r max_results=%s",
        preview_for_log(query, 300),
        search_limit,
    )
    try:
        raw_results = DDGS().text(query, max_results=search_limit)
    except Exception as exc:
        logger.exception("Tool web_search: failed")
        return [{"error": f"Web search failed: {exc}"}]

    results = []
    for item in raw_results:
        results.append(
            {
                "title": _clip_text(item.get("title") or "Untitled result", 200),
                "url": item.get("href") or "",
                "snippet": _clip_text(item.get("body") or "", 500),
            }
        )
    logger.info("Tool web_search: results=%d", len(results))
    return results


@tool
def read_url(url: str) -> str:
    """Fetches an HTTP(S) page, extracts readable text with trafilatura, and trims it.
    Завантажує сторінку HTTP(S), витягує читабельний текст через trafilatura й обрізає."""

    logger.info("Tool read_url: url=%r", url[:500] + ("..." if len(url) > 500 else ""))
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        logger.warning("Tool read_url: invalid url")
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
    except Exception as exc:
        logger.exception("Tool read_url: exception")
        return f"Failed to read {url}: {exc}"
    out = _clip_text(extracted, settings.max_url_content_length)
    logger.info("Tool read_url: extracted_chars=%d", len(out))
    return out


@tool
def knowledge_search(query: str) -> str:
    """Queries the local hybrid Chroma index for the string and returns trimmed Markdown.
    Запитує локальний гібридний індекс Chroma за рядком і повертає обрізаний Markdown."""

    logger.info(
        "Tool knowledge_search: query=%r",
        preview_for_log(query, 400),
    )
    result = hybrid_search(query, settings=settings)
    limit = settings.max_knowledge_chars
    if len(result) <= limit:
        logger.info("Tool knowledge_search: result_chars=%d", len(result))
        return result
    trimmed = result[: max(limit - 3, 0)].rstrip() + "..."
    logger.info("Tool knowledge_search: result_chars=%d (trimmed)", len(trimmed))
    return trimmed


@tool
def save_report(filename: str, content: str) -> str:
    """Writes Markdown report content to a safe file under output/ and returns status text.
    Записує вміст звіту Markdown у безпечний файл у output/ і повертає текст статусу."""

    logger.info(
        "Tool save_report: filename=%r content_chars=%d preview=%s",
        filename,
        len(content),
        preview_for_log(content, 240),
    )
    record_supervisor_tool("save_report")
    try:
        target = _safe_report_path(filename)
        target.write_text(content, encoding="utf-8")
        logger.info("Tool save_report: wrote path=%s", target)
        return f"Report saved to {target}"
    except Exception as exc:
        logger.exception("Tool save_report: failed")
        return f"Failed to save report: {exc}"


PLANNER_TOOLS = [web_search, knowledge_search]
RESEARCH_TOOLS = [web_search, read_url, knowledge_search]
SUPERVISOR_TOOLS = [save_report]
