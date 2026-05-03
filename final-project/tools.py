from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import tool

from agent_metrics import record_supervisor_tool
from config import BASE_DIR, Settings, preview_for_log
from retriever import hybrid_search


logger = logging.getLogger(__name__)

settings = Settings()

TRANSLIT_MAP = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "h",
        "ґ": "g",
        "д": "d",
        "е": "e",
        "є": "ie",
        "ж": "zh",
        "з": "z",
        "и": "y",
        "і": "i",
        "ї": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ь": "",
        "ю": "iu",
        "я": "ia",
    }
)


def _clip_text(text: str, limit: int) -> str:
    """Normalizes whitespace and truncates text to a maximum length."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _extract_url_text(url: str) -> str:
    """Fetches an HTTP(S) page and extracts readable text."""

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
    except Exception as exc:
        logger.exception("Tool URL extraction: exception")
        return f"Failed to read {url}: {exc}"

    return extracted


def _slugify_topic(topic: str) -> str:
    """Builds a stable filesystem-safe slug from a report topic."""

    normalized = unicodedata.normalize("NFKC", topic or "").casefold()
    transliterated = normalized.translate(TRANSLIT_MAP)
    slug = re.sub(r"[^a-z0-9]+", "-", transliterated)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:120].strip("-") or "report"


def _versioned_report_path(output_dir: Path, slug: str) -> Path:
    path = output_dir / f"{slug}.md"
    if not path.exists():
        return path

    version = 2
    while True:
        candidate = output_dir / f"{slug}.v{version}.md"
        if not candidate.exists():
            return candidate
        version += 1


def _safe_report_path(topic: str) -> Path:
    """Resolves a versioned Markdown path for a stable report topic."""

    output_dir = BASE_DIR / settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return _versioned_report_path(output_dir, _slugify_topic(topic))


@tool
def web_search(query: str) -> list[dict]:
    """Searches the web via DuckDuckGo and returns compact result dicts."""

    search_limit = max(1, min(settings.max_search_results, 10))
    logger.info(
        "Tool web_search: query=%r max_results=%s",
        preview_for_log(query, 300),
        search_limit,
    )
    try:
        raw_results = DDGS().text(query, max_results=search_limit)
    except DDGSException as exc:
        if str(exc).strip() == "No results found.":
            logger.info("Tool web_search: no results")
            return []
        logger.exception("Tool web_search: failed")
        return [{"error": f"Web search failed: {exc}"}]
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
    """Fetches an HTTP(S) page, extracts readable text, and trims it."""

    logger.info("Tool read_url: url=%r", url[:500] + ("..." if len(url) > 500 else ""))
    extracted = _extract_url_text(url)
    out = _clip_text(extracted, settings.max_url_content_length)
    logger.info("Tool read_url: extracted_chars=%d", len(out))
    return out


@tool
def read_full_normative_text(url: str) -> str:
    """Reads an official normative document URL and returns a large text extract."""

    logger.info(
        "Tool read_full_normative_text: url=%r",
        url[:500] + ("..." if len(url) > 500 else ""),
    )
    extracted = _extract_url_text(url)
    limit = max(settings.max_normative_doc_chars, settings.max_url_content_length)
    out = _clip_text(extracted, limit)
    if len(extracted) > len(out):
        logger.warning(
            "Tool read_full_normative_text: truncated full text chars=%d limit=%d",
            len(extracted),
            limit,
        )
        return (
            out
            + "\n\n[TRUNCATED: extracted normative text exceeded "
            + f"MAX_NORMATIVE_DOC_CHARS={limit}. Increase this setting if the full "
            + "document text is required.]"
        )
    logger.info("Tool read_full_normative_text: extracted_chars=%d", len(out))
    return out


@tool
def knowledge_search(query: str) -> str:
    """Queries the local hybrid Chroma index and returns trimmed Markdown."""

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
def save_report(topic: str, content: str) -> str:
    """Writes Markdown report content to a versioned file for the given topic."""

    logger.info(
        "Tool save_report: topic=%r content_chars=%d preview=%s",
        topic,
        len(content),
        preview_for_log(content, 240),
    )
    record_supervisor_tool("save_report")
    try:
        target = _safe_report_path(topic)
        target.write_text(content, encoding="utf-8")
        logger.info("Tool save_report: wrote path=%s", target)
        return f"Report saved to {target}"
    except Exception as exc:
        logger.exception("Tool save_report: failed")
        return f"Failed to save report: {exc}"


PLANNER_TOOLS = [web_search, knowledge_search]
RESEARCH_TOOLS = [web_search, read_url, read_full_normative_text, knowledge_search]
SUPERVISOR_TOOLS = [save_report]
