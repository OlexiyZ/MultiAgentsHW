"""SearchMCP: web_search, read_url, knowledge_search + knowledge-base-stats resource."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BASE_DIR, Settings, preview_for_log  # noqa: E402
from kb_common import index_dir  # noqa: E402
from retriever import hybrid_search  # noqa: E402

logger = logging.getLogger(__name__)
settings = Settings()
mcp = FastMCP("SearchMCP")


def _clip_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


@mcp.tool()
def web_search(query: str) -> list[dict]:
    """Search the web via DuckDuckGo; returns title, url, snippet dicts."""

    search_limit = max(1, min(settings.max_search_results, 10))
    logger.info(
        "MCP web_search: query=%r max_results=%s",
        preview_for_log(query, 300),
        search_limit,
    )
    try:
        raw_results = DDGS().text(query, max_results=search_limit)
    except DDGSException as exc:
        if str(exc).strip() == "No results found.":
            logger.info("MCP web_search: no results")
            return []
        logger.exception("MCP web_search: failed")
        return [{"error": f"Web search failed: {exc}"}]
    except Exception as exc:
        logger.exception("MCP web_search: failed")
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
    logger.info("MCP web_search: results=%d", len(results))
    return results


@mcp.tool()
def read_url(url: str) -> str:
    """Fetch URL and extract readable text with trafilatura."""

    logger.info("MCP read_url: url=%r", url[:500] + ("..." if len(url) > 500 else ""))
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
        logger.exception("MCP read_url: exception")
        return f"Failed to read {url}: {exc}"
    out = _clip_text(extracted, settings.max_url_content_length)
    logger.info("MCP read_url: extracted_chars=%d", len(out))
    return out


@mcp.tool()
def knowledge_search(query: str) -> str:
    """Query local hybrid Chroma index."""

    logger.info("MCP knowledge_search: query=%r", preview_for_log(query, 400))
    result = hybrid_search(query, settings=settings)
    logger.info("MCP knowledge_search: result_chars=%d", len(result))
    return result


@mcp.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> str:
    """Document chunk count and index path for the local knowledge base."""

    idx = index_dir(settings)
    if not idx.is_dir():
        payload = {
            "document_chunks": 0,
            "index_path": str(idx),
            "last_updated": None,
            "note": "Run python ingest.py after adding PDFs to data/",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(idx))
        coll = client.get_or_create_collection(settings.chroma_collection)
        n = coll.count()
        mtime = datetime.fromtimestamp(idx.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception as exc:
        return json.dumps(
            {"error": str(exc), "index_path": str(idx)},
            ensure_ascii=False,
            indent=2,
        )

    payload = {
        "document_chunks": n,
        "collection": settings.chroma_collection,
        "index_path": str(idx),
        "last_index_mtime_utc": mtime,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    port = settings.search_mcp_port
    mcp.run(transport="http", host="0.0.0.0", port=port)
