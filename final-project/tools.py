from __future__ import annotations

import gzip
from html.parser import HTMLParser
from io import BytesIO
import logging
import re
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader
import trafilatura
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_core.tools import tool

from agent_metrics import record_supervisor_tool
from kb_common import _decode_text, _load_doc_file, _load_docx_file
from config import BASE_DIR, Settings, preview_for_log
from retriever import hybrid_search


logger = logging.getLogger(__name__)

settings = Settings()

OFFICIAL_NORMATIVE_DOMAINS = ("zakon.rada.gov.ua", "bank.gov.ua")
ATTACHMENT_EXTENSIONS = {".pdf", ".doc", ".docx"}
ATTACHMENT_URL_HINTS = (
    "/admin_uploads/",
    "/laws/file/",
    "/laws/download/",
    "/file/text/",
    "/document/",
)
ATTACHMENT_TEXT_HINTS = (
    "pdf",
    "doc",
    "docx",
    "повний текст",
    "текст",
    "додат",
    "положення",
    "постанова",
    "resolution",
    "завантаж",
    "download",
)
SECONDARY_ATTACHMENT_HINTS = (
    "allres",
    "result",
    "результат",
    "обговорення",
    "зауваж",
    "пропозиц",
    "аналіз регуляторного впливу",
)

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


class _LinkExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[tuple[str, str]] = []
        self._href_stack: list[str | None] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        self._href_stack.append(urljoin(self.base_url, href) if href else None)
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href_stack:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href_stack:
            return
        href = self._href_stack.pop()
        text = " ".join("".join(self._text_parts).split())
        self._text_parts = []
        if href:
            self.links.append((href, text))


def _download_url(url: str) -> tuple[bytes, str]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=settings.request_timeout) as response:
        content_type = response.headers.get("content-type", "")
        content_encoding = response.headers.get("content-encoding", "").casefold()
        raw = response.read()
        if content_encoding == "gzip" or raw.startswith(b"\x1f\x8b"):
            raw = gzip.decompress(raw)
        elif content_encoding == "deflate":
            import zlib

            raw = zlib.decompress(raw)
        return raw, content_type


def _normalize_normative_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.casefold().endswith("zakon.rada.gov.ua"):
        if parsed.path.startswith("/go/"):
            doc_id = parsed.path.removeprefix("/go/").strip("/")
            return f"https://zakon.rada.gov.ua/laws/show/{doc_id}#Text"
        if parsed.path.startswith("/laws/show/") and not parsed.fragment:
            return f"{url}#Text"
    return url


def _extract_html_text(raw: bytes, url: str) -> str:
    html = _decode_text(raw)
    extracted = trafilatura.extract(
        html,
        url=url,
        include_links=True,
        include_formatting=False,
    )
    return extracted or _clip_text(re.sub(r"(?s)<[^>]+>", " ", html), len(html))


def _extract_url_text(url: str) -> str:
    """Fetches an HTTP(S) page and extracts readable text."""

    url = _normalize_normative_url(url)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"Invalid URL: {url}"

    try:
        raw, _content_type = _download_url(url)
        extracted = _extract_html_text(raw, url)
    except Exception as exc:
        logger.exception("Tool URL extraction: exception")
        return f"Failed to read {url}: {exc}"

    return extracted


def _link_domain_allowed(url: str) -> bool:
    netloc = urlparse(url).netloc.casefold()
    return any(netloc == domain or netloc.endswith(f".{domain}") for domain in OFFICIAL_NORMATIVE_DOMAINS)


def _official_result_priority(url: str) -> tuple[int, str]:
    netloc = urlparse(url).netloc.casefold()
    for index, domain in enumerate(OFFICIAL_NORMATIVE_DOMAINS):
        if netloc == domain or netloc.endswith(f".{domain}"):
            return index, url
    return len(OFFICIAL_NORMATIVE_DOMAINS), url


def _attachment_score(url: str, text: str) -> int:
    lowered = unquote(f"{url} {text}").casefold()
    suffix = Path(urlparse(url).path).suffix.casefold()
    score = 0
    if suffix in ATTACHMENT_EXTENSIONS:
        score += 20
    if any(hint in lowered for hint in ATTACHMENT_URL_HINTS):
        score += 10
    score += sum(3 for hint in ATTACHMENT_TEXT_HINTS if hint in lowered)
    score -= sum(8 for hint in SECONDARY_ATTACHMENT_HINTS if hint in lowered)
    if "пояснюв" in lowered or "news" in lowered or "/news/" in lowered:
        score -= 10
    return score


def _extract_links(raw_html: bytes, base_url: str) -> list[tuple[str, str]]:
    parser = _LinkExtractor(base_url)
    parser.feed(_decode_text(raw_html))
    deduped: dict[str, str] = {}
    for href, text in parser.links:
        parsed = urlparse(href)
        if parsed.scheme in {"http", "https"} and _link_domain_allowed(href):
            deduped.setdefault(href, text)
    return list(deduped.items())


def _suffix_for_download(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.casefold()
    if suffix in {".pdf", ".doc", ".docx", ".html", ".htm", ".txt"}:
        return suffix
    lowered = content_type.casefold()
    if "pdf" in lowered:
        return ".pdf"
    if "wordprocessingml" in lowered or "docx" in lowered:
        return ".docx"
    if "msword" in lowered:
        return ".doc"
    if "html" in lowered:
        return ".html"
    return ".txt"


def _extract_downloaded_document_text(url: str) -> str:
    raw, content_type = _download_url(url)
    suffix = _suffix_for_download(url, content_type)
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(page for page in pages if page.strip())
    if suffix in {".html", ".htm"}:
        return _extract_html_text(raw, url)
    if suffix == ".txt":
        return _decode_text(raw)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        if suffix == ".docx":
            return _load_docx_file(tmp_path)
        if suffix == ".doc":
            return _load_doc_file(tmp_path)
        return _decode_text(raw)
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_normative_text_with_attachments(url: str) -> str:
    raw, content_type = _download_url(url)
    main_text = (
        _extract_downloaded_document_text(url)
        if _suffix_for_download(url, content_type) in ATTACHMENT_EXTENSIONS
        else _extract_html_text(raw, url)
    )

    links = _extract_links(raw, url) if "html" in content_type.casefold() else []
    candidates = sorted(
        (
            (score, href, text)
            for href, text in links
            if (score := _attachment_score(href, text)) > 0
        ),
        reverse=True,
    )

    sections = [f"[SOURCE PAGE]\n{main_text.strip()}"]
    for _score, href, text in candidates[:8]:
        try:
            attachment_text = _extract_downloaded_document_text(href).strip()
        except Exception as exc:
            logger.warning("Could not read normative attachment %s: %s", href, exc)
            continue
        if len(attachment_text) < 200:
            continue
        sections.append(
            "\n".join(
                [
                    f"[ATTACHMENT: {text or href}]",
                    f"URL: {href}",
                    attachment_text,
                ]
            )
        )

    return "\n\n".join(section for section in sections if section.strip())


def _split_keywords(raw: str) -> list[str]:
    keywords: list[str] = []
    for item in re.split(r"[,;\n]+", raw or ""):
        keyword = " ".join(item.split())
        if len(keyword) >= 2 and keyword.casefold() not in {k.casefold() for k in keywords}:
            keywords.append(keyword)
    return keywords


def _normative_keywords(search_terms: str = "") -> list[str]:
    return _split_keywords(search_terms)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _extract_relevant_normative_fragments(text: str, search_terms: str = "") -> str | None:
    keywords = _normative_keywords(search_terms)
    if not keywords:
        return None

    haystack = text.casefold()
    hits: list[tuple[int, str]] = []
    for keyword in keywords:
        start = 0
        needle = keyword.casefold()
        while True:
            index = haystack.find(needle, start)
            if index == -1:
                break
            hits.append((index, keyword))
            start = index + max(len(needle), 1)

    if not hits:
        return None

    window = max(settings.normative_excerpt_window, 200)
    max_fragments = max(settings.normative_excerpt_max_fragments, 1)
    selected_hits: list[tuple[int, str]] = []
    for keyword in keywords:
        keyword_hits = [hit for hit in hits if hit[1].casefold() == keyword.casefold()]
        if keyword_hits:
            selected_hits.append(min(keyword_hits, key=lambda hit: hit[0]))
    seen = {(index, keyword.casefold()) for index, keyword in selected_hits}
    for hit in sorted(hits, key=lambda item: item[0]):
        key = (hit[0], hit[1].casefold())
        if key not in seen:
            selected_hits.append(hit)
            seen.add(key)
        if len(selected_hits) >= max_fragments * 4:
            break
    ranges = sorted(
        (max(index - window, 0), min(index + window, len(text)))
        for index, _keyword in selected_hits
    )
    ranges = _merge_ranges(ranges)[:max_fragments]

    fragments: list[str] = []
    for number, (start, end) in enumerate(ranges, start=1):
        fragment = text[start:end].strip()
        matched = sorted(
            {
                keyword
                for index, keyword in hits
                if start <= index <= end
            },
            key=str.casefold,
        )
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        fragments.append(
            "\n".join(
                [
                    f"[RELEVANT FRAGMENT {number}]",
                    f"Matched keywords: {', '.join(matched)}",
                    f"{prefix}{fragment}{suffix}",
                ]
            )
        )

    return "\n\n".join(
        [
            "[FOCUSED NORMATIVE EXCERPTS]",
            f"Document characters scanned: {len(text)}",
            f"Keywords used: {', '.join(keywords)}",
            *fragments,
        ]
    )


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
    results.sort(key=lambda item: _official_result_priority(item.get("url") or ""))
    logger.info("Tool web_search: results=%d", len(results))
    return results


@tool
def read_url(url: str) -> str:
    """Fetches an HTTP(S) page, extracts readable text, and trims it."""

    url = _normalize_normative_url(url)
    logger.info("Tool read_url: url=%r", url[:500] + ("..." if len(url) > 500 else ""))
    extracted = _extract_url_text(url)
    out = _clip_text(extracted, settings.max_url_content_length)
    logger.info("Tool read_url: extracted_chars=%d", len(out))
    return out


@tool
def read_full_normative_text(url: str, search_terms: str = "") -> str:
    """Reads an official normative document URL and returns focused relevant extracts."""

    url = _normalize_normative_url(url)
    logger.info(
        "Tool read_full_normative_text: url=%r",
        url[:500] + ("..." if len(url) > 500 else ""),
    )
    parsed = urlparse(url)
    if not _link_domain_allowed(url):
        logger.warning("Tool read_full_normative_text: non-official domain")
        extracted = _extract_url_text(url)
    elif parsed.scheme not in {"http", "https"} or not parsed.netloc:
        extracted = f"Invalid URL: {url}"
    else:
        try:
            extracted = _extract_normative_text_with_attachments(url)
        except Exception as exc:
            logger.exception("Tool read_full_normative_text: attachment-aware read failed")
            extracted = f"Failed to read full normative text from {url}: {exc}"
    focused = _extract_relevant_normative_fragments(extracted, search_terms)
    if focused:
        extracted = focused
    limit = max(settings.max_normative_doc_chars, settings.max_url_content_length)
    normalized = " ".join(extracted.split())
    truncated = len(normalized) > limit
    out = _clip_text(extracted, limit)
    if truncated:
        logger.warning(
            "Tool read_full_normative_text: truncated full text chars=%d limit=%d",
            len(normalized),
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
