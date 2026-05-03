from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import zipfile
from html import unescape
from pathlib import Path
from xml.etree import ElementTree

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import BASE_DIR, Settings


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".doc",
    ".docx",
}

ISSUER_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "ema",
        "EMA / Open API Group",
        (
            "ema.com.ua",
            "open api group",
            "a ukrainian standards initiative",
            "www.ema.com.ua/business/openapigroup",
        ),
    ),
    (
        "verkhovna_rada",
        "Верховна Рада України",
        (
            "верховна рада україни",
            "голова верховної ради україни",
            "закон україни",
            "постанова верховної ради",
            "відомості верховної ради",
        ),
    ),
    (
        "nbu",
        "Національний банк України",
        (
            "національний банк україни",
            "правління національного банку україни",
            "постанова правління національного банку",
            "нбу",
        ),
    ),
    (
        "cabinet_ministers",
        "Кабінет Міністрів України",
        (
            "кабінет міністрів україни",
            "постанова кабінету міністрів",
            "розпорядження кабінету міністрів",
            "прем'єр-міністр україни",
            "прем'єр міністр україни",
        ),
    ),
    (
        "president",
        "Президент України",
        (
            "президент україни",
            "указ президента україни",
            "розпорядження президента україни",
        ),
    ),
    (
        "minjust",
        "Міністерство юстиції України",
        (
            "міністерство юстиції україни",
            "мін'юст",
            "міністр юстиції україни",
        ),
    ),
    (
        "nssmc",
        "Національна комісія з цінних паперів та фондового ринку",
        (
            "національна комісія з цінних паперів та фондового ринку",
            "нкцпфр",
        ),
    ),
    (
        "tax_service",
        "Державна податкова служба України",
        (
            "державна податкова служба україни",
            "дпс україни",
        ),
    ),
    (
        "constitutional_court",
        "Конституційний Суд України",
        ("конституційний суд україни",),
    ),
    (
        "supreme_court",
        "Верховний Суд",
        ("верховний суд", "велика палата верховного суду"),
    ),
)


def ingest_tag_filters(settings: Settings) -> set[str]:
    """Returns normalized ingest tag filters from settings."""

    return {
        tag.strip().casefold()
        for tag in settings.ingest_tag_filters.split(",")
        if tag.strip()
    }


def data_dir(settings: Settings) -> Path:
    """Returns the absolute path to the configured data directory."""

    return BASE_DIR / settings.data_dir


def index_dir(settings: Settings) -> Path:
    """Returns the absolute path to the persistent vector index directory."""

    return BASE_DIR / settings.index_dir


def _normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\r", "\n").split("\n")]
    collapsed = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", collapsed).strip()


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def _load_text_file(path: Path) -> str:
    return _normalize_text(_decode_text(path.read_bytes()))


def _structured_data_to_text(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _load_json_file(path: Path) -> str:
    raw_text = _decode_text(path.read_bytes())
    try:
        return _normalize_text(_structured_data_to_text(json.loads(raw_text)))
    except json.JSONDecodeError:
        logger.warning("Invalid JSON, loading as plain text: %s", path)
        return _normalize_text(raw_text)


def _load_yaml_file(path: Path) -> str:
    raw_text = _decode_text(path.read_bytes())
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML is not installed, loading YAML as plain text: %s", path)
        return _normalize_text(raw_text)

    try:
        return _normalize_text(_structured_data_to_text(yaml.safe_load(raw_text)))
    except Exception as exc:
        logger.warning("Invalid YAML, loading as plain text: %s (%s)", path, exc)
        return _normalize_text(raw_text)


def _load_html_file(path: Path) -> str:
    raw_html = _decode_text(path.read_bytes())
    match = re.search(r"(?is)<div\b[^>]*\bid\s*=\s*['\"]?article['\"]?[^>]*>", raw_html)
    if match:
        raw_html = raw_html[match.start() :]
    text = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", raw_html)
    text = re.sub(
        r"(?i)<\s*/?\s*(br|p|div|tr|li|pre|h[1-6]|table|section|article)\b[^>]*>",
        "\n",
        text,
    )
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return _normalize_text(unescape(text))


def _load_docx_file(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
        if text.strip():
            paragraphs.append(text)
    return _normalize_text("\n".join(paragraphs))


def _run_converter(command: list[str], source: Path) -> str:
    result = subprocess.run(
        command + [str(source)],
        check=False,
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        return ""
    return _normalize_text(_decode_text(result.stdout))


def _load_doc_file(path: Path) -> str:
    for executable in ("antiword", "catdoc"):
        if shutil.which(executable):
            text = _run_converter([executable], path)
            if text:
                return text

    if shutil.which("soffice"):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    tmp,
                    str(path),
                ],
                check=False,
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                converted = Path(tmp) / f"{path.stem}.txt"
                if converted.is_file():
                    return _load_text_file(converted)

    raw = path.read_bytes()
    text = _decode_text(raw)
    printable_runs = re.findall(r"[^\x00-\x08\x0b\x0c\x0e-\x1f]{20,}", text)
    return _normalize_text("\n".join(printable_runs))


def _classify_issuer(text: str, path: Path) -> tuple[str, str]:
    haystack = f"{path.name}\n{text[:12000]}".casefold()
    for key, label, patterns in ISSUER_RULES:
        if any(pattern in haystack for pattern in patterns):
            return key, label
    return "other", "Інші"


def _issuer_match_keys(text: str, path: Path) -> list[str]:
    haystack = f"{path.name}\n{text[:12000]}".casefold()
    return [
        key
        for key, _label, patterns in ISSUER_RULES
        if any(pattern in haystack for pattern in patterns)
    ]


def _metadata_for(path: Path, text: str) -> dict[str, str]:
    issuer_key, issuer = _classify_issuer(text, path)
    file_type = path.suffix.lower().lstrip(".")
    tags = [
        f"issuer:{issuer_key}",
        f"format:{file_type}",
    ]
    tags.extend(f"issuer_match:{key}" for key in _issuer_match_keys(text, path))
    return {
        "source": str(path),
        "file_name": path.name,
        "file_type": file_type,
        "issuer": issuer,
        "issuer_key": issuer_key,
        "tags": ",".join(tags),
    }


def _document_matches_filters(document: Document, filters: set[str]) -> bool:
    if not filters:
        return True
    raw_tags = str((document.metadata or {}).get("tags") or "")
    tags = {tag.strip().casefold() for tag in raw_tags.split(",") if tag.strip()}
    return bool(tags & filters)


def _load_single_document(path: Path) -> Document | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".txt":
            text = _load_text_file(path)
        elif suffix == ".json":
            text = _load_json_file(path)
        elif suffix in {".yaml", ".yml"}:
            text = _load_yaml_file(path)
        elif suffix in {".html", ".htm"}:
            text = _load_html_file(path)
        elif suffix == ".docx":
            text = _load_docx_file(path)
        elif suffix == ".doc":
            text = _load_doc_file(path)
        else:
            return None
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None

    if not text:
        logger.warning("Skipping empty document: %s", path)
        return None
    return Document(page_content=text, metadata=_metadata_for(path, text))


def _tag_pdf_documents(documents: list[Document]) -> list[Document]:
    tagged: list[Document] = []
    for doc in documents:
        original = dict(doc.metadata or {})
        source = Path(str(original.get("source") or original.get("file_path") or "unknown.pdf"))
        enriched = _metadata_for(source, doc.page_content)
        enriched.update(original)
        tagged.append(Document(page_content=doc.page_content, metadata=enriched))
    return tagged


def load_langchain_documents(settings: Settings) -> list[Document]:
    """Loads supported local files as LangChain Document objects with issuer tags."""

    root = data_dir(settings)
    if not root.is_dir():
        return []

    documents: list[Document] = []
    filters = ingest_tag_filters(settings)
    logger.info("Loading documents from %s", root)
    if filters:
        logger.info("Applying ingest tag filters: %s", ", ".join(sorted(filters)))
    loader = PyPDFDirectoryLoader(str(root), glob="**/*.pdf")
    pdf_documents = [
        document
        for document in _tag_pdf_documents(loader.load())
        if _document_matches_filters(document, filters)
    ]
    documents.extend(pdf_documents)
    if pdf_documents:
        logger.info("Loaded %d matching PDF page documents", len(pdf_documents))

    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS - {".pdf"}
    ]
    logger.info("Found %d non-PDF files to load", len(paths))
    skipped = 0
    for index, path in enumerate(paths, start=1):
        if not path.is_file():
            continue
        document = _load_single_document(path)
        if document is not None and _document_matches_filters(document, filters):
            documents.append(document)
        else:
            skipped += 1
        if index == 1 or index % 100 == 0 or index == len(paths):
            logger.info(
                "Scanned %d/%d non-PDF files; matched=%d skipped=%d",
                index,
                len(paths),
                len(documents),
                skipped,
            )

    logger.info("Loaded %d source documents total", len(documents))
    return documents


def split_langchain_documents(
    settings: Settings, documents: list[Document]
) -> list[Document]:
    """Splits documents into overlapping text chunks using configured size and overlap."""

    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
