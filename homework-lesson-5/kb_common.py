"""Shared PDF loading and text chunking utilities reused by ingest pipelines and retrievers.
Спільні утиліти завантаження PDF і розбиття тексту для пайплайнів інжесту та ретриверів."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import BASE_DIR, Settings

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode


def data_dir(settings: Settings) -> Path:
    """Return the absolute path to the PDF data directory from settings.
    Повертає абсолютний шлях до каталогу з PDF-даними згідно з налаштуваннями."""
    return BASE_DIR / settings.data_dir


def index_dir(settings: Settings) -> Path:
    """Return the absolute path to the Chroma index directory from settings.
    Повертає абсолютний шлях до каталогу індексу Chroma згідно з налаштуваннями."""
    return BASE_DIR / settings.index_dir


def list_pdf_paths(settings: Settings) -> list[Path]:
    """List sorted PDF file paths directly under the configured data directory.
    Повертає відсортовані шляхи до PDF-файлів безпосередньо в налаштованому каталозі data."""
    root = data_dir(settings)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.pdf") if p.is_file())


def load_langchain_documents(settings: Settings) -> list[Document]:
    """Load all PDFs from the data directory as LangChain Document objects.
    Завантажує всі PDF з каталогу data як об’єкти LangChain Document."""
    root = data_dir(settings)
    if not root.is_dir():
        return []
    loader = PyPDFDirectoryLoader(str(root), glob="**/*.pdf")
    return loader.load()


def split_langchain_documents(
    settings: Settings, documents: list[Document]
) -> list[Document]:
    """Split LangChain documents into chunks using configured size and overlap.
    Розбиває документи LangChain на чанки згідно з налаштованим розміром і перекриттям."""
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def load_langchain_splits(settings: Settings) -> list[Document]:
    """Load PDFs and return LangChain text chunks ready for BM25 or ingest checks.
    Завантажує PDF і повертає текстові чанки LangChain для BM25 або перевірок інжесту."""
    return split_langchain_documents(settings, load_langchain_documents(settings))


def load_llama_nodes(settings: Settings) -> list[BaseNode]:
    """Load PDFs via LlamaIndex and split them into nodes for indexing or BM25.
    Завантажує PDF через LlamaIndex і розбиває їх на вузли для індексації або BM25."""
    # Lazy import - завантажуємо LlamaIndex тільки коли потрібно
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter

    root = data_dir(settings)
    if not root.is_dir() or not list(root.glob("*.pdf")):
        return []
    reader = SimpleDirectoryReader(
        input_dir=str(root),
        required_exts=[".pdf"],
        recursive=True,
    )
    documents = reader.load_data()
    if not documents:
        return []
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.get_nodes_from_documents(documents)
