"""Shared helpers: paths, PDF loading, chunking (same settings for ingest + BM25)."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from config import BASE_DIR, Settings


def data_dir(settings: Settings) -> Path:
    return BASE_DIR / settings.data_dir


def index_dir(settings: Settings) -> Path:
    return BASE_DIR / settings.index_dir


def list_pdf_paths(settings: Settings) -> list[Path]:
    root = data_dir(settings)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.pdf") if p.is_file())


def load_langchain_documents(settings: Settings) -> list[Document]:
    root = data_dir(settings)
    if not root.is_dir():
        return []
    loader = PyPDFDirectoryLoader(str(root), glob="**/*.pdf")
    return loader.load()


def split_langchain_documents(
    settings: Settings, documents: list[Document]
) -> list[Document]:
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def load_langchain_splits(settings: Settings) -> list[Document]:
    return split_langchain_documents(settings, load_langchain_documents(settings))


def load_llama_nodes(settings: Settings) -> list[BaseNode]:
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
