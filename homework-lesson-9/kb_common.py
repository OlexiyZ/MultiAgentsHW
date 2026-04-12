from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import BASE_DIR, Settings


def data_dir(settings: Settings) -> Path:
    return BASE_DIR / settings.data_dir


def index_dir(settings: Settings) -> Path:
    return BASE_DIR / settings.index_dir


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
