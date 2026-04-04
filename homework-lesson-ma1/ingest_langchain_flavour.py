"""Script to ingest PDFs from data/ into a persistent Chroma store via LangChain embeddings.
Скрипт інжесту PDF з data/ у персистентне сховище Chroma через ембеддинги LangChain."""

from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import Settings
from kb_common import index_dir, load_langchain_documents, split_langchain_documents


def main() -> None:
    """Rebuild the Chroma index from PDFs in data/ using the LangChain ingestion pipeline.
    Перебудовує індекс Chroma з PDF у data/ за допомогою пайплайну інжесту LangChain."""
    settings = Settings()
    target = index_dir(settings)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    raw = load_langchain_documents(settings)
    splits = split_langchain_documents(settings, raw)
    if not splits:
        raise SystemExit(
            "No PDF chunks produced. Add .pdf files under data/ and run again."
        )

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(target),
        collection_name=settings.chroma_collection,
    )
    print(f"Ingested {len(splits)} LangChain chunks into {target}")


if __name__ == "__main__":
    main()
