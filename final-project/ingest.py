from __future__ import annotations

import logging
import shutil

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import Settings, configure_logging
from kb_common import index_dir, load_langchain_documents, split_langchain_documents


def main() -> None:
    """Rebuilds the Chroma index from supported files in data/."""

    configure_logging()
    logger = logging.getLogger(__name__)
    settings = Settings()
    target = index_dir(settings)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    raw = load_langchain_documents(settings)
    splits = split_langchain_documents(settings, raw)
    if not splits:
        raise SystemExit(
            "No chunks produced. Add .pdf, .txt, .html, .htm, .doc, or .docx "
            "files under data/ and run again."
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
    logger.info("Ingested %d chunks into %s", len(splits), target)
    print(f"Ingested {len(splits)} chunks into {target}")


if __name__ == "__main__":
    main()
