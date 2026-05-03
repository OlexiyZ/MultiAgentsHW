from __future__ import annotations

import logging
import shutil

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import Settings, configure_logging
from kb_common import index_dir, load_langchain_documents, split_langchain_documents


EMBED_BATCH_SIZE = 256


def main() -> None:
    """Builds or appends to the Chroma index from supported files in data/."""

    configure_logging()
    logger = logging.getLogger(__name__)
    settings = Settings()
    target = index_dir(settings)

    logger.info("Loading source documents")
    raw = load_langchain_documents(settings)
    logger.info("Splitting %d source documents", len(raw))
    splits = split_langchain_documents(settings, raw)
    if not splits:
        raise SystemExit(
            "No chunks produced. Add .pdf, .txt, .json, .yaml, .yml, .html, "
            ".htm, .doc, or .docx files under data/ and run again."
        )
    logger.info("Produced %d chunks", len(splits))

    logger.info(
        "Embedding chunks into Chroma collection %s at %s (rebuild=%s)",
        settings.chroma_collection,
        target,
        settings.ingest_rebuild_index,
    )
    if settings.ingest_rebuild_index and target.exists():
        logger.info("Removing existing Chroma index at %s", target)
        shutil.rmtree(target)
    elif target.exists():
        logger.info("Appending to existing Chroma index at %s", target)
    else:
        logger.info("Creating new Chroma index at %s", target)
    target.mkdir(parents=True, exist_ok=True)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(target),
        collection_name=settings.chroma_collection,
    )
    for start in range(0, len(splits), EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, len(splits))
        store.add_documents(splits[start:end])
        logger.info("Embedded %d/%d chunks", end, len(splits))
    logger.info("Ingested %d chunks into %s", len(splits), target)
    print(f"Ingested {len(splits)} chunks into {target}")


if __name__ == "__main__":
    main()
