from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import Settings, preview_for_log
from kb_common import index_dir


logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    *ranked_lists: list[Document], k: float = 60.0, limit: int = 20
) -> list[Document]:
    scores: dict[str, float] = {}
    by_text: dict[str, Document] = {}
    for results in ranked_lists:
        for rank, doc in enumerate(results):
            text = doc.page_content
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)
            by_text.setdefault(text, doc)
    return sorted(by_text.values(), key=lambda d: scores[d.page_content], reverse=True)[
        :limit
    ]


def _format_hits(documents: list[Document]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file_path") or "unknown"
        parts.append(f"### Snippet {i} (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def hybrid_search(query: str, settings: Settings | None = None) -> str:
    settings = settings or Settings()
    idx = index_dir(settings)
    if not idx.is_dir():
        logger.warning("Retriever: index directory missing: %s", idx)
        return (
            "Локальний індекс не знайдено. Спочатку виконайте `python ingest.py` "
            "після додавання PDF у каталог data/."
        )

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    store = Chroma(
        persist_directory=str(idx),
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )

    collection = store._collection
    all_data = collection.get(include=["documents", "metadatas"])
    if not all_data["documents"]:
        logger.warning("Retriever: Chroma collection has no documents")
        return (
            "База знань порожня: у data/ немає PDF або не вдалося їх прочитати. "
            "Додайте файли та запустіть інжест."
        )

    splits = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    vector_hits = store.similarity_search(query, k=settings.retrieval_vector_k)
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = settings.retrieval_bm25_k
    bm25_hits = bm25.invoke(query)

    fused = reciprocal_rank_fusion(
        vector_hits,
        bm25_hits,
        limit=settings.retrieval_fusion_top_n,
    )
    top_hits = fused[: settings.rerank_top_n]
    text = _format_hits(top_hits)
    logger.debug(
        "Retriever hybrid_search: query=%s fused=%d top=%d out_chars=%d",
        preview_for_log(query, 300),
        len(fused),
        len(top_hits),
        len(text),
    )

    limit = settings.max_knowledge_chars
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."
