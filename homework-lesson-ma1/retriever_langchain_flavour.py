"""LangChain-side hybrid search over Chroma and BM25 with cross-encoder reranking for RAG.
Гібридний пошук на стороні LangChain: Chroma + BM25 із переранжуванням cross-encoder для RAG."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import Settings
from kb_common import index_dir

# Встановлюємо UTF-8 кодування для stdout, щоб уникнути проблем з кирилицею
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


def reciprocal_rank_fusion(
        *ranked_lists: list[Document],
        k: float = 60.0,
        limit: int = 20,
) -> list[Document]:
    """Merge multiple ranked document lists with reciprocal rank fusion and cap the list length.
    Об'єднує кілька ранжованих списків документів методом reciprocal rank fusion і обмежує довжину."""
    scores: dict[str, float] = {}
    by_text: dict[str, Document] = {}
    for results in ranked_lists:
        for rank, doc in enumerate(results):
            text = doc.page_content
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)
            by_text.setdefault(text, doc)
    return sorted(
        by_text.values(),
        key=lambda d: scores[d.page_content],
        reverse=True,
    )[:limit]


_ce_instance: CrossEncoder | None = None
_ce_name: str | None = None


def _cross_encoder(model_name: str) -> CrossEncoder:
    """Return a cached CrossEncoder instance, reinitializing it when the model name changes.
    Повертає кешований екземпляр CrossEncoder і переініціалізує його при зміні назви моделі."""
    # Lazy import - завантажуємо sentence_transformers тільки при першому використанні
    from sentence_transformers import CrossEncoder

    global _ce_instance, _ce_name
    if _ce_instance is None or _ce_name != model_name:
        _ce_instance = CrossEncoder(model_name)
        _ce_name = model_name
    return _ce_instance


def _rerank_documents(
    query: str,
    documents: list[Document],
    model_name: str,
    top_n: int,
) -> list[Document]:
    """Rerank documents with a cross-encoder for the query and keep only the top_n results.
    Переранжовує документи cross-encoder за запитом і залишає лише top_n результатів."""
    if not documents:
        return []
    model = _cross_encoder(model_name)
    pairs = [[query, d.page_content] for d in documents]
    scores = model.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked[:top_n]]


def _format_hits(documents: list[Document]) -> str:
    """Format retrieved LangChain documents as numbered Markdown snippets with source labels.
    Форматує знайдені документи LangChain як нумеровані Markdown-фрагменти з позначкою джерела."""
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file_path") or "unknown"
        parts.append(f"### Snippet {i} (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def hybrid_search_langchain(query: str, settings: Settings | None = None) -> str:
    """Run Chroma similarity search plus BM25, fuse, rerank, and return a trimmed Markdown string.
    Виконує пошук у Chroma та BM25, злиття, rerank і повертає обрізаний Markdown-текст."""
    settings = settings or Settings()
    idx = index_dir(settings)
    if not idx.is_dir():
        return (
            "Локальний індекс не знайдено. Спочатку виконайте "
            "`make ingest-langchain` після додавання PDF у каталог data/."
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

    # Витягуємо всі документи з Chroma для BM25 (уникаємо повторного завантаження PDF)
    collection = store._collection
    all_data = collection.get(include=["documents", "metadatas"])

    if not all_data["documents"]:
        return (
            "База знань порожня: у data/ немає PDF або не вдалося їх прочитати. "
            "Додайте файли та запустіть інжест."
        )

    # Створюємо Document об'єкти з даних Chroma
    splits = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    vector_hits = store.similarity_search(
        query,
        k=settings.retrieval_vector_k,
    )
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = settings.retrieval_bm25_k
    bm25_hits = bm25.invoke(query)

    fused = reciprocal_rank_fusion(
        vector_hits,
        bm25_hits,
        limit=settings.retrieval_fusion_top_n,
    )
    reranked = _rerank_documents(
        query,
        fused,
        settings.reranker_model,
        settings.rerank_top_n,
    )
    text = _format_hits(reranked)
    limit = settings.max_knowledge_chars
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."
