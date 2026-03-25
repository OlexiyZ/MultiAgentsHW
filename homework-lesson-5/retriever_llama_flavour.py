"""Hybrid retrieval (QueryFusionRetriever: vector + BM25) + SentenceTransformerRerank — LlamaIndex."""

from __future__ import annotations

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi

from config import Settings
from kb_common import index_dir, load_llama_nodes


class RankBm25LlamaRetriever(BaseRetriever):
    """BM25 over the same chunked nodes used alongside the vector index."""

    def __init__(self, nodes: list[BaseNode], similarity_top_k: int) -> None:
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        tokenized = []
        for node in nodes:
            text = node.get_content(metadata_mode="none")
            tokenized.append(text.lower().split())
        self._bm25 = BM25Okapi(tokenized)
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        q_tokens = query_bundle.query_str.lower().split()
        scores = self._bm25.get_scores(q_tokens)
        idxs = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[: self._similarity_top_k]
        return [
            NodeWithScore(node=self._nodes[i], score=float(scores[i])) for i in idxs
        ]


def hybrid_search_llama(query: str, settings: Settings | None = None) -> str:
    settings = settings or Settings()
    idx_path = index_dir(settings)
    if not idx_path.is_dir():
        return (
            "Локальний індекс не знайдено. Спочатку виконайте "
            "`make ingest-llama` після додавання PDF у каталог data/."
        )

    try:
        client = chromadb.PersistentClient(path=str(idx_path))
        collection = client.get_collection(settings.chroma_collection)
    except Exception:
        return (
            "Не вдалося відкрити колекцію Chroma. Переконайтеся, що інжест LlamaIndex "
            "виконано для поточного каталогу index/."
        )

    nodes = load_llama_nodes(settings)
    if not nodes:
        return (
            "База знань порожня: у data/ немає PDF або не вдалося їх прочитати. "
            "Додайте файли та запустіть інжест."
        )

    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    vector_retriever = index.as_retriever(
        similarity_top_k=settings.retrieval_vector_k,
    )
    bm25_retriever = RankBm25LlamaRetriever(
        nodes,
        similarity_top_k=settings.retrieval_bm25_k,
    )
    fusion = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        num_queries=1,
        use_async=False,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=settings.retrieval_fusion_top_n,
    )
    rerank = SentenceTransformerRerank(
        model=settings.reranker_model,
        top_n=settings.rerank_top_n,
    )

    bundle = QueryBundle(query_str=query)
    fused_nodes = fusion.retrieve(bundle)
    final_nodes = rerank.postprocess_nodes(fused_nodes, query_bundle=bundle)

    parts: list[str] = []
    for i, nws in enumerate(final_nodes, start=1):
        node = nws.node
        meta = node.metadata or {}
        source = meta.get("file_path") or meta.get("file_name") or "unknown"
        parts.append(
            f"### Snippet {i} (source: {source})\n{node.get_content(metadata_mode='none')}"
        )
    text = "\n\n".join(parts)
    limit = settings.max_knowledge_chars
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."
