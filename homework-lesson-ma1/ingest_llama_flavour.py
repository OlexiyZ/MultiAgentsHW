"""Script to ingest PDFs from data/ into Chroma using LlamaIndex and OpenAI embeddings.
Скрипт інжесту PDF з data/ у Chroma за допомогою LlamaIndex та ембеддингів OpenAI."""

from __future__ import annotations

import shutil

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import Settings
from kb_common import index_dir, load_llama_nodes


def main() -> None:
    """Rebuild the Chroma index from PDFs in data/ using the LlamaIndex ingestion pipeline.
    Перебудовує індекс Chroma з PDF у data/ за допомогою пайплайну інжесту LlamaIndex."""
    settings = Settings()
    target = index_dir(settings)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    nodes = load_llama_nodes(settings)
    if not nodes:
        raise SystemExit(
            "No PDF nodes produced. Add .pdf files under data/ and run again."
        )

    client = chromadb.PersistentClient(path=str(target))
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    collection = client.get_or_create_collection(settings.chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"Ingested {len(nodes)} LlamaIndex nodes into {target}")


if __name__ == "__main__":
    main()
