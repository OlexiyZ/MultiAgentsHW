# Homework Lesson 5 — RAG Agent

The Research Agent from `homework_lesson_3` is extended with a `knowledge_search` tool backed by a hybrid retrieval pipeline (semantic + BM25) and cross-encoder reranking. The agent now decides on its own whether to search the web or the local knowledge base.

It is assumed that one has `make` utility and has executed a setup step, e.g. installing `uv` and installing the necessary environment. Moreover, it is assumed that `.env` file has been created in `homework_lesson_5` folder and populated by `OPENAI_API_KEY=<HERE COMES AN OPEN API KEY>`.

## How to run

Before running the agent, ingest documents from `data/` into the Chroma vector store:

```bash
make ingest-llama       # ingest using LlamaIndex pipeline
make ingest-langchain   # ingest using LangChain pipeline (alternative)
```

Then start the agent:

```bash
make run-rag            # interactive RAG agent
make run-rag-debug      # with debug logging (tool calls and results shown)
```

## Usage

| Command | Description |
| --- | --- |
| `make ingest-llama` | Ingest documents into Chroma (LlamaIndex flavour) |
| `make ingest-langchain` | Ingest documents into Chroma (LangChain flavour) |
| `make run-rag` | Run RAG agent (interactive) |
| `make run-rag-debug` | Run RAG agent in --debug mode |
| `make test_lesson_5` | Run unit tests |

```bash
make help           # list all targets
```
## Demos
The ingestion pipeline's demo can be seen below:

![demo ingestion](./static_files/ingestion_pipeline.gif)

An example of a conversation with an agent:

![conversation with an agent](./static_files/demo_agent.gif)

## Retrieval pipeline

Both retriever flavours implement the same three-stage pipeline:

1. **Semantic search** — cosine similarity search over embeddings stored in ChromaDB (`text-embedding-3-large`)
2. **BM25 search** — lexical keyword search over the same chunks
3. **Reranking** — cross-encoder (`BAAI/bge-reranker-base`) re-scores and filters the fused candidate set

The LlamaIndex flavour (`retriever_llama_flavour.py`) uses `QueryFusionRetriever` + `SentenceTransformerRerank`.
The LangChain flavour (`retriever_langchain_flavour.py`) uses `EnsembleRetriever` + a cross-encoder reranker.

## Knowledge base

PDF documents placed in `data/` are ingested into `index/` (Chroma persistent store). The current knowledge base contains:

- `langchain.pdf`
- `large-language-model.pdf`
- `retrieval-augmented-generation.pdf`

## Structure

```
homework_lesson_5/
├── main.py                       # Entry point — interactive REPL loop (LangGraph agent)
├── agent.py                      # Agent setup (LLM, tools, memory, create_agent)
├── tools.py                      # web_search, read_url, write_report, knowledge_search
├── retriever_llama_flavour.py    # Hybrid retrieval + reranking (LlamaIndex)
├── retriever_langchain_flavour.py# Hybrid retrieval + reranking (LangChain)
├── ingest_llama_flavour.py       # Ingestion pipeline: docs → chunks → Chroma (LlamaIndex)
├── ingest_langchain_flavour.py   # Ingestion pipeline: docs → chunks → Chroma (LangChain)
├── config.py                     # Settings, paths, system prompt loading
├── system_prompt.yaml            # System prompt for the agent
├── data/                         # PDF documents for ingestion
├── index/                        # Chroma vector store (created at runtime)
└── output/                       # Generated reports (created at runtime)
```