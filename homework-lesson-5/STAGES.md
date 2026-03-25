### Крок 1 — Залежності та середовище
- Додано **`requirements.txt`** (Chroma, LangChain, LlamaIndex, `rank-bm25`, `sentence-transformers`, `pypdf`, тощо).
- **`Makefile`**: `make venv` створює `.venv` і ставить залежності; цілі `ingest-llama`, `ingest-langchain`, `run-rag`, `run-rag-debug`, `test_lesson_5`, `help`.
- У **`.gitignore`** додано `index/` і `.venv/`.

### Крок 2 — Конфіг і промпт
- **`config.py`**: шляхи `data/` та `index/`, модель ембеддингів (`text-embedding-3-large`), колекція Chroma, **`KNOWLEDGE_FLAVOUR`** (`langchain` або `llama`), параметри чанків, retrieval і reranker, ліміт тексту для `knowledge_search`.
- **`system_prompt.yaml`**: системний промпт з правилами, коли викликати **`knowledge_search`**, а коли веб.
- **`config.py`** читає YAML через **`load_system_prompt()`**; якщо файлу немає — є запасний текст.

### Крок 3 — Спільна робота з PDF (`kb_common.py`)
- Один набір налаштувань чанків для інжесту й BM25.
- **LangChain**: `PyPDFDirectoryLoader` + `RecursiveCharacterTextSplitter`.
- **LlamaIndex**: `SimpleDirectoryReader` + `SentenceSplitter` → вузли для індексу та BM25.

### Крок 4 — Інжест
- **`ingest_langchain_flavour.py`**: PDF → чанки → **Chroma** (OpenAI embeddings).
- **`ingest_llama_flavour.py`**: PDF → вузли → **Chroma** через LlamaIndex.
- Перед інжестом каталог **`index/`** очищається (повна перебудова індексу).

### Крок 5 — Retrieval
- **`retriever_langchain_flavour.py`**: семантичний пошук у Chroma + **BM25** (`BM25Retriever`) → **RRF** (`reciprocal_rank_fusion`) → rerank **`BAAI/bge-reranker-base`** через `CrossEncoder`.
- **`retriever_llama_flavour.py`**: завантаження індексу з Chroma → **`QueryFusionRetriever`** (vector + власний **BM25** на `rank_bm25`) у режимі **reciprocal rank** → **`SentenceTransformerRerank`**.

### Крок 6 — Агент
- **`tools.py`**: інструмент **`knowledge_search`**; бекенд обирається з **`KNOWLEDGE_FLAVOUR`** (має **співпадати** з тим, яким `make ingest-*` ви будували індекс).
- **`main.py`**: підказка «RAG Research Agent», прапорець **`--debug`** (друк усіх повідомлень графа, включно з tool calls).

### Крок 7 — Тести
- **`tests/test_lesson_5.py`**: RRF, наявність tool, обрізання довгої відповіді, fallback промпта.
- **`make test_lesson_5`** — усі **4 тести проходять**.

### Крок 8 — Дані
- У **`data/`** лише **`.gitkeep`**. У репозиторії **немає PDF**; потрібно покласти туди файли з README (`langchain.pdf`, `large-language-model.pdf`, `retrieval-augmented-generation.pdf` або свої).

---

