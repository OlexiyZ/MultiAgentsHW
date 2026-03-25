# Покроковий опис роботи застосунку (урок 5)

Кожен крок — **одне речення**: які **функції** з якого **модуля** беруть участь і **що вони роблять**. Окремо зазначено виклики з **сторонніх бібліотек** (LangChain, LangGraph тощо), бо вони не є нашими модулями.

---

## 1. Старт Python і завантаження конфігурації

1. При імпорті `config` викликається **`config.load_system_prompt()`**, яка читає `system_prompt.yaml` (або підставляє текст за замовчуванням) і формує рядок системного промпта.
2. У модулі `config` змінна **`SYSTEM_PROMPT`** отримує результат **`load_system_prompt()`**, щоб інші модулі могли імпортувати готовий промпт без повторного читання файлу.
3. При імпорті `tools` створюється **`settings = config.Settings()`**, який зчитує змінні з `.env` і поля за замовчуванням (модель, шляхи `data/` та `index/`, `KNOWLEDGE_FLAVOUR` тощо).

---

## 2. Побудова агента (імпорт `agent`)

1. У **`agent.py`** знову викликається **`config.Settings()`** і будується об’єкт **`langchain_openai.ChatOpenAI`** з ключем і назвою моделі з налаштувань (це не наш модуль, а фабрика LLM).
2. Створюється **`langgraph.checkpoint.memory.MemorySaver`** як збереження стану діалогу між кроками графа (зовнішня бібліотека).
3. Викликається **`langchain.agents.create_agent()`** з моделлю, списком **`tools.TOOLS`**, **`config.SYSTEM_PROMPT`** і checkpointer — формується готовий **`agent`** (граф LangGraph/LangChain).
4. Константа **`AGENT_CONFIG`** у **`agent.py`** задає `thread_id` і **`recursion_limit`** з **`settings.max_iterations`**, щоб обмежити глибину циклу викликів інструментів.

---

## 3. Запуск CLI (`main.py`)

1. Під **`if __name__ == "__main__"`** викликається **`main.main()`**, яка розбирає аргументи (наприклад `--debug`) і за потреби налаштовує **`logging.basicConfig`** для рівня DEBUG.
2. У циклі з **`input()`** зчитується запит користувача; для виходу використовуються команди `exit`/`quit` або переривання вводу.
3. На кожен запит викликається **`agent.invoke(..., config=AGENT_CONFIG)`** (метод графа з LangChain) — всередині модель вирішує, чи викликати tools і в якому порядку.
4. Після відповіді викликається **`main._extract_last_ai_message(result["messages"])`**, яка проходить список повідомлень з кінця й повертає текст останнього повідомлення типу AI для виводу в консоль.
5. У режимі `--debug` додатково друкуються всі елементи **`result["messages"]`**, щоб бачити виклики інструментів і їхні результати.

---

## 4. Інструмент `web_search` (`tools.py`)

1. LangChain викликає обгортку tool над **`tools.web_search`**, яка обмежує **`max_results`** і викликає **`DDGS().text(...)`** (бібліотека `ddgs`) для пошуку в DuckDuckGo.
2. Для кожного результату заголовок і сніпет проходять через **`tools._clip_text`**, щоб скоротити довжину рядків перед поверненням у модель.

---

## 5. Інструмент `read_url` (`tools.py`)

1. Викликається **`tools.read_url`**, яка перевіряє URL через **`urllib.parse.urlparse`** і відхиляє не-HTTP(S) посилання.
2. **`trafilatura.fetch_url`** завантажує HTML, **`trafilatura.extract`** витягує основний текст, після чого **`tools._clip_text`** обрізає результат до **`settings.max_url_content_length`**.

---

## 6. Інструмент `write_report` (`tools.py`)

1. **`tools.write_report`** викликає **`tools._safe_report_path`**, яка нормалізує ім’я файла (лише базове ім’я, суфікс `.md`), створює каталог **`BASE_DIR / settings.output_dir`** і повертає повний шлях збереження.
2. Файл записується методом **`Path.write_text`**, а користувачу повертається рядок із підтвердженням шляху або повідомленням про помилку.

---

## 7. Інструмент `knowledge_search` і бекенд ретриву (`tools.py` + ретривери)

1. **`tools.knowledge_search`** викликає **`tools._knowledge_backend_search`**, яка за **`settings.knowledge_flavour`** або імпортує та викликає **`retriever_llama_flavour.hybrid_search_llama`**, або **`retriever_langchain_flavour.hybrid_search_langchain`**.
2. Потім **`knowledge_search`** обрізає рядок відповіді до **`settings.max_knowledge_chars`**, якщо він довший за ліміт.

### 7а. Гілка LangChain (`retriever_langchain_flavour.py`)

1. **`hybrid_search_langchain`** викликає **`kb_common.index_dir`**, щоб отримати шлях до каталогу індексу Chroma, і перевіряє наявність індексу та PDF.
2. **`kb_common.load_langchain_splits`** внутрішньо викликає **`load_langchain_documents`** (через **`kb_common.data_dir`** і завантажувач PDF) і **`split_langchain_documents`**, щоб отримати ті самі чанки, що потрібні для BM25.
3. Відкривається **`langchain_chroma.Chroma`** з **`langchain_openai.OpenAIEmbeddings`**, після чого **`store.similarity_search`** виконує семантичний пошук по запиту.
4. **`langchain_community.retrievers.BM25Retriever.from_documents(splits)`** будує лексичний пошук; **`bm25.invoke(query)`** повертає кандидатів BM25.
5. **`retriever_langchain_flavour.reciprocal_rank_fusion`** об’єднує два ранжовані списки документів методом RRF.
6. **`retriever_langchain_flavour._rerank_documents`** через **`_cross_encoder`** (кешований **`sentence_transformers.CrossEncoder`**) переранжовує кандидатів і залишає top-N.
7. **`retriever_langchain_flavour._format_hits`** збирає фрагменти в один Markdown-текст; за потреби рядок ще раз обрізається в кінці **`hybrid_search_langchain`** за **`max_knowledge_chars`**.

### 7б. Гілка LlamaIndex (`retriever_llama_flavour.py`)

1. **`hybrid_search_llama`** викликає **`kb_common.index_dir`**, відкриває **`chromadb.PersistentClient`** і колекцію Chroma, а також **`kb_common.load_llama_nodes`** для чанків під BM25.
2. **`llama_index.embeddings.openai.OpenAIEmbedding`** і **`VectorStoreIndex.from_vector_store`** відновлюють векторний індекс зі сховища; **`index.as_retriever`** дає семантичний ретривер.
3. Створюється **`retriever_llama_flavour.RankBm25LlamaRetriever`**: у **`__init__`** будується **`rank_bm25.BM25Okapi`** по токенізованих текстах вузлів; у **`_retrieve`** рахуються бали BM25 для запиту.
4. **`llama_index.core.retrievers.QueryFusionRetriever`** з **`FUSION_MODES.RECIPROCAL_RANK`** зливає результати векторного та BM25-ретриверів (без додаткової генерації запитів LLM при `num_queries=1`).
5. **`llama_index.core.postprocessor.SentenceTransformerRerank.postprocess_nodes`** переранжовує вузли; далі текст збирається в циклі в **`hybrid_search_llama`** і обрізається за **`max_knowledge_chars`**.

---

## 8. Підготовка бази: інжест LangChain (`ingest_langchain_flavour.py`)

1. **`ingest_langchain_flavour.main`** викликає **`config.Settings()`**, **`kb_common.index_dir`**, очищає каталог індексу і створює його знову.
2. **`kb_common.load_langchain_documents`** завантажує PDF, **`kb_common.split_langchain_documents`** ріже їх на чанки.
3. **`langchain_openai.OpenAIEmbeddings`** і **`langchain_chroma.Chroma.from_documents`** записують чанки з ембеддингами у персистентний Chroma у **`index/`**.

---

## 9. Підготовка бази: інжест LlamaIndex (`ingest_llama_flavour.py`)

1. **`ingest_llama_flavour.main`** викликає **`config.Settings()`**, **`kb_common.index_dir`**, очищає каталог **`index/`** і створює клієнта Chroma з потрібною колекцією.
2. **`kb_common.load_llama_nodes`** читає PDF через LlamaIndex і розбиває їх **`SentenceSplitter`** на вузли.
3. **`llama_index.vector_stores.chroma.ChromaVectorStore`**, **`StorageContext`**, **`OpenAIEmbedding`** і **`VectorStoreIndex(nodes=...)`** індексують вузли в Chroma.

---

## 10. Допоміжні функції `kb_common.py` (узагальнено)

1. **`kb_common.data_dir`** і **`kb_common.index_dir`** повертають абсолютні шляхи до `data/` та `index/` відносно **`config.BASE_DIR`** і полів **`Settings`**.
2. **`kb_common.load_langchain_documents`** використовує **`PyPDFDirectoryLoader`**, **`split_langchain_documents`** — **`RecursiveCharacterTextSplitter`**, а **`load_langchain_splits`** поєднує обидва кроки для готових чанків.
3. **`kb_common.load_llama_nodes`** використовує **`SimpleDirectoryReader`** і **`SentenceSplitter.get_nodes_from_documents`** для вузлів LlamaIndex.
4. **`kb_common.list_pdf_paths`** (за потреби) повертає відсортований список шляхів до `*.pdf` у `data/` — у поточному пайплайні інжесту/ретриву основні шляхи йдуть через завантажувачі каталогу, але функція доступна для обходу файлів.

---

## Важливо для узгодженості

- **`KNOWLEDGE_FLAVOUR`** у `.env` має відповідати тому, яким **`make ingest-langchain`** або **`make ingest-llama`** ви будували індекс, інакше векторне сховище та формат колекції можуть не збігатися з очікуваннями ретривера.
