# План виконання завдання: Homework Lesson 5 (RAG Agent)

Цей документ описує кроки для реалізації завдання з [`README.md`](./README.md) у каталозі `homework-lesson-5`, виходячи з того, що **базовий код Research Agent уже перенесено з `homework-lesson-3`** (`main.py`, `agent.py`, `tools.py`, `config.py`).

---

## Мета завдання

Розширити дослідницького агента з уроку 3 інструментом **`knowledge_search`**, який використовує **гібридний retrieval** (семантичний пошук + BM25) і **переранжування cross-encoder**. Агент має **самостійно обирати**: шукати в інтернеті (`web_search` / `read_url`) чи в **локальній базі знань** (PDF з `data/`, проіндексовані в Chroma).

---

## Поточний стан vs очікувана структура

Зараз у `homework-lesson-5` є лише фрагмент, аналогічний уроку 3. Згідно з README, потрібно додати:

| Компонент | Призначення |
|-----------|-------------|
| `ingest_llama_flavour.py` | Інжест PDF → чанки → Chroma (LlamaIndex) |
| `ingest_langchain_flavour.py` | Інжест PDF → чанки → Chroma (LangChain) |
| `retriever_llama_flavour.py` | Гібрид + rerank: QueryFusionRetriever + SentenceTransformerRerank |
| `retriever_langchain_flavour.py` | Гібрид + rerank: EnsembleRetriever + cross-encoder |
| `system_prompt.yaml` | Системний промпт (завантаження з `config.py`) |
| `data/` | PDF для бази знань (`langchain.pdf`, `large-language-model.pdf`, `retrieval-augmented-generation.pdf`) |
| `index/` | Персистентне сховище Chroma (створюється під час інжесту) |
| `Makefile` | Цілі `ingest-llama`, `ingest-langchain`, `run-rag`, `run-rag-debug`, `test_lesson_5`, `help` |
| Тести | Відповідають цілі `make test_lesson_5` |
| `pyproject.toml` / залежності | `uv` або еквівалент (як передбачає README) |

---

## Етап 0: Середовище

1. Переконатися, що встановлено **`make`**, **`uv`** (або узгоджений з курсом менеджер залежностей).
2. Створити **`.env`** у корені `homework-lesson-5` з `OPENAI_API_KEY=...` (шаблон — `.env.example`).
3. Додати залежності для: **LangChain / LangGraph**, **Chroma**, **LlamaIndex** (якщо використовується flavour), **embeddings** (`text-embedding-3-large` через OpenAI), **BM25** (залежно від обраної реалізації), **cross-encoder** (`BAAI/bge-reranker-base` через sentence-transformers або аналог у фреймворку).

---

## Етап 1: Дані та інжест

1. Покласти PDF у **`data/`** (як у README).
2. Реалізувати **спільну логіку**, якщо можливо: шлях до `data/`, шлях до `index/`, розмір чанку, overlap — у **`config.py`** (константи / `Settings`).
3. **`ingest_llama_flavour.py`**: завантаження PDF → розбиття на чанки → ембеддинги → запис у Chroma у `index/`.
4. **`ingest_langchain_flavour.py`**: той самий результат (той самий Chroma collection / сумісний формат), але через стек LangChain.
5. Переконатися, що обидва пайплайни зберігають **текст чанків**, придатний і для **векторного**, і для **лексичного** пошуку (BM25 будується по тих самих чанках).

**Критерій готовності:** `make ingest-llama` та `make ingest-langchain` успішно заповнюють `index/` без ручних кроків.

---

## Етап 2: Гібридний retrieval + reranking

Реалізувати **однакову триетапну схему** в обох flavour (як у README):

1. **Семантичний пошук** — косинусна схожість по ембеддингах у Chroma (`text-embedding-3-large`).
2. **BM25** — лексичний пошук по тих самих чанках.
3. **Reranking** — cross-encoder `BAAI/bge-reranker-base` для пересортування та відсікання кандидатів після злиття результатів.

**LlamaIndex:** `QueryFusionRetriever` + `SentenceTransformerRerank`.  
**LangChain:** `EnsembleRetriever` + cross-encoder reranker.

**Критерій готовності:** для тестового запиту обидва retriever повертають узгоджений за змістом топ-k фрагментів (формат зручний для підстановки в промпт / tool output).

---

## Етап 3: Інструмент `knowledge_search`

1. У **`tools.py`** додати tool **`knowledge_search`** (або з параметром вибору backend, або два окремі внутрішні шляхи — за узгодженістю з курсом).
2. Tool викликає обраний retriever (LlamaIndex або LangChain — залежно від конфігурації) і повертає **стислий текст** (з обрізанням, як у уроці 3 — context engineering).
3. Додати **`knowledge_search`** до списку **`TOOLS`** у `agent.py` / `tools.py`.

---

## Етап 4: Промпти та поведінка агента

1. Винести системний промпт у **`system_prompt.yaml`**.
2. У **`config.py`** завантажувати YAML і формувати `SYSTEM_PROMPT`.
3. Оновити інструкції для моделі:
   - коли варто викликати **`web_search` / `read_url`** (актуальні джерела, зовнішній веб);
   - коли варто викликати **`knowledge_search`** (факти з наданих PDF, внутрішня база);
   - зберегти вимоги уроку 3: багатокроковість, `write_report`, чесність щодо невизначеності, мова відповіді.

---

## Етап 5: Точка входу та зручність запуску

1. Оновити **`main.py`**: заголовок / підказки для користувача (наприклад, «RAG Agent»), опційно прапорець **`--debug`** для детального логування викликів tools (відповідає `make run-rag-debug`).
2. Додати **`Makefile`** з цілями з README: `ingest-llama`, `ingest-langchain`, `run-rag`, `run-rag-debug`, `test_lesson_5`, `help`.
3. За потреби додати **`static_files/`** з демо-GIF або посиланнями (як у README — не обов’язково для автоперевірки, але для здачі курсу може бути корисно).

---

## Етап 6: Тести та перевірка якості

1. Написати **unit-тести** для: інжесту (моки файлової системи / Chroma, де доречно), retriever (фікстури чанків), `knowledge_search` (мок retriever), граничних випадків (порожній індекс, дуже довгий результат → обрізання).
2. Запуск: **`make test_lesson_5`**.
3. Ручна перевірка: після інжесту — **`make run-rag`**, запитання, які **відповідають лише з PDF**, та запитання, які **потребують вебу** — переконатися, що агент обирає правильний інструмент.

---

## Етап 7: Документація (мінімум)

1. Переконатися, що **`README.md`** (англійською, як зараз) залишається узгодженим з фактичними командами та файлами.
2. За бажанням додати короткий розділ українською в README або посилання на цей **`PLAN_UA.md`** для одногрупників.

---

## Ризики та залежності

- **Розмір моделей reranker** — перший запуск може завантажити ваги; варто закласти це в README або кеш.
- **Сумісність версій** LangChain / LlamaIndex / Chroma — зафіксувати у lock-файлі.
- **Шляхи** — усі шляхи до `data/`, `index/`, `output/` мають бути відносними до `BASE_DIR`, як у поточному `config.py`.

---

## Підсумок порядку робіт

1. Залежності та `.env` → 2. Інжест (Llama + LangChain) → 3. Retriever з fusion + rerank → 4. `knowledge_search` + оновлення `TOOLS` → 5. `system_prompt.yaml` + промпт → 6. `Makefile` та CLI/debug → 7. Тести та ручна перевірка сценаріїв.

Після цього проєкт відповідає опису **Homework Lesson 5 — RAG Agent** у [`README.md`](./README.md).
