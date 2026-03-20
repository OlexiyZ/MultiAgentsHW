# Покроковий план виконання: Research Agent

Цей документ описує порядок реалізації завдання з README.md.

---

## Передумови

- Python 3.x
- Файл `.env` створений з `.env.example` і заповнений `OPENAI_API_KEY` та `MODEL_NAME` (наприклад, `gpt-4o-mini`)
- Встановлені залежності: `pip install -r requirements.txt`

---

## Крок 1. Інструменти (tools) — файл `tools.py`

### 1.1. `web_search(query: str) -> list[dict]`

1. Додати імпорт: `from ddgs import DDGS`.
2. Реалізувати функцію-обгортку над `DDGS().text(query, max_results=...)`.
3. Нормалізувати результат у формат для LLM: список словників з ключами `title`, `url`, `snippet` (відповідно до `title`, `href`, `body` з DDGS).
4. Додати **обрізання**: якщо сумарний текст результатів завеликий — обрізати до N символів (наприклад, використати `max_url_content_length` з config або окремий ліміт для пошуку).
5. Оформити як LangChain tool з `@tool` та JSON Schema (description, parameters). Параметр `max_results` — зафіксувати (наприклад, 5) або додати в схему.

### 1.2. `read_url(url: str) -> str`

1. Додати імпорти: `trafilatura` (рекомендовано), опційно `httpx` для таймаутів.
2. Реалізувати: `trafilatura.fetch_url(url)` → `trafilatura.extract(downloaded)`.
3. **Обрізання**: обрізати текст до ліміту з config (наприклад, 5000–10000 символів) перед поверненням.
4. **Обробка помилок**: невалідний URL, таймаут, 404 — повертати зрозуміле текстове повідомлення про помилку, не падати.
5. Оформити як `@tool` з описом та параметром `url`.

### 1.3. `write_report(filename: str, content: str) -> str`

1. Визначити директорію збереження з config (`output_dir`, за замовчуванням `output/`).
2. Збирати повний шлях: `output_dir / filename` (або `os.path.join`), створювати директорію за потреби (`os.makedirs(..., exist_ok=True)`).
3. Записати `content` у файл (текстовий режим, UTF-8).
4. Повернути підтвердження з повним шляхом до файлу.
5. Оформити як `@tool` з description та параметрами `filename`, `content`.

### 1.4. Експорт та використання config у tools

- Імпортувати з `config.py` потрібні константи: `max_search_results`, `max_url_content_length`, `output_dir`.
- Переконатися, що в `config.py` ці значення беруться з `Settings` (вже є в прикладі).

---

## Крок 2. Конфігурація та промпти — файл `config.py`

1. Доповнити `Settings`: переконатися, що є `api_key`, `model_name`, `max_search_results`, `max_url_content_length`, `output_dir`, `max_iterations` (вже частково є).
2. Заповнити **SYSTEM_PROMPT**:
   - Роль: research-агент, який досліджує питання користувача.
   - Опис доступних інструментів: `web_search`, `read_url`, `write_report` — коли і для чого використовувати.
   - Стратегія: спочатку пошук, потім при потребі читання URL для деталей, в кінці — структурований Markdown-звіт та збереження через `write_report`.
   - Вимога до формату відповіді: структурований Markdown (заголовки, списки, порівняння тощо).

---

## Крок 3. Агент (LangChain) — файл `agent.py`

1. **LLM**: створити екземпляр моделі (наприклад, `ChatOpenAI`) з параметрами з config: `api_key`, `model_name`. Використовувати `model_config` з `config.Settings()`.
2. **Tools**: імпортувати з `tools.py` готові функції, оформлені через `@tool`, у список `tools` для агента.
3. **Prompt**: використати шаблон для ReAct-агента (LangChain надає `create_react_agent` і відповідний prompt). Підставити в промпт `SYSTEM_PROMPT` з config.
4. **Memory**: підключити `MemorySaver` (checkpointer) для збереження історії діалогу в межах сесії — щоб підтримувати зв’язний діалог ("а тепер порівняй з X").
5. **Створення агента**: викликати `create_react_agent(model, tools, prompt)` (або еквівалент у вашій версії LangChain).
6. **Compile**: зібрати агента з checkpointer та `max_iterations` з config. Переконатися, що при інвокації передається `config` з `thread_id` для memory.
7. Експортувати змінну `agent` (або callable), яку використовує `main.py`.

Документація LangChain:
- [create_react_agent](https://python.langchain.com/docs/how_to/react/)
- [MemorySaver / checkpointer](https://python.langchain.com/docs/how_to/adding_message_memory/)

---

## Крок 4. Інтеграція з main.py

1. Переконатися, що при виклику агента передається **thread_id** для checkpointer (наприклад, фіксований `"default"` або один на сесію), щоб пам’ять працювала.
2. Передавати в агент повідомлення у форматі, який очікує LangChain (наприклад, `{"messages": [HumanMessage(content=user_input)]}` з підтримкою історії).
3. Обробляти потік відповіді так, щоб користувач бачив фінальну відповідь; при бажанні виводити проміжні думки (thought) або виклики tools для прозорості.
4. Обробляти помилки: якщо агент повертає помилку або tool fail — не падати, показати повідомлення користувачу.

Приклад передачі thread_id при stream/invoke:
```python
config = {"configurable": {"thread_id": "default"}}
agent.stream({"messages": [("user", user_input)]}, config=config)
```

---

## Крок 5. Context engineering (перевірка)

1. У `web_search`: обрізання сумарного тексту результатів до обмеження (наприклад, 8000 символів) перед поверненням у контекст.
2. У `read_url`: обрізання до `max_url_content_length` (наприклад, 5000–10000 символів).
3. У config мають бути явні константи/налаштування для цих лімітів.

---

## Крок 6. Залежності та середовище

1. Оновити `requirements.txt`: вказати мінімальні версії `langchain>=1.2.0`, `ddgs>=7.0`, `trafilatura>=2.0.0`, `pydantic`, `pydantic-settings`; додати `langchain-openai` (або інший пакет провайдера) та `httpx` якщо використовується.
2. У `.env.example` додати приклад `MODEL_NAME=...` (якщо використовується в config).

---

## Крок 7. Документація та приклад

1. **README.md**: переконатися, що описано:
   - як запустити: `python main.py`;
   - які залежності: `pip install -r requirements.txt`;
   - який API-ключ потрібен і де його вказати (`.env`);
   - короткий опис архітектури: main → agent (LLM + tools + memory), tools (web_search, read_url, write_report), config.
2. **Приклад роботи**: зберегти один згенерований звіт у `example_output/report.md` (вже є приклад структури — можна замінити реальним виходом агента).

---

## Крок 8. Фінальна перевірка

- [ ] Запуск з терміналу: `python main.py` — інтерактивний ввід, вихід по "exit"/"quit".
- [ ] Мінімум 3–5 викликів tools на одне складне питання (multi-step).
- [ ] Зв’язний діалог: друге повідомлення типу "а тепер порівняй з X" враховує контекст.
- [ ] Ліміт кроків (max_iterations) налаштований, агент не зациклюється.
- [ ] Помилки tools (наприклад, невалідний URL) повертаються в контекст, агент реагує адекватно.
- [ ] System prompt та налаштування в `config.py`, не захардкоджені в agent/main.
- [ ] Секрети тільки в `.env`, не в репозиторії.

---

## Порядок реалізації (коротко)

1. **config.py** — доповнити Settings, написати SYSTEM_PROMPT.
2. **tools.py** — реалізувати `web_search`, `read_url`, `write_report` з обрізанням та обробкою помилок, оформити як `@tool`.
3. **agent.py** — LLM, список tools, prompt, MemorySaver, create_react_agent, compile з max_iterations та checkpointer.
4. **main.py** — передача thread_id у config при виклику агента, обробка потокового виводу та помилок.
5. **requirements.txt** та **.env.example** — оновити.
6. **README.md** та **example_output** — перевірити/оновити.
7. Запустити, протестувати, виправити помилки.

Після виконання цих кроків завдання буде виконано відповідно до README.md.
