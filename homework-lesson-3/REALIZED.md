## Реалізовано

### Tools

1. `web_search(query, max_results)`
   Пошук через `ddgs` з поверненням компактного списку `title`, `url`, `snippet`.

2. `read_url(url)`
   Завантажує сторінку через `trafilatura`, витягує основний текст і обрізає його до `MAX_URL_CONTENT_LENGTH`.

3. `write_report(filename, content)`
   Зберігає Markdown-звіт у директорію `output/` і повертає повний шлях до файла.

### Agent loop

- Використано `create_react_agent`
- Модель підключається через `ChatOpenAI`
- Пам'ять сесії реалізована через `MemorySaver`
- Ліміт кроків задається через `MAX_ITERATIONS`

### Конфігурація

Параметри винесені в `config.py` і читаються з `.env` через `pydantic-settings`.

Підтримувані змінні:

```env
OPENAI_API_KEY=...
MODEL_NAME=gpt-4o-mini
MAX_SEARCH_RESULTS=5
MAX_URL_CONTENT_LENGTH=5000
MAX_ITERATIONS=8
OUTPUT_DIR=output
REQUEST_TIMEOUT=20
```

## Запуск

1. Створіть віртуальне оточення.
2. Встановіть залежності:

```bash
pip install -r requirements.txt
```

3. Створіть `.env` на основі шаблону:

```bash
cp .env.example .env
```

Для Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

4. Заповніть `OPENAI_API_KEY`.
5. Запустіть агента:

```bash
python main.py
```

## Приклад взаємодії

```text
You: Compare naive RAG, sentence-window retrieval, and parent-child retrieval.
Agent: ...структурована відповідь з посиланнями та trade-offs...
```

Якщо попросити агента зберегти результат, він може викликати `write_report` і створити файл в `output/`.

## Архітектурні рішення

- `config.py` містить settings і system prompt
- `tools.py` інкапсулює всю зовнішню взаємодію
- `agent.py` збирає LLM, tools і memory
- `main.py` відповідає лише за CLI-цикл

## Обмеження

- Для `web_search` і `read_url` потрібен мережевий доступ під час виконання програми.
- Якість результату залежить від доступності зовнішніх сторінок і моделі.
- Поточна реалізація орієнтована на OpenAI-сумісну модель через `langchain-openai`.
