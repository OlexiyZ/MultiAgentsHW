"""Specialized agent definitions for multi-agent LangGraph workflow.
Визначення спеціалізованих агентів для мульти-агентного LangGraph workflow."""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from tools import web_search, read_url, knowledge_search, write_report


settings = Settings()


def create_llm():
    """Create and return configured LLM instance.
    Створює та повертає налаштований екземпляр LLM."""
    return ChatOpenAI(
        api_key=settings.api_key.get_secret_value(),
        model=settings.model_name,
        timeout=settings.request_timeout,
    )


# Supervisor Agent - координує роботу інших агентів
SUPERVISOR_PROMPT = """Ти супервайзер, який координує команду спеціалізованих агентів.
Твоя команда:
- researcher: Шукає інформацію в Інтернеті та читає веб-сторінки
- knowledge_expert: Працює з локальною базою знань (PDF документи)
- report_writer: Створює структуровані звіти у форматі Markdown

Аналізуй запит користувача та вирішуй, який агент або агенти мають працювати над завданням.
Відповідай у форматі JSON:
{
  "next_agent": "researcher" | "knowledge_expert" | "report_writer" | "FINISH",
  "task": "конкретне завдання для обраного агента"
}

Якщо завдання виконано, використовуй "next_agent": "FINISH"."""


# Research Agent - пошук в Інтернеті
RESEARCHER_PROMPT = """Ти агент-дослідник, який спеціалізується на пошуку інформації в Інтернеті.
Твої можливості:
- Пошук інформації через DuckDuckGo (web_search)
- Читання та витягування тексту з веб-сторінок (read_url)

Твої завдання:
1. Шукати актуальну та достовірну інформацію в Інтернеті
2. Читати та аналізувати веб-сторінки
3. Надавати стислі та структуровані результати

Працюй ефективно та надавай лише релевантну інформацію."""


# Knowledge Expert Agent - робота з базою знань
KNOWLEDGE_EXPERT_PROMPT = """Ти експерт з бази знань, який спеціалізується на роботі з локальними PDF документами.
Твої можливості:
- Пошук у локальній базі знань (knowledge_search)
- Гібридний пошук (векторний + BM25 + reranking)

Твої завдання:
1. Шукати релевантну інформацію в проіндексованих PDF документах
2. Надавати точні цитати та посилання на джерела
3. Працювати з юридичними та регуляторними документами

Контекст: База знань містить документи про відкритий банкінг в Україні, платіжні послуги та регуляції.

Надавай детальні та точні відповіді на основі бази знань."""


# Report Writer Agent - створення звітів
REPORT_WRITER_PROMPT = """Ти агент-письменник, який спеціалізується на створенні структурованих звітів.
Твої можливості:
- Створення Markdown звітів (write_report)

Твої завдання:
1. Аналізувати інформацію від інших агентів
2. Створювати структуровані, добре оформлені Markdown звіти
3. Включати всі релевантні деталі, посилання та джерела

Структура звіту:
- Заголовок та короткий опис
- Основні розділи з підзаголовками
- Чіткі висновки
- Список джерел (якщо є)

Використовуй правильне Markdown форматування."""


class AgentNode:
    """Base class for specialized agent nodes in the graph.
    Базовий клас для вузлів спеціалізованих агентів у графі."""

    def __init__(self, name: str, system_prompt: str, tools: list):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.llm = create_llm()
        if tools:
            self.llm = self.llm.bind_tools(tools)

    def __call__(self, state: dict) -> dict:
        """Execute the agent with the current state.
        Виконує агента з поточним станом."""
        messages = state.get("messages", [])

        # Add system prompt
        agent_messages = [
            SystemMessage(content=self.system_prompt),
            *messages
        ]

        # Invoke LLM
        response = self.llm.invoke(agent_messages)

        # Return updated state
        return {
            "messages": messages + [response],
            "sender": self.name
        }


# Create specialized agent instances
researcher_node = AgentNode(
    name="researcher",
    system_prompt=RESEARCHER_PROMPT,
    tools=[web_search, read_url]
)

knowledge_expert_node = AgentNode(
    name="knowledge_expert",
    system_prompt=KNOWLEDGE_EXPERT_PROMPT,
    tools=[knowledge_search]
)

report_writer_node = AgentNode(
    name="report_writer",
    system_prompt=REPORT_WRITER_PROMPT,
    tools=[write_report]
)
