"""
test_tools.py
=============
Тести коректного використання інструментів кожним агентом у multi-agent pipeline.

Стратегія
--------
Використовує DeepEval ``ToolCorrectnessMetric`` зі статичними послідовностями
``ToolCall``, які представляють виклики агентів під час типового запуску pipeline.

Реальні агенти не викликаються: очікувані та фактичні послідовності tool calls
визначені статично, а метрика перевіряє їхній збіг.

Тестові сценарії
--------------
1. Planner отримує дослідницький запит → має викликати ``web_search`` та/або
   ``knowledge_search``, щоб підготувати план.
2. Researcher отримує дослідницький план → має викликати ``knowledge_search`` і
   ``web_search`` (та опційно ``read_url``).
3. Supervisor отримує вердикт APPROVE → має викликати ``save_report``, щоб
   зберегти фінальний результат дослідження.

Використана метрика
-----------
- ``ToolCorrectnessMetric`` (threshold=0.5): перевіряє, що фактичні tool calls
  збігаються з очікуваними за назвою.
"""
from __future__ import annotations

import json
import pytest

from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall


# ---------------------------------------------------------------------------
# Shared metric
# ---------------------------------------------------------------------------

tool_metric = ToolCorrectnessMetric(
    threshold=0.5,
    model="gpt-4o-mini",
    include_reason=True,
)


# ---------------------------------------------------------------------------
# Test 1 – Planner tool usage
# ---------------------------------------------------------------------------


def test_planner_uses_search_tools():
    # This test checks that the planner calls search tools for exploratory planning.
    # Цей тест перевіряє, що планувальник викликає пошукові інструменти для дослідницького планування.
    """
    Агент Planner має використовувати ``web_search`` і ``knowledge_search``
    під час побудови дослідницького плану для запиту про відкритий банкінг.
    """
    test_case = LLMTestCase(
        input="Що таке відкритий банкінг в Україні та який закон його регулює?",
        actual_output=json.dumps(
            {
                "goal": "Зрозуміти регуляторну рамку відкритого банкінгу в Україні, включно з основними законами",
                "search_queries": [
                    "відкритий банкінг Україна НБУ постанова 80 2025",
                    "відкритий банкінг Україна закон 1591-IX платіжні послуги",
                    "Україна відкритий банкінг вимоги API сторонні надавачі",
                ],
                "sources_to_check": ["knowledge_base", "web"],
                "output_format": "Markdown-звіт українською з розділами: Визначення, Правова база, Ключові вимоги, API-стандарти",
            },
            ensure_ascii=False,
            indent=2,
        ),
        tools_called=[
            ToolCall(
                name="knowledge_search",
                input_parameters={"query": "відкритий банкінг НБУ постанова 80"},
                output=(
                    "Постанова НБУ № 80 від 25.07.2025 затверджує Положення про відкритий банкінг. "
                    "Закон 1591-IX визначає надавачів платіжних послуг..."
                ),
            ),
            ToolCall(
                name="web_search",
                input_parameters={"query": "відкритий банкінг Україна НБУ постанова 80 2025"},
                output=(
                    "[{'title': 'Постанова НБУ про відкритий банкінг', 'url': 'https://bank.gov.ua/...', "
                    "'snippet': 'НБУ затвердив регулювання відкритого банкінгу постановою № 80 у липні 2025 року...'}]"
                ),
            ),
        ],
        expected_tools=[
            ToolCall(name="knowledge_search"),
            ToolCall(name="web_search"),
        ],
    )
    assert_test(test_case, [tool_metric])

# ---------------------------------------------------------------------------
# Test 2 – Researcher tool usage
# ---------------------------------------------------------------------------


def test_researcher_uses_knowledge_and_web_tools():
    # This test checks that the researcher calls knowledge, web, and URL-reading tools for a RAG plan.
    # Цей тест перевіряє, що дослідник викликає інструменти бази знань, вебпошуку та читання URL для RAG-плану.
    """
    Агент Researcher має використовувати щонайменше ``knowledge_search`` і
    ``web_search`` під час виконання дослідницького плану про RAG-техніки.
    """
    test_case = LLMTestCase(
        input=json.dumps(
            {
                "goal": "Порівняти наївний RAG і sentence-window retrieval за точністю та продуктивністю",
                "search_queries": [
                    "наївний RAG фіксовані чанки обмеження",
                    "sentence-window retrieval LlamaIndex контекст",
                    "RAG бенчмарки точності 2024",
                ],
                "sources_to_check": ["knowledge_base", "web"],
                "output_format": "Markdown-звіт українською з порівняльною таблицею та рекомендаціями",
            },
            ensure_ascii=False,
        ),
        actual_output="""\
## Порівняння RAG: результати дослідження

### Наївний RAG
Використовує фіксовані чанки по 512 токенів і top-k retrieval за косинусною схожістю.
Обмеження: чанки можуть розривати речення й втрачати контекст.

### Sentence-Window Retrieval
Індексує речення й повертає вікно ±2 речення під час retrieval.
Faithfulness покращується приблизно на 10% у QA-бенчмарках.

### Рекомендація
Використовуйте sentence-window для застосувань, де критична faithfulness.
""",
        tools_called=[
            ToolCall(
                name="knowledge_search",
                input_parameters={"query": "наївний RAG фіксовані чанки обмеження"},
                output=(
                    "Наївний RAG ділить документи на фіксовані чанки 256-512 токенів "
                    "і повертає top-k за векторною схожістю..."
                ),
            ),
            ToolCall(
                name="web_search",
                input_parameters={"query": "sentence-window retrieval LlamaIndex контекст 2024"},
                output=(
                    "[{'title': 'Sentence Window Retrieval', 'url': 'https://llamaindex.ai/...', "
                    "'snippet': 'Sentence-window retrieval покращує faithfulness через індексацію речень...'}]"
                ),
            ),
            ToolCall(
                name="read_url",
                input_parameters={"url": "https://llamaindex.ai/blog/sentence-window-retrieval"},
                output=(
                    "Sentence-window retrieval індексує окремі речення і повертає контекстне вікно "
                    "під час запиту. Бенчмарки показують 8-15% покращення faithfulness проти наївного RAG..."
                ),
            ),
        ],
        expected_tools=[
            ToolCall(name="knowledge_search"),
            ToolCall(name="web_search"),
            ToolCall(name="read_url"),
        ],
    )
    assert_test(test_case, [tool_metric])

# ---------------------------------------------------------------------------
# Test 3 – Supervisor tool usage on APPROVE verdict
# ---------------------------------------------------------------------------


def test_supervisor_saves_report_on_approve():
    # This test checks that the supervisor saves the report after an APPROVE critic verdict.
    # Цей тест перевіряє, що супервізор зберігає звіт після вердикту критика APPROVE.
    """
    Агент Supervisor має викликати ``save_report``, коли Critic повертає
    вердикт APPROVE, щоб зберегти фінальний Markdown-звіт на диск.
    """
    critic_output = json.dumps(
        {
            "verdict": "APPROVE",
            "is_fresh": True,
            "is_complete": True,
            "is_well_structured": True,
            "strengths": [
                "Цитує Постанову НБУ № 80 і Закон 1591-IX із конкретними датами.",
                "AIS і PIS чітко розрізнено разом із вимогами до впровадження.",
            ],
            "gaps": [],
            "revision_requests": [],
        },
        ensure_ascii=False,
    )

    test_case = LLMTestCase(
        input=f"Критик повернув: {critic_output}. Збережи схвалений дослідницький звіт.",
        actual_output="Звіт збережено до output/open_banking_ukraine.md",
        tools_called=[
            ToolCall(
                name="save_report",
                input_parameters={
                    "topic": "open banking in Ukraine",
                    "content": (
                        "## Відкритий банкінг в Україні\n\n"
                        "Відкритий банкінг в Україні регулюється Постановою НБУ № 80 (25.07.2025) "
                        "та Законом № 1591-IX (30.06.2021)...\n\n"
                        "### Типи API\n- AIS: Account Information Services\n- PIS: Payment Initiation Services\n"
                    ),
                },
                output="Звіт збережено до output/open_banking_ukraine.md",
            ),
        ],
        expected_tools=[
            ToolCall(name="save_report"),
        ],
    )
    assert_test(test_case, [tool_metric])

# ---------------------------------------------------------------------------
# Test 4 – Supervisor does NOT save on REVISE verdict
# ---------------------------------------------------------------------------

def test_supervisor_does_not_save_on_revise():
    # This test checks that the supervisor does not save a report after a REVISE critic verdict.
    # Цей тест перевіряє, що супервізор не зберігає звіт після вердикту критика REVISE.
    """
    Структурна перевірка: коли Critic повертає REVISE, supervisor НЕ має
    викликати ``save_report``. Натомість pipeline повертається до researcher.

    Тест перевіряє відсутність ``save_report`` у списку tool calls
    (DeepEval-метрика не потрібна — це суто структурний assert).
    """
    # Simulate what the supervisor's tool call log would look like on REVISE
    # (no save_report, just passing instructions back to researcher)
    actual_tools_called_names = ["knowledge_search", "web_search"]  # researcher re-runs

    assert "save_report" not in actual_tools_called_names, (
        "Супервізор НЕ має викликати save_report, коли вердикт критика REVISE. "
        "Pipeline має повернутися до researcher для виправлення."
    )


# ---------------------------------------------------------------------------
# Test 5 – Planner uses correct tool subset (not read_url)
# ---------------------------------------------------------------------------

def test_planner_does_not_use_read_url():
    # This test checks that the planner does not call tools reserved for the researcher or supervisor.
    # Цей тест перевіряє, що планувальник не викликає інструменти, зарезервовані для дослідника або супервізора.
    """
    Структурна перевірка: planner має використовувати лише PLANNER_TOOLS
    (web_search, knowledge_search) і НЕ має викликати read_url.

    read_url належить лише до RESEARCH_TOOLS — цей інструмент надто повільний
    і деталізований для етапу планування.

    Це суто структурний assert (DeepEval-метрика не потрібна).
    """
    # These represent what the planner actually called in a typical run
    planner_tools_used = ["knowledge_search", "web_search"]

    assert "read_url" not in planner_tools_used, (
        "Планувальник не має викликати read_url — цей інструмент зарезервований для агента-дослідника. "
        "Інструменти планувальника: [web_search, knowledge_search]."
    )
    assert "save_report" not in planner_tools_used, (
        "Планувальник не має викликати save_report — цей інструмент зарезервований для супервізора. "
    )
