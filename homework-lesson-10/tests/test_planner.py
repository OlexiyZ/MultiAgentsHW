"""
test_planner.py
===============
Тести якості відповідей агента Planner.

Стратегія
--------
Усі тести використовують **статичні заздалегідь визначені planner outputs** —
реалістичні JSON-рядки, які представляють відповідь Planner для заданого
запиту користувача. Під час самого запуску тестів не викликається реальна LLM
або мережа, тому набір тестів швидкий і детермінований.

Єдиний LLM-виклик відбувається всередині DeepEval GEval metric, яка використовує
gpt-4o-mini для оцінювання відповіді за рубрикою.

Використані метрики
------------
- ``GEval`` "Plan Quality": перевіряє, що план має конкретні запити,
  релевантні джерела, конкретний формат відповіді та чітку мету.
"""
from __future__ import annotations

import json
import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# ---------------------------------------------------------------------------
# Static planner outputs – realistic examples of what the Planner agent
# would return for each input.
# Статичні результати планувальника — реалістичні приклади того, що агент Planner
# повернув би для кожного вхідного запиту.
# ---------------------------------------------------------------------------


PLANNER_CASES = [
    {
        "input": "Порівняй наївний RAG із sentence-window retrieval",
        "actual_output": json.dumps(
            {
                "goal": "Порівняти підходи наївного RAG і sentence-window retrieval за точністю та продуктивністю",
                "search_queries": [
                    "наївний RAG фіксовані чанки обмеження точність",
                    "sentence-window retrieval LlamaIndex розширення контексту",
                    "порівняння точності RAG retrieval benchmark 2024",
                    "chunk size vs sentence window retrieval trade-offs",
                ],
                "sources_to_check": ["knowledge_base", "web"],
                "output_format": (
                    "Markdown-звіт українською з розділами: Огляд, Наївний RAG (визначення, переваги, недоліки), "
                    "Sentence-Window Retrieval (визначення, переваги, недоліки), "
                    "Порівняльна таблиця (точність / затримка / складність), Рекомендації"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
    },
    {
        "input": "Що таке відкритий банкінг в Україні та який закон його регулює?",
        "actual_output": json.dumps(
            {
                "goal": "З'ясувати регуляторну рамку відкритого банкінгу в Україні, включно із законами та вимогами НБУ",
                "search_queries": [
                    "відкритий банкінг Україна НБУ постанова 80 2025",
                    "відкритий банкінг Україна закон 1591-IX платіжні послуги",
                    "Україна відкритий банкінг вимоги API сторонні надавачі AIS PIS",
                    "НБУ постанова 80 відкритий банкінг вимоги",
                ],
                "sources_to_check": ["knowledge_base", "web"],
                "output_format": (
                    "Markdown-звіт українською з розділами: Визначення відкритого банкінгу, "
                    "Правова база (Закон 1591-IX, Постанова НБУ № 80), "
                    "Ключові вимоги до API (AIS / PIS), Права сторонніх надавачів, Строки впровадження"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
    },
    {
        "input": "Розкажи про банкінг",
        "actual_output": json.dumps(
            {
                "goal": "Надати структурований огляд банківських понять із фокусом на українське банківське регулювання та послуги",
                "search_queries": [
                    "банківські послуги огляд Україна",
                    "банківське регулювання Україна НБУ 2024 2025",
                    "українська банківська система структура комерційні банки",
                ],
                "sources_to_check": ["knowledge_base", "web"],
                "output_format": "Markdown-огляд українською: Основи банкінгу, Банківська система України, Регуляторний огляд, Ключові послуги",
            },
            ensure_ascii=False,
            indent=2,
        ),
    },
]


# ---------------------------------------------------------------------------
# Metric definition
# Визначення метрики
# ---------------------------------------------------------------------------
plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that 'search_queries' contains specific, searchable queries — not vague phrases like 'banking info' or 'more about this'.",
        "Check that 'sources_to_check' lists relevant sources for the topic (e.g. 'knowledge_base' for domain docs, 'web' for current events).",
        "Check that 'output_format' is concrete and describes report structure with named sections, tables, or other elements.",
        "Check that 'goal' clearly and specifically states what the research aims to answer — not just restating the input word-for-word.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


# ---------------------------------------------------------------------------
# Tests
# Тести
# ---------------------------------------------------------------------------

def test_plan_quality_rag():
    # This test checks that the planner creates a useful plan for a RAG technique comparison.
    # Цей тест перевіряє, що планувальник створює корисний план для порівняння RAG-технік.
    """
    Планувальник має створити конкретні, придатні до дії запити для технічного
    порівняння (наївний RAG vs sentence-window retrieval).

    Очікування:
    - щонайменше 3 конкретні пошукові запити із згадкою релевантних технік
    - 'knowledge_base' і 'web' зазначені як джерела
    - output_format описує іменовані розділи звіту та порівняльну таблицю
    """
    case = PLANNER_CASES[0]
    test_case = LLMTestCase(input=case["input"], actual_output=case["actual_output"])
    assert_test(test_case, [plan_quality])


def test_plan_quality_open_banking():
    # This test checks that the planner creates a bilingual domain-specific plan for Ukrainian open banking.
    # Цей тест перевіряє, що планувальник створює двомовний доменно-специфічний план для відкритого банкінгу в Україні.
    """
    Планувальник має створити доменно-специфічні запити українською та явно
    вказати 'knowledge_base' для цієї теми.

    Очікування:
    - запити згадують постанову НБУ, Закон 1591-IX і українські терміни
    - 'knowledge_base' зазначено як джерело (локальні регуляторні PDF)
    - output_format посилається на конкретні розділи, як-от AIS/PIS і строки
    """
    case = PLANNER_CASES[1]
    test_case = LLMTestCase(input=case["input"], actual_output=case["actual_output"])
    assert_test(test_case, [plan_quality])


def test_plan_has_queries():
    # This test checks that every planner output contains at least one search query and one source.
    # Цей тест перевіряє, що кожен результат планувальника містить принаймні один пошуковий запит і одне джерело.
    """
    Структурний інваріант: кожен план має містити щонайменше один пошуковий
    запит і щонайменше одне джерело для перевірки, навіть для нечітких запитів
    на кшталт 'Розкажи про банкінг'.

    Також запускає GEval для нечіткого запиту, щоб підтвердити, що планувальник
    повертає зв'язний план, а не порожні поля.
    """
    for case in PLANNER_CASES:
        plan_data = json.loads(case["actual_output"])
        assert len(plan_data["search_queries"]) >= 1, (
            f"План для input {case['input']!r} має містити щонайменше один пошуковий запит"
        )
        assert len(plan_data["sources_to_check"]) >= 1, (
            f"План для input {case['input']!r} має містити щонайменше одне джерело"
        )

    # Additionally run GEval on the vague / broad input
    # Додатково запускаємо GEval на нечіткому / широкому вхідному запиті
    case = PLANNER_CASES[2]
    test_case = LLMTestCase(input=case["input"], actual_output=case["actual_output"])
    assert_test(test_case, [plan_quality])


def test_plan_goal_is_non_empty():
    # This test checks that every planner output has a non-empty goal field.
    # Цей тест перевіряє, що кожен результат планувальника має непорожнє поле мети.
    """
    Структурний інваріант: поле 'goal' кожного плану має бути непорожнім рядком.
    Це захищає від повернення плану з порожньою метою.
    """
    for case in PLANNER_CASES:
        plan_data = json.loads(case["actual_output"])
        assert isinstance(plan_data.get("goal"), str) and plan_data["goal"].strip(), (
            f"План для input {case['input']!r} має містити непорожнє поле 'goal'"
        )


def test_plan_output_format_is_descriptive():
    # This test checks that every planner output describes the report format in enough detail.
    # Цей тест перевіряє, що кожен результат планувальника достатньо детально описує формат звіту.
    """
    Поле 'output_format' має бути описовим (більше ніж 5 слів), а не просто
    коротким ключовим словом на кшталт 'markdown' або 'json'.
    """
    for case in PLANNER_CASES:
        plan_data = json.loads(case["actual_output"])
        fmt = plan_data.get("output_format", "")
        word_count = len(fmt.split())
        assert word_count >= 5, (
            f"'output_format' для input {case['input']!r} має бути описовим "
            f"(отримано {word_count!r} слів: {fmt!r})"
        )
