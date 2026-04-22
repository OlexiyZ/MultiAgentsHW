"""
test_critic.py
==============
Тести якості відповідей агента Critic.

Стратегія
--------
Усі тести використовують **статичні заздалегідь визначені CritiqueResult JSON** —
реалістичні приклади того, що агент Critic повернув би під час оцінювання
результатів дослідження.

Перевіряються два основні сценарії:

1. **APPROVE path** — критика правильно визначає якісний результат дослідження,
   перелічує конкретні сильні сторони, не має суттєвих прогалин і не створює
   запитів на виправлення.

2. **REVISE path** — критика правильно визначає неповний результат дослідження,
   перелічує конкретні прогалини та надає дієві, конкретні запити на виправлення
   (а не нечіткі інструкції на кшталт "покращити це").

Використані метрики
------------
- ``GEval`` "Critique Quality": перевіряє, що критика APPROVE є позитивною та
  конкретною, а критика REVISE надає дієві рекомендації.
"""
from __future__ import annotations

import json
import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# ---------------------------------------------------------------------------
# Static critic outputs
# Статичні результати критика
# ---------------------------------------------------------------------------

# -- APPROVE case: well-researched open banking report ----------------------
# -- APPROVE кейс: добре досліджений звіт про відкритий банкінг ------------


APPROVE_FINDINGS_INPUT = """\
## Відкритий банкінг в Україні: результати дослідження

**Регуляторна база**
Відкритий банкінг в Україні регулюється Постановою НБУ № 80 від 25.07.2025
та Законом № 1591-IX від 30.06.2021 "Про платіжні послуги".

**Типи API**
Банки мають надати API для:
- сервісів інформації про рахунок (AIS)
- сервісів ініціювання платежів (PIS)

**Вимоги до сторонніх надавачів послуг**
TPP мають: (1) отримати ліцензію НБУ, (2) зареєструватися в реєстрі НБУ,
(3) використовувати OAuth 2.0 / OpenID Connect для автентифікації.

**Строки впровадження**
Банки мають до Q1 2026 надати production-grade API відкритого банкінгу.

*Джерела: Постанова НБУ № 80, Закон 1591-IX, офіційні роз'яснення НБУ*
"""

APPROVE_CRITIQUE_OUTPUT = json.dumps(
    {
        "verdict": "APPROVE",
        "is_fresh": True,
        "is_complete": True,
        "is_well_structured": True,
        "strengths": [
            "Наведено конкретні регуляторні акти з датами: Постанова НБУ № 80 (25.07.2025) і Закон 1591-IX (30.06.2021).",
            "Чітко розрізнено два типи API-сервісів (AIS і PIS) та наведено їхні абревіатури.",
            "Надано нумерований перелік кроків ліцензування TPP — це конкретно й придатно до дії.",
            "Додано строк впровадження (Q1 2026), важливий для зацікавлених сторін.",
            "Наприкінці наведено джерела для перевірності.",
        ],
        "gaps": [],
        "revision_requests": [],
    },
    ensure_ascii=False,
    indent=2,
)

# -- REVISE case: incomplete RAG comparison report --------------------------
# -- REVISE кейс: неповний звіт-порівняння RAG -----------------------------


REVISE_FINDINGS_INPUT = """\
## Огляд RAG-технік

RAG означає Retrieval-Augmented Generation. Є різні способи це робити.
Наївний RAG використовує чанки. Sentence-window іноді кращий.

Деякі бенчмарки показують покращення. Це залежить від випадку використання.
"""

REVISE_CRITIQUE_OUTPUT = json.dumps(
    {
        "verdict": "REVISE",
        "is_fresh": False,
        "is_complete": False,
        "is_well_structured": False,
        "strengths": [
            "Правильно визначено дві retrieval-стратегії, які порівнюються.",
        ],
        "gaps": [
            "Немає конкретних бенчмарків або джерел — твердження на кшталт 'іноді кращий' не підкріплені.",
            "Не пояснено механіку наївного RAG (розмір чанка, векторна схожість, top-k retrieval).",
            "Не описано механіку sentence-window retrieval (розмір вікна, різниця між індексацією та retrieval).",
            "Немає порівняльної таблиці — користувач явно просив структуроване порівняння.",
            "Немає розділу рекомендацій — користувачу потрібні практичні поради щодо вибору підходу.",
            "Звіт надто короткий (< 100 слів) для технічного порівняння.",
        ],
        "revision_requests": [
            "Додай конкретні бенчмарки: процитуй покращення faithfulness на 8-15% для sentence-window проти наївного RAG у QA-бенчмарках.",
            "Поясни механіку наївного RAG: фіксований розмір чанка (256-512 токенів), dense vector cosine similarity, top-k selection.",
            "Поясни механіку sentence-window: sentence-level indexing, runtime window expansion (наприклад, ±2 речення), реалізація в LlamaIndex.",
            "Додай порівняльну таблицю з колонками: Вимір | Наївний RAG | Sentence-Window, рядки: Якість контексту, Затримка, Складність, Faithfulness.",
            "Додай розділ рекомендацій із поясненням, коли кожен підхід доречний (наприклад, faithfulness-critical vs speed-critical).",
        ],
    },
    ensure_ascii=False,
    indent=2,
)

# -- Additional REVISE case: off-domain / irrelevant findings ---------------
# -- Додатковий REVISE кейс: позадоменні / нерелевантні знахідки -----------


OFF_DOMAIN_FINDINGS_INPUT = """\
## Результати дослідження

Я знайшов інформацію про кулінарію та рецепти. Українська кухня дуже багата.
Борщ готують із буряком. Банкінг — це фінансова система.
"""

OFF_DOMAIN_CRITIQUE_OUTPUT = json.dumps(
    {
        "verdict": "REVISE",
        "is_fresh": False,
        "is_complete": False,
        "is_well_structured": False,
        "strengths": [],
        "gaps": [
            "Результат дослідження повністю поза доменом — обговорює кулінарію замість банківського регулювання або RAG-технік.",
            "Не надано регуляторної інформації, хоча ціль дослідження стосується українського банкінгу.",
            "Окреме нечітке речення про банкінг ('Банкінг — це фінансова система') не є інформативним.",
            "Не наведено джерел.",
            "Немає структури: крім заголовка, немає секцій або списків.",
        ],
        "revision_requests": [
            "Повністю прибери кулінарний контент — він не релевантний цілі дослідження.",
            "Досліди й опиши фактичну тему: українське банківське регулювання / відкритий банкінг або RAG retrieval-техніки, як зазначено в початковому плані.",
            "Використай knowledge_search з релевантними запитами (наприклад, 'відкритий банкінг НБУ' або 'RAG retrieval techniques'), щоб знайти контент у межах домену.",
            "Структуруй виправлений результат щонайменше у 3 іменовані Markdown-розділи, що покривають ключові аспекти цілі дослідження.",
        ],
    },
    ensure_ascii=False,
    indent=2,
)


# ---------------------------------------------------------------------------
# Metric definition
# Визначення метрики
# ---------------------------------------------------------------------------

critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "If verdict is APPROVE: check that 'strengths' lists at least 2 concrete, specific positive aspects "
        "(not vague praise like 'good report'). Check that 'revision_requests' is empty or near-empty.",
        "If verdict is REVISE: check that 'gaps' lists specific, identifiable problems (not vague like 'needs improvement'). "
        "Check that 'revision_requests' contains actionable, concrete instructions (what exactly to add/change/fix).",
        "Check that 'is_fresh', 'is_complete', 'is_well_structured' values are consistent with the verdict: "
        "APPROVE critiques should generally have True values; REVISE critiques should have at least one False.",
        "Check that the critique is internally consistent — APPROVE verdict with empty strengths is suspicious; "
        "REVISE verdict with empty revision_requests is unhelpful.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


# ---------------------------------------------------------------------------
# Tests
# Тести
# ---------------------------------------------------------------------------

def test_critic_approve_verdict_quality():
    # This test checks that an APPROVE critic verdict is specific, complete, and internally consistent.
    # Цей тест перевіряє, що вердикт критика APPROVE є конкретним, повним і внутрішньо узгодженим.
    """
    Сценарій APPROVE: критик оцінює добре структурований звіт про відкритий банкінг.

    Очікування:
    - verdict дорівнює APPROVE
    - strengths конкретні та посилаються на деталі (номери законів, дати, структуру)
    - gaps і revision_requests порожні (нічого виправляти)
    - is_fresh, is_complete, is_well_structured мають значення True
    - оцінка GEval Critique Quality >= 0.7
    """
    critique_data = json.loads(APPROVE_CRITIQUE_OUTPUT)
    assert critique_data["verdict"] == "APPROVE", "Статична критика має мати вердикт APPROVE"
    assert critique_data["is_fresh"] is True
    assert critique_data["is_complete"] is True
    assert critique_data["is_well_structured"] is True
    assert len(critique_data["strengths"]) >= 2, "Критика APPROVE має містити щонайменше 2 сильні сторони"
    assert len(critique_data["revision_requests"]) == 0, "Критика APPROVE не має містити запитів на виправлення"

    test_case = LLMTestCase(
        input=APPROVE_FINDINGS_INPUT,
        actual_output=APPROVE_CRITIQUE_OUTPUT,
    )
    assert_test(test_case, [critique_quality])


def test_critic_revise_verdict_quality():
    # This test checks that a REVISE critic verdict contains specific gaps and actionable revision requests.
    # Цей тест перевіряє, що вердикт критика REVISE містить конкретні прогалини та дієві запити на виправлення.
    """
    Сценарій REVISE: критик оцінює нечіткий і неповний звіт-порівняння RAG.

    Очікування:
    - verdict дорівнює REVISE
    - gaps описують конкретний відсутній контент (бенчмарки, механіку, таблицю)
    - revision_requests є дієвими та конкретними (що саме додати)
    - is_complete і is_well_structured мають значення False
    - оцінка GEval Critique Quality >= 0.7
    """
    critique_data = json.loads(REVISE_CRITIQUE_OUTPUT)
    assert critique_data["verdict"] == "REVISE", "Статична критика має мати вердикт REVISE"
    assert critique_data["is_complete"] is False
    assert critique_data["is_well_structured"] is False
    assert len(critique_data["gaps"]) >= 3, "Критика REVISE має визначити щонайменше 3 прогалини"
    assert len(critique_data["revision_requests"]) >= 3, (
        "Критика REVISE має містити щонайменше 3 дієві запити на виправлення"
    )

    test_case = LLMTestCase(
        input=REVISE_FINDINGS_INPUT,
        actual_output=REVISE_CRITIQUE_OUTPUT,
    )
    assert_test(test_case, [critique_quality])


def test_critic_off_domain_content_revise():
    # This test checks that off-domain research is rejected with a useful REVISE critique.
    # Цей тест перевіряє, що дослідження поза доменом відхиляється корисною критикою REVISE.
    """
    Сценарій REVISE: критик оцінює повністю позадоменний результат дослідження
    (про кулінарію замість банкінгу/RAG).

    Очікування:
    - verdict дорівнює REVISE
    - список strengths порожній (немає що зарахувати як сильну сторону)
    - gaps містять вказівку, що контент поза доменом
    - revision_requests спрямовують агента на коректні запити knowledge_search
    - оцінка GEval Critique Quality >= 0.7
    """
    critique_data = json.loads(OFF_DOMAIN_CRITIQUE_OUTPUT)
    assert critique_data["verdict"] == "REVISE"
    assert len(critique_data["strengths"]) == 0, "Позадоменна відповідь не має містити сильних сторін"
    assert len(critique_data["gaps"]) >= 3
    assert len(critique_data["revision_requests"]) >= 2

    # Verify at least one revision request mentions relevant search
    # Перевіряємо, що принаймні один запит на виправлення згадує релевантний пошук
    all_requests = " ".join(critique_data["revision_requests"]).lower()
    assert any(
        keyword in all_requests
        for keyword in ["knowledge_search", "search", "query", "research"]
    ), "Запити на виправлення мають спрямовувати агента до коректних пошукових інструментів"

    test_case = LLMTestCase(
        input=OFF_DOMAIN_FINDINGS_INPUT,
        actual_output=OFF_DOMAIN_CRITIQUE_OUTPUT,
    )
    assert_test(test_case, [critique_quality])


def test_critic_verdict_consistency():
    # This test checks structural consistency between critic verdicts, quality flags, and revision fields.
    # Цей тест перевіряє структурну узгодженість між вердиктами критика, прапорцями якості та полями виправлень.
    """
    Структурний інваріант: критика APPROVE має всі boolean-прапорці True,
    а критика REVISE має щонайменше один прапорець False.

    Також перевіряє, що критика APPROVE має непорожні strengths, а критика
    REVISE має непорожні revision_requests.
    """
    approve_data = json.loads(APPROVE_CRITIQUE_OUTPUT)
    assert approve_data["is_fresh"] is True
    assert approve_data["is_complete"] is True
    assert approve_data["is_well_structured"] is True
    assert len(approve_data["strengths"]) > 0

    for revise_output in [REVISE_CRITIQUE_OUTPUT, OFF_DOMAIN_CRITIQUE_OUTPUT]:
        revise_data = json.loads(revise_output)
        assert revise_data["verdict"] == "REVISE"
        flags = [revise_data["is_fresh"], revise_data["is_complete"], revise_data["is_well_structured"]]
        assert not all(flags), (
            "Критика REVISE має містити щонайменше один False-прапорець якості; "
            f"got: is_fresh={revise_data['is_fresh']}, is_complete={revise_data['is_complete']}, "
            f"is_well_structured={revise_data['is_well_structured']}"
        )
        assert len(revise_data["revision_requests"]) > 0, (
            "Критика REVISE має містити дієві revision_requests"
        )
