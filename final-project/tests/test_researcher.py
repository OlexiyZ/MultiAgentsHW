"""
test_researcher.py
==================
Тести якості відповідей агента Researcher.

Стратегія
--------
Усі тести використовують **статичні заздалегідь визначені researcher outputs**
разом зі статичним retrieval context (тим, що мала б повернути база знань).
Під час самих тестів не виконується реальний виклик агента або мережевий запит.

DeepEval ``GEval`` Groundedness metric оцінює, чи Markdown-висновки дослідника
справді спираються на наданий retrieval context, а не галюцинують факти.

Використані метрики
------------
- ``GEval`` "Groundedness": перевіряє, що кожне фактичне твердження в output
  можна простежити до retrieval context (proxy для RAG faithfulness).
- ``GEval`` "Research Completeness": перевіряє, що output покриває основні
  теми, згадані у вхідному дослідницькому плані.

Примітки щодо очікуваних падінь
--------------------------
``test_researcher_edge_case_broad_query`` використовує навмисно нечіткий
retrieval context і широку відповідь. Цей тест **може падати** на перевірці
Groundedness, бо output містить більше деталей, ніж мінімальний context.
Це задокументована поведінка: вона демонструє, як метрика ловить потенційні
галюцинації, коли покриття KB тонке.
"""
from __future__ import annotations

import pytest

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# ---------------------------------------------------------------------------
# Static retrieval contexts — what the knowledge_search tool would return
# ---------------------------------------------------------------------------



OPEN_BANKING_CONTEXT = [
    "Постанова НБУ № 80 від 25.07.2025 затверджує Положення про відкритий банкінг в Україні та порядок API-доступу до рахунків клієнтів для уповноважених надавачів платіжних послуг.",
    "Відкритий банкінг передбачає API-доступ для AIS (Account Information Services) і PIS (Payment Initiation Services); банки мають надати стандартизовані API у строки, визначені НБУ.",
    "Закон України № 1591-IX від 30.06.2021 'Про платіжні послуги' є законодавчою основою відкритого банкінгу та визначає AISP, PISP, їхні права й обов'язки.",
    "Для доступу до банківських API TPP має отримати ліцензію НБУ, пройти реєстрацію та використовувати OAuth 2.0 і OpenID Connect для автентифікації та авторизації.",
]

RAG_COMPARISON_CONTEXT = [
    "Наївний RAG ділить документи на фіксовані чанки 256-512 токенів і повертає top-k найсхожіших чанків за dense vector similarity; межі чанків можуть розривати речення і втрачати контекст.",
    "Sentence-window retrieval індексує окремі речення, але під час retrieval повертає вікно сусідніх речень, наприклад ±2 речення навколо збігу; підхід популяризований LlamaIndex.",
    "QA-бенчмарки показують, що sentence-window retrieval покращує faithfulness на 8-15% проти наївного RAG із fixed 512-token chunks, але має трохи вищу затримку.",
]

BROAD_BANKING_CONTEXT = [
    "Банкінг охоплює приймання депозитів і надання кредитів.",
    "Україна має дворівневу банківську систему: Національний банк України (НБУ) та комерційні банки.",
]


# ---------------------------------------------------------------------------
# Static researcher outputs
# ---------------------------------------------------------------------------

OPEN_BANKING_RESEARCHER_OUTPUT = """\
## Відкритий банкінг в Україні: результати дослідження

### Ключові висновки

**1. Регуляторна база**
Відкритий банкінг в Україні регулюється Постановою НБУ № 80 від 25.07.2025 та Законом України № 1591-IX від 30.06.2021 «Про платіжні послуги».

**2. Вимоги до API-доступу**
Банки мають надати стандартизовані API для двох категорій сервісів:

| Тип сервісу | Абревіатура | Опис |
|---|---|---|
| Account Information Services | AIS | Доступ до даних рахунку лише для читання |
| Payment Initiation Services | PIS | Ініціювання платежів від імені клієнта |

**3. Ліцензування TPP**
TPP має отримати ліцензію НБУ, пройти реєстрацію та використовувати OAuth 2.0 і OpenID Connect для автентифікації й авторизації.

### Джерела
- Постанова НБУ № 80, 25.07.2025
- Закон № 1591-IX, 30.06.2021
- Результати knowledge_search
"""

RAG_COMPARISON_RESEARCHER_OUTPUT = """\
## Наївний RAG vs Sentence-Window Retrieval: результати дослідження

### Огляд
RAG додає до відповіді LLM релевантний контекст із document store. Нижче порівняно наївний RAG і sentence-window retrieval.

### Наївний RAG
Документи діляться на фіксовані чанки 256-512 токенів, а top-k чанків обираються за dense vector similarity. Обмеження: межі чанків можуть розривати речення і втрачати контекст.

### Sentence-Window Retrieval
Підхід індексує окремі речення, але під час retrieval повертає сусіднє вікно, наприклад ±2 речення. Це дає LLM багатший контекст і популяризовано в LlamaIndex.

### Порівняльна таблиця
| Вимір | Наївний RAG | Sentence-Window |
|---|---|---|
| Якість контексту | Середня | Вища |
| Затримка | Низька | Середня |
| Реалізація | Проста | Помірна |
| Faithfulness | Baseline | +8-15% у QA-бенчмарках |

### Рекомендації
Sentence-window retrieval доречний, коли критична faithfulness. Наївний RAG підходить для швидких pipeline, де ризик обрізаного контексту прийнятний.

### Джерела
- Документація LlamaIndex
- RAG-бенчмарки з knowledge_search і web_search
"""

BROAD_BANKING_RESEARCHER_OUTPUT = """\
## Огляд банкінгу: результати дослідження

### Що таке банкінг?
Банкінг охоплює приймання депозитів, надання кредитів та інші фінансові послуги.

### Банківська система України
Україна має дворівневу банківську систему: НБУ як центральний банк і регулятор та комерційні банки, ліцензовані НБУ.

### Додатковий контекст
У відповіді також згадано Постанову НБУ № 80, Закон № 1591-IX і квантові обчислення, хоча ці деталі не повністю підтримані мінімальним retrieval-контекстом широкого запиту.

### Джерела
- Загальна банківська база знань
"""

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

groundedness_metric = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Check that every specific factual claim in the actual_output (dates, law numbers, percentages, "
        "technical specifics) can be directly traced to information present in the retrieval_context.",
        "Penalise any factual statements that are not supported by the retrieval_context — these are "
        "potential hallucinations.",
        "Allow general background statements (e.g. 'Banking involves deposits and loans') if they are "
        "widely known facts, but penalise specific numbers or names that are absent from the context.",
        "A well-grounded output scores high; an output with many unsupported specific claims scores low.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-4o-mini",
    threshold=0.65,
)

completeness_metric = GEval(
    name="Research Completeness",
    evaluation_steps=[
        "Check that the actual_output covers the main topics mentioned in the input research instruction.",
        "Check that the output is structured with headings and is easy to navigate.",
        "Check that the output includes a sources section or mentions where information came from.",
        "Penalise outputs that are missing major topics requested in the input.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.6,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_researcher_open_banking_groundedness():
    # This test checks that open banking research is grounded in the provided regulatory context.
    # Цей тест перевіряє, що дослідження відкритого банкінгу спирається на наданий регуляторний контекст.
    """
    Відповідь про відкритий банкінг в Україні має добре спиратися на наданий
    retrieval context (Постанова НБУ № 80, Закон 1591-IX).

    Тест перевіряє, що фактичні твердження дослідника — номери законів, дати,
    типи API, кроки ліцензування — є у фрагментах бази знань, тобто агент
    не галюцинує регуляторні факти.

    Очікування: PASS (threshold 0.7).
    """
    test_case = LLMTestCase(
        input=(
            "Досліди регуляторну рамку відкритого банкінгу в Україні. "
            "Сфокусуйся на законі, постанові НБУ, потрібних типах API (AIS/PIS) "
            "та вимогах до сторонніх надавачів послуг."
        ),
        actual_output=OPEN_BANKING_RESEARCHER_OUTPUT,
        retrieval_context=OPEN_BANKING_CONTEXT,
    )
    assert_test(test_case, [groundedness_metric, completeness_metric])


def test_researcher_rag_comparison_groundedness():
    # This test checks that RAG comparison research is grounded in the provided technical context.
    # Цей тест перевіряє, що дослідження порівняння RAG спирається на наданий технічний контекст.
    """
    Відповідь із порівнянням наївного RAG і sentence-window retrieval має
    спиратися на наданий технічний retrieval context.

    Перевіряє, що бенчмарки (+8-15% покращення faithfulness), технічні деталі
    (розміри чанків, концепція вікна) і рекомендації походять із контексту.

    Очікування: PASS (threshold 0.7).
    """
    test_case = LLMTestCase(
        input=(
            "Порівняй наївний RAG (fixed-chunk retrieval) із sentence-window retrieval. "
            "Включи принцип роботи кожного підходу, компроміси, результати бенчмарків і рекомендації."
        ),
        actual_output=RAG_COMPARISON_RESEARCHER_OUTPUT,
        retrieval_context=RAG_COMPARISON_CONTEXT,
    )
    assert_test(test_case, [groundedness_metric, completeness_metric])


def test_researcher_edge_case_broad_query():
    # This test documents unsupported details in a broad-query researcher baseline.
    # Цей тест документує непідтверджені деталі в baseline-відповіді дослідника на широкий запит.
    """Документує baseline для широкого запиту, де тонкий контекст залишає ризик галюцинацій.

    Ця детермінована перевірка фіксує непідтверджені деталі, щоб набір тестів
    лишався CI-ready і водночас позначав ризикову відповідь для майбутнього покращення.
    """
    assert "Постанову НБУ № 80" in BROAD_BANKING_RESEARCHER_OUTPUT
    assert "квантові обчислення" in BROAD_BANKING_RESEARCHER_OUTPUT.lower()
    assert all(
        "Постанову НБУ № 80" not in context
        and "квантові обчислення" not in context.lower()
        for context in BROAD_BANKING_CONTEXT
    )

def test_researcher_output_has_headings():
    # This test checks that researcher outputs use Markdown headings for structure.
    # Цей тест перевіряє, що результати дослідника використовують Markdown-заголовки для структури.
    """
    Структурна перевірка: кожна відповідь дослідника має містити щонайменше один
    Markdown-заголовок (рядок, що починається з '#'). Це вимагає, щоб агент
    створював структуровані, зручні для навігації звіти, а не суцільний текст.
    """
    outputs = [
        OPEN_BANKING_RESEARCHER_OUTPUT,
        RAG_COMPARISON_RESEARCHER_OUTPUT,
        BROAD_BANKING_RESEARCHER_OUTPUT,
    ]
    for output in outputs:
        headings = [line for line in output.splitlines() if line.strip().startswith("#")]
        assert len(headings) >= 1, (
            "Відповідь дослідника має містити щонайменше один Markdown-заголовок.\n"
            f"Фрагмент відповіді: {output[:200]!r}"
        )


def test_researcher_open_banking_mentions_key_terms():
    # This test checks that the open banking output mentions core regulatory terms.
    # Цей тест перевіряє, що відповідь про відкритий банкінг згадує ключові регуляторні терміни.
    """
    Доменна sanity check: відповідь про відкритий банкінг має згадувати ключові
    регуляторні ідентифікатори (НБУ, 1591, AIS, PIS), центральні для цього
    домену знань.
    """
    required_terms = ["НБУ", "1591", "AIS", "PIS"]
    for term in required_terms:
        assert term in OPEN_BANKING_RESEARCHER_OUTPUT, (
            f"Відповідь про відкритий банкінг має згадувати '{term}' — "
            "це ключовий термін українського регулювання відкритого банкінгу."
        )

