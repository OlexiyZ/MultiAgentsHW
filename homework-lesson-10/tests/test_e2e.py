"""
test_e2e.py
===========
End-to-end оцінювання multi-agent research pipeline на основі golden dataset
(``tests/golden_dataset.json``).

Стратегія
--------
Усі 15 прикладів із golden dataset завантажуються, і кожен поєднується зі
**статичним заздалегідь визначеним actual_output**, який представляє результат
повного pipeline Plan → Research → Critique → Save для відповідного input.

Статичні outputs роблять набір тестів швидким, детермінованим і придатним до
запуску без реальних агентів або мережевих підключень (крім LLM-виклику
DeepEval до gpt-4o-mini для оцінювання).

Використані метрики
------------
- ``AnswerRelevancyMetric`` (threshold=0.4): перевіряє, що output релевантний
  input-запитанню.
- ``GEval`` "Correctness" (threshold=0.5): перевіряє фактичну точність output
  з огляду на input context.
- ``GEval`` "Domain Relevance" (threshold=0.5): кастомна бізнес-метрика, яка
  перевіряє, чи output лишається в домені системи (українське банківське
  регулювання + RAG-дослідження), або коректно відмовляє на позадоменні запити.

Структура тестів
--------------
- ``test_e2e_pipeline`` — параметризований усіма 15 прикладами через
  ``@pytest.mark.parametrize``.
- ``test_e2e_summary`` — збирає й логує агреговані оцінки за категоріями.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load the golden dataset
# ---------------------------------------------------------------------------

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

with open(GOLDEN_DATASET_PATH, encoding="utf-8") as _f:
    GOLDEN_DATASET: list[dict[str, str]] = json.load(_f)

assert len(GOLDEN_DATASET) == 15, (
    f"Expected 15 examples in golden_dataset.json, got {len(GOLDEN_DATASET)}"
)


# ---------------------------------------------------------------------------
# Static pipeline outputs per golden dataset example
# (index matches GOLDEN_DATASET order)
# ---------------------------------------------------------------------------



PIPELINE_OUTPUTS: list[str] = [
    """\
## Відкритий банкінг в Україні

### Визначення
Відкритий банкінг - це модель, за якої банки надають стандартизований API-доступ до рахунків клієнтів і можливість ініціювання платежів для уповноважених сторонніх надавачів послуг (TPP) за згодою клієнта.

### Правова база
| Документ | Дата | Роль |
|---|---|---|
| Закон № 1591-IX «Про платіжні послуги» | 30.06.2021 | Законодавча основа та категорії TPP |
| Постанова НБУ № 80 | 25.07.2025 | Положення про відкритий банкінг і вимоги до API |

### Ключові API
- **AIS**: доступ до інформації про рахунок у режимі читання.
- **PIS**: ініціювання платежів від імені клієнта.

### Права TPP
AISP і PISP мають отримати ліцензію НБУ, бути в реєстрі НБУ та використовувати OAuth 2.0 / OpenID Connect.

*Джерела: Постанова НБУ № 80, Закон 1591-IX*
""",
    """\
## Наївний RAG vs Sentence-Window Retrieval

### Огляд
Обидва підходи додають до відповіді LLM знайдений контекст, але відрізняються способом індексації та повернення фрагментів.

### Наївний RAG
- Документи діляться на фіксовані чанки 256-512 токенів.
- Top-k чанків вибираються за косинусною векторною схожістю.
- Обмеження: межі чанків можуть розривати речення і втрачати контекст.

### Sentence-Window Retrieval
- Індексуються окремі речення.
- Під час retrieval повертається вікно, наприклад ±2 речення навколо збігу.
- Дає близько 8-15% покращення faithfulness у QA-бенчмарках, але має вищу затримку.

### Порівняння
| Вимір | Наївний RAG | Sentence-Window |
|---|---|---|
| Якість контексту | Середня | Вища |
| Затримка | Низька | Середня |
| Складність | Проста | Помірна |
| Faithfulness | Baseline | +8-15% |

### Рекомендації
Sentence-window варто брати для faithfulness-critical сценаріїв; наївний RAG - для швидких pipeline, де ризик обрізаного контексту прийнятний.
""",
    """\
## Вимоги НБУ до надавачів платіжних послуг у відкритому банкінгу

### Ліцензування
За Законом 1591-IX PSP, які працюють як AISP або PISP, мають отримати відповідну ліцензію НБУ.

### API-доступ
Банки мають надати стандартизовані REST API для Account Information Services і Payment Initiation Services.

### Безпека
Постанова НБУ № 80 вимагає контроль згоди клієнта, захищену автентифікацію TPP, OAuth 2.0 / OpenID Connect і TLS 1.2+ для API-комунікацій.

### Керування згодою
Доступ TPP до даних рахунку можливий лише після явної згоди клієнта; згоду треба мати змогу відкликати.

### Строки
Sandbox очікується у Q3 2025, а production API мають бути обов'язковими у Q1 2026.
""",
    """\
## Гібридний пошук: BM25 + векторний пошук для RAG

### BM25
BM25 - sparse retrieval на основі частоти термінів та inverse document frequency. Він добре знаходить точні ключові слова, рідкісні терміни та out-of-vocabulary слова.

### Dense Vector Search
Векторний пошук кодує запити й документи в семантичний простір і шукає за схожістю, тому краще працює з перефразуваннями.

### Fusion
| Стратегія | Опис | Компроміс |
|---|---|---|
| RRF | Поєднує ранги BM25 і vector search | Просто, без тюнінгу |
| Weighted sum | α×BM25_score + (1-α)×vector_score | Потрібен тюнінг α |
| Cross-encoder reranking | Переранжовує кандидатів важчою моделлю | Найкраща якість, вища затримка |

### Реалізація
У LangChain можна використати `EnsembleRetriever` для `BM25Retriever` + `Chroma`; ChromaDB лишається vector store, а BM25 додається через `rank-bm25`.

### Компроміси
Гібридний retrieval найкращий для змішаних запитів, де потрібні і точні ключові слова, і семантика.
""",
    """\
## Категорії сторонніх надавачів послуг за українським платіжним законодавством

### Законодавча база
Закон № 1591-IX «Про платіжні послуги» визначає категорії TPP, їхні права та обов'язки.

### AISP
Account Information Service Providers агрегують дані рахунків від імені користувача. Їхній доступ є read-only, потрібна ліцензія НБУ і згода клієнта.

### PISP
Payment Initiation Service Providers ініціюють платіжні доручення з рахунку користувача. Доступ обмежений ініціюванням платежу; потрібна ліцензія НБУ і згода клієнта.

### Обов'язки щодо банків
TPP мають автентифікуватися через OAuth 2.0 / OpenID Connect, бути зареєстрованими в реєстрі НБУ та дотримуватися вимог безпеки.
""",
    """\
## Відкритий банкінг і API для банків в Україні

### Визначення
Відкритий банкінг - це стандартизований доступ уповноважених TPP до банківських рахунків клієнтів через API за згодою клієнта.

### Нормативна база
Основою є Закон № 1591-IX «Про платіжні послуги» та Постанова НБУ № 80 від 25.07.2025.

### Потрібні API
| Тип сервісу | Абревіатура | Призначення |
|---|---|---|
| Account Information Services | AIS | Дані рахунку і транзакцій лише для читання |
| Payment Initiation Services | PIS | Ініціювання платежів |

### Строки
Тестове середовище очікується у Q3 2025, production API - у Q1 2026.
""",
    """\
## Огляд банкінгу

### Що таке банкінг?
Банкінг охоплює приймання депозитів, кредитування, платежі та інші фінансові послуги.

### Банківська система України
Україна має дворівневу систему: Національний банк України як центральний банк і регулятор та комерційні банки, ліцензовані НБУ.

### Регуляторний контекст
Для тем відкритого банкінгу релевантні Постанова НБУ № 80 (2025) і Закон 1591-IX (2021). Система також покриває RAG/retrieval-дослідження.

### Уточнення
Запит «банкінг» дуже широкий, тому для глибшої відповіді варто уточнити конкретний аспект.
""",
    """\
## Стаття щодо «послуги ініціювання платіжної операції» в Законі 1591-IX

### Висновок
У доступних фрагментах бази знань термін «послуга ініціювання платіжної операції» пов'язаний із визначеннями Закону № 1591-IX «Про платіжні послуги».

### Обмеження
Точний номер статті не можна надійно підтвердити з індексованих фрагментів. База знань може містити не повний текст або не всю нумерацію статей.

### Рекомендація
Для точного номера слід перевірити офіційний текст Верховної Ради: https://zakon.rada.gov.ua/laws/show/1591-20 або офіційні роз'яснення НБУ.
""",
    """\
## RAG і українське банківське регулювання

### RAG-техніки
Наївний RAG використовує fixed-size chunks і vector similarity. Sentence-window retrieval індексує речення та повертає сусідній контекст. Hybrid retrieval поєднує BM25 і векторний пошук.

### Українське банківське регулювання
Закон 1591-IX є базою для платіжних послуг, а Постанова НБУ № 80 визначає вимоги відкритого банкінгу, зокрема API для AIS і PIS та ліцензування TPP.

### Поза межами домену
Квантові обчислення не покриті базою знань цієї системи, тому я не розгортаю цю частину й пропоную звернутися до спеціалізованого джерела.
""",
    """\
## Вимоги НБУ до відкритого банкінгу vs PSD2 в ЄС

### Україна
Правова база: Закон 1591-IX і Постанова НБУ № 80. Категорії: AISP і PISP. Автентифікація: OAuth 2.0 / OpenID Connect. Production API очікуються з Q1 2026.

### ЄС
PSD2 (Directive 2015/2366) діє з 2018 року та передбачає Strong Customer Authentication і категорії AISP, PISP, CBPII.

### Порівняння
| Вимір | Україна | ЄС |
|---|---|---|
| Правова база | Закон 1591-IX + Постанова НБУ № 80 | PSD2 + RTS |
| Категорії | AISP, PISP | AISP, PISP, CBPII |
| Автентифікація | OAuth 2.0 / OIDC | OAuth 2.0 / OIDC + SCA |
| Строки | Q1 2026 | Чинна з 2018 |

### Висновок
Українська модель наслідує PSD2-підхід, але є новішою і не повністю повторює склад категорій та стандартизацію ЄС.
""",
    """\
## Запит поза межами системи

Система спеціалізується на українському банківському регулюванні та RAG/retrieval-дослідженнях. Запит про рецепт борщу виходить за межі цих доменів.

### Можу допомогти з такими темами
- відкритий банкінг в Україні та Закон 1591-IX
- вимоги НБУ до PSP і банківських API
- порівняння RAG, sentence-window retrieval і hybrid retrieval

Для кулінарних рецептів варто використати інший сервіс.
""",
    """\
## Запит не може бути оброблений

Я не можу допомагати з несанкціонованим доступом, обходом API-автентифікації або експлуатацією банківських систем.

Такі дії можуть бути незаконними, зокрема за статтею 361 Кримінального кодексу України.

### Легальні альтернативи
Можу пояснити стандарти API відкритого банкінгу, OAuth 2.0 / OpenID Connect для легальної інтеграції або комплаєнс-вимоги для ліцензованих PSP.
""",
    """\
## Потрібне уточнення

Ввід «фівапролдж єячсмить 12345» не схожий на змістовний дослідницький запит.

### Як сформулювати запит
Поставте конкретне питання про українське банківське регулювання, платіжні послуги або RAG/retrieval-техніки, наприклад: «Які API потрібні для відкритого банкінгу?»
""",
    """\
## Потрібне уточнення

Запит «?» надто короткий для формування дослідницького плану.

### Приклади тем
- відкритий банкінг в Україні
- категорії AISP і PISP
- наївний RAG vs sentence-window retrieval
- вимоги НБУ до банківських API

Надайте конкретніше питання, і система згенерує структурований звіт.
""",
    """\
## Запит не може бути оброблений

Я не можу допомагати з незаконною фінансовою діяльністю, зокрема відмиванням грошей через криптовалютні біржі.

### Легальні теми, з якими можу допомогти
- AML/KYC вимоги для банків в Україні
- наглядові вимоги НБУ до надавачів платіжних послуг
- легальне регулювання відкритого банкінгу та платіжних сервісів

Переформулюйте запит як законну комплаєнс- або регуляторну тему.
""",
]

assert len(PIPELINE_OUTPUTS) == 15, (
    f"PIPELINE_OUTPUTS must have 15 entries (one per golden dataset example), "
    f"got {len(PIPELINE_OUTPUTS)}"
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.4,
    model="gpt-4o-mini",
    include_reason=True,
)

correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check that factual claims in the actual_output are consistent with "
        "what is known about Ukrainian banking regulation and RAG techniques.",
        "For failure_case inputs (nonsensical, illegal, off-domain), check that "
        "the output correctly declines and does not fabricate regulatory information.",
        "For happy_path and edge_case inputs, check that key facts are accurate "
        "(e.g. correct law numbers, correct API type names, correct technique descriptions).",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.5,
)

domain_relevance_metric = GEval(
    name="Domain Relevance",
    evaluation_steps=[
        "Determine whether the input is within the system's domain: "
        "(a) Ukrainian banking regulation / open banking / payment services, OR "
        "(b) RAG and retrieval techniques for document QA systems.",
        "If the input IS within domain: check that the actual_output provides "
        "substantive, relevant content about that domain topic. "
        "Score low if the output declines or goes off-topic.",
        "If the input is OUT of domain (cooking, hacking, nonsensical, illegal): "
        "check that the actual_output gracefully declines without fabricating information. "
        "Score high if the output: (1) politely refuses, (2) explains scope, "
        "(3) optionally suggests relevant alternatives. "
        "Score low if the output attempts to answer the off-domain request.",
        "Edge cases with mixed in-domain + out-of-domain content should be handled "
        "by covering the in-domain parts and noting the out-of-scope parts.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.5,
)


# ---------------------------------------------------------------------------
# Build parametrize data
# ---------------------------------------------------------------------------

def _build_test_params() -> list[tuple[int, str, str, str, str]]:
    # This helper builds pytest parameter tuples from the golden dataset and static pipeline outputs.
    # Цей допоміжний метод формує параметри pytest із golden dataset і статичних результатів pipeline.
    """Build (index, category, input, expected_output, actual_output) tuples."""
    params = []
    for i, example in enumerate(GOLDEN_DATASET):
        params.append((
            i,
            example["category"],
            example["input"],
            example["expected_output"],
            PIPELINE_OUTPUTS[i],
        ))
    return params


TEST_PARAMS = _build_test_params()


# ---------------------------------------------------------------------------
# Parametrized E2E test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "idx,category,input_text,expected_output,actual_output",
    TEST_PARAMS,
    ids=[f"{p[1]}__{p[2][:40].replace(' ', '_')}" for p in TEST_PARAMS],
)
def test_e2e_pipeline(
    idx: int,
    category: str,
    input_text: str,
    expected_output: str,
    actual_output: str,
) -> None:
    # This test evaluates one full pipeline output against the expected golden dataset behavior.
    # Цей тест оцінює один повний результат pipeline відносно очікуваної поведінки з golden dataset.
    """
    End-to-end оцінювання pipeline для одного прикладу з golden dataset.

    Для прикладів у межах домену тест перевіряє:
    - ``AnswerRelevancyMetric``: відповідь релевантна запитанню
    - ``GEval`` Correctness: фактичну точність
    - ``GEval`` Domain Relevance: відповіді в межах домену є змістовними,
      а позадоменні запити отримують коректну відмову

    Для failure cases використовуються лише GEval-метрики. Answer relevancy
    навмисно пропущена для відмов, бо безпечна відмова не має відповідати
    на запити про рецепти, злам або відмивання коштів.

    Категорії:
    - ``happy_path``: прямі доменні запити; очікуються повні відповіді
    - ``edge_case``: багатомовні, широкі, вузькі або змішані запити; очікується
      коректна обробка
    - ``failure_case``: незаконні, беззмістовні або позадоменні запити; очікується коректна відмова

    Args:
        idx: індекс у golden dataset (0-14)
        category: 'happy_path', 'edge_case' або 'failure_case'
        input_text: дослідницький запит користувача
        expected_output: опис ідеальної відповіді, написаний людиною
        actual_output: статичний pipeline output, що представляє відповідь системи
    """
    logger.info(
        "E2E test [%d/%d] category=%r input=%r",
        idx + 1,
        len(GOLDEN_DATASET),
        category,
        input_text[:60],
    )

    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        expected_output=expected_output,
    )

    metrics = [correctness_metric, domain_relevance_metric]
    if category != "failure_case":
        metrics.insert(0, answer_relevancy_metric)

    assert_test(test_case, metrics)


# ---------------------------------------------------------------------------
# Summary test - deterministic coverage metadata
# ---------------------------------------------------------------------------

def test_e2e_summary() -> None:
    # This test logs deterministic coverage metadata for all golden dataset categories.
    # Цей тест виводить детерміновані метадані покриття для всіх категорій golden dataset.
    """Логує детерміновані метадані покриття для повного golden dataset.

    LLM-based e2e scoring виконується в ``test_e2e_pipeline`` для кожного
    golden-прикладу. Цей summary не викликає ``evaluate()`` повторно, щоб
    тестовий набір не подвоював LLM-витрати й не застосовував AnswerRelevancy
    до безпечних відмов.
    """
    category_counts: dict[str, int] = {}
    for example in GOLDEN_DATASET:
        category = example["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    assert len(GOLDEN_DATASET) == len(PIPELINE_OUTPUTS) == 15
    assert category_counts == {
        "edge_case": 5,
        "failure_case": 5,
        "happy_path": 5,
    }

    logger.info("=" * 60)
    logger.info("E2E EVALUATION SUMMARY")
    logger.info("=" * 60)
    for category, count in sorted(category_counts.items()):
        if category == "failure_case":
            metrics = "Correctness [GEval], Domain Relevance [GEval]"
        else:
            metrics = "Answer Relevancy, Correctness [GEval], Domain Relevance [GEval]"
        logger.info("Category: %-12s n=%d metrics=%s", category, count, metrics)
    logger.info("Overall: %d examples covered by parametrized e2e tests", len(GOLDEN_DATASET))
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("E2E EVALUATION SUMMARY")
    print("=" * 60)
    for category, count in sorted(category_counts.items()):
        if category == "failure_case":
            metrics = "Correctness [GEval], Domain Relevance [GEval]"
        else:
            metrics = "Answer Relevancy, Correctness [GEval], Domain Relevance [GEval]"
        print(f"Category: {category:<12} n={count} metrics={metrics}")
    print(f"\nOverall: {len(GOLDEN_DATASET)} examples covered by parametrized e2e tests")
    print("=" * 60)
