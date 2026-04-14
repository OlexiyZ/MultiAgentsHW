"""
Live DeepEval test for the real homework-lesson-8 agent pipeline.

Живий DeepEval-тест для реального pipeline агентів з homework-lesson-8.

This file intentionally does not run by default: it calls the real Planner,
Researcher, and Critic agents from homework-lesson-8, so it needs API keys,
network access, and the prepared lesson-8 knowledge base.

Цей файл навмисно не запускається за замовчуванням: він викликає реальних
агентів Planner, Researcher і Critic з homework-lesson-8, тому потребує
API-ключів, доступу до мережі та підготовленої бази знань lesson-8.
"""
from __future__ import annotations

import os

import pytest

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_PIPELINE") != "1"
    or not (os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")),
    reason="RUN_LIVE_PIPELINE=1 and OPENAI_API_KEY/API_KEY are required.",
)


LIVE_REQUEST = "Що таке відкритий банкінг в Україні та які API потрібні банкам?"

LIVE_EXPECTED_OUTPUT = (
    "Відповідь має українською пояснити відкритий банкінг в Україні, згадати "
    "НБУ, Закон 1591-IX або платіжне законодавство, API для Account Information "
    "Services / AIS та Payment Initiation Services / PIS, а також показати "
    "структуровані висновки з джерелами або застереженнями."
)


live_answer_relevancy = AnswerRelevancyMetric(
    threshold=0.4,
    model="gpt-4o-mini",
    include_reason=True,
)

live_correctness = GEval(
    name="Live Pipeline Correctness",
    evaluation_steps=[
        "Перевір, що actual_output відповідає українською на input.",
        "Перевір, що actual_output описує відкритий банкінг як API-доступ до "
        "банківських рахунків і платіжних дій за згодою клієнта.",
        "Перевір, що actual_output згадує ключові терміни: НБУ або законодавчу "
        "базу, AIS або Account Information Services, PIS або Payment Initiation Services.",
        "Не занижуй оцінку лише через інше формулювання, якщо зміст збігається.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.5,
)


def _build_research_request(user_request: str, plan_json: str) -> str:
    # Keep the real lesson-8 Researcher in charge, but pass the Planner output explicitly.
    # Залишаємо реального Researcher з lesson-8 відповідальним за роботу, але явно передаємо план Planner.
    return (
        "Виконай дослідження українською мовою для запиту користувача.\n\n"
        f"Запит користувача:\n{user_request}\n\n"
        f"План дослідження від Planner:\n{plan_json}\n\n"
        "Поверни структурований Markdown-звіт із ключовими висновками, доказами, "
        "джерелами та явними припущеннями або невизначеностями."
    )


def _build_critic_request(user_request: str, findings: str) -> str:
    return (
        "Оціни дослідження відносно початкового запиту користувача.\n\n"
        f"Початковий запит:\n{user_request}\n\n"
        f"Знахідки Researcher:\n{findings}"
    )


def test_live_lesson8_pipeline_planner_researcher_critic() -> None:
    # This test runs the real lesson-8 Planner -> Researcher -> Critic pipeline.
    # Цей тест запускає реальний pipeline lesson-8: Planner -> Researcher -> Critic.
    """
    Runs the real lesson-8 pipeline and evaluates the generated output with DeepEval.

    Запускає реальний pipeline lesson-8 і оцінює згенеровану відповідь через DeepEval.
    """
    from agent_metrics import get_agent_invoke_counts, reset_agent_invoke_counts
    from agents.critic import critique_findings
    from agents.planner import plan_request
    from agents.research import research_request

    reset_agent_invoke_counts()

    plan = plan_request(LIVE_REQUEST)
    assert plan.goal.strip(), "Planner має повернути непорожню мету."
    assert plan.search_queries, "Planner має повернути принаймні один пошуковий запит."

    findings = research_request(
        _build_research_request(
            user_request=LIVE_REQUEST,
            plan_json=plan.model_dump_json(indent=2),
        )
    )
    assert findings.strip(), "Researcher має повернути непорожні знахідки."

    critique = critique_findings(_build_critic_request(LIVE_REQUEST, findings))
    assert critique.verdict in {"APPROVE", "REVISE"}, (
        "Critic має повернути вердикт APPROVE або REVISE."
    )

    counts = get_agent_invoke_counts()
    assert counts.get("planner", 0) >= 1, "Живий pipeline має викликати Planner."
    assert counts.get("research", 0) >= 1, "Живий pipeline має викликати Researcher."
    assert counts.get("critic", 0) >= 1, "Живий pipeline має викликати Critic."

    actual_output = (
        f"{findings}\n\n"
        "## Critic verdict\n"
        f"{critique.model_dump_json(indent=2)}"
    )
    test_case = LLMTestCase(
        input=LIVE_REQUEST,
        actual_output=actual_output,
        expected_output=LIVE_EXPECTED_OUTPUT,
    )

    assert_test(test_case, [live_answer_relevancy, live_correctness])
