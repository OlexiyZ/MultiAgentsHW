from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agent_metrics import record_supervisor_tool
from agents.critic import critique_findings_json
from agents.planner import plan_request_json
from agents.research import research_request
from config import SUPERVISOR_SYSTEM_PROMPT, Settings, preview_for_log
from tools import save_report


logger = logging.getLogger(__name__)

settings = Settings()
llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.supervisor_model_name,
    timeout=settings.request_timeout,
)


@tool
def plan(request: str, decision_reason: str = "") -> str:
    """Build a structured research plan for the request.
    Формує структурований план дослідження для запиту."""

    logger.info(
        "Supervisor tool plan: reason=%s request=%s",
        preview_for_log(decision_reason, 200),
        preview_for_log(request),
    )
    record_supervisor_tool("plan")
    out = plan_request_json(request)
    logger.info("Supervisor tool plan: done json_chars=%d", len(out))
    return out


@tool
def research(request: str, decision_reason: str = "") -> str:
    """Run research with web and knowledge tools.
    Виконує дослідження з веб-пошуком та пошуком у локальній базі знань."""

    logger.info(
        "Supervisor tool research: reason=%s request=%s",
        preview_for_log(decision_reason, 200),
        preview_for_log(request),
    )
    record_supervisor_tool("research")
    out = research_request(request)
    logger.info("Supervisor tool research: done output_chars=%d", len(out))
    return out


@tool
def critique(findings: str, decision_reason: str = "") -> str:
    """Critique findings and return a structured verdict as JSON.
    Оцінює знахідки й повертає структурований вердикт у вигляді JSON."""

    logger.info(
        "Supervisor tool critique: reason=%s findings_chars=%d preview=%s",
        preview_for_log(decision_reason, 200),
        len(findings),
        preview_for_log(findings, 300),
    )
    record_supervisor_tool("critique")
    out = critique_findings_json(findings)
    logger.info("Supervisor tool critique: done json_chars=%d", len(out))
    return out


memory = InMemorySaver()

supervisor = create_agent(
    model=llm,
    tools=[plan, research, critique, save_report],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"save_report": True})],
    checkpointer=memory,
)


SUPERVISOR_CONFIG = {
    "configurable": {"thread_id": "lesson-10-supervisor-cli"},
    "recursion_limit": settings.max_iterations,
}
