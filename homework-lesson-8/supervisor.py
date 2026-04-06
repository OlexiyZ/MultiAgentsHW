from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agents.critic import critique_findings_json
from agents.planner import plan_request_json
from agents.research import research_request
from config import SUPERVISOR_SYSTEM_PROMPT, Settings
from tools import save_report


settings = Settings()
llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
    timeout=settings.request_timeout,
)


@tool
def plan(request: str) -> str:
    """Build a structured research plan for the request.
    Формує структурований план дослідження для запиту."""

    return plan_request_json(request)


@tool
def research(request: str) -> str:
    """Run research with web and knowledge tools.
    Виконує дослідження з веб-пошуком та пошуком у локальній базі знань."""

    return research_request(request)


@tool
def critique(findings: str) -> str:
    """Critique findings and return a structured verdict as JSON.
    Оцінює знахідки й повертає структурований вердикт у вигляді JSON."""

    return critique_findings_json(findings)


memory = InMemorySaver()

supervisor = create_agent(
    model=llm,
    tools=[plan, research, critique, save_report],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"save_report": True})],
    checkpointer=memory,
)


SUPERVISOR_CONFIG = {
    "configurable": {"thread_id": "lesson-8-supervisor-cli"},
    "recursion_limit": settings.max_iterations,
}
