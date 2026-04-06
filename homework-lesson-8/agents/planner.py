from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import PLANNER_SYSTEM_PROMPT, Settings
from schemas import ResearchPlan
from tools import PLANNER_TOOLS


settings = Settings()
llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
    timeout=settings.request_timeout,
)

planner_agent = create_agent(
    model=llm,
    tools=PLANNER_TOOLS,
    system_prompt=PLANNER_SYSTEM_PROMPT,
    response_format=ResearchPlan,
)


def plan_request(request: str) -> ResearchPlan:
    """Runs the planner agent on a user request and returns a validated ResearchPlan.
    Запускає планувальника на запиті користувача й повертає перевірений ResearchPlan."""

    result = planner_agent.invoke({"messages": [("user", request)]})
    structured = result.get("structured_response")
    if isinstance(structured, ResearchPlan):
        return structured

    if hasattr(structured, "model_dump"):
        return ResearchPlan.model_validate(structured.model_dump())

    if isinstance(structured, dict):
        return ResearchPlan.model_validate(structured)

    return ResearchPlan(
        goal=request,
        search_queries=[request],
        sources_to_check=["knowledge_base", "web"],
        output_format="Markdown report with headings and sources",
    )


def plan_request_json(request: str) -> str:
    """Serializes the structured research plan to a pretty-printed JSON string.
    Серіалізує структурований план дослідження у JSON-рядок із форматуванням."""

    return json.dumps(plan_request(request).model_dump(), ensure_ascii=False, indent=2)
