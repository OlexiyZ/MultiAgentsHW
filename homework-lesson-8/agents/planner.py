from __future__ import annotations

import json
import logging

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import PLANNER_SYSTEM_PROMPT, Settings, preview_for_log
from schemas import ResearchPlan
from tools import PLANNER_TOOLS

try:
    from langchain.agents.structured_output import StructuredOutputValidationError
except ImportError:  # pragma: no cover
    StructuredOutputValidationError = None  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)


def _fallback_plan(request: str) -> ResearchPlan:
    return ResearchPlan(
        goal=request,
        search_queries=[request],
        sources_to_check=["knowledge_base", "web"],
        output_format="Markdown report with headings and sources",
    )


def _normalize_plan(plan: ResearchPlan, request: str) -> ResearchPlan:
    goal = (plan.goal or "").strip()
    queries = [q.strip() for q in plan.search_queries if (q or "").strip()]
    sources = [s for s in plan.sources_to_check if s]
    fmt = (plan.output_format or "").strip()
    updates: dict = {}
    if not goal:
        updates["goal"] = request
    if not queries:
        updates["search_queries"] = [request]
    if not sources:
        updates["sources_to_check"] = ["knowledge_base", "web"]
    if not fmt:
        updates["output_format"] = "Markdown report with headings and sources"
    if updates:
        return plan.model_copy(update=updates)
    return plan


def _is_structured_output_validation_error(exc: BaseException) -> bool:
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in chain:
        chain.append(cur)
        cur = cur.__cause__
    for err in chain:
        if StructuredOutputValidationError is not None and isinstance(
            err, StructuredOutputValidationError
        ):
            return True
        if err.__class__.__name__ == "StructuredOutputValidationError":
            return True
    return False

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

    logger.info("PlannerAgent: invoke start request=%s", preview_for_log(request))
    try:
        result = planner_agent.invoke({"messages": [("user", request)]})
    except Exception as exc:
        if _is_structured_output_validation_error(exc):
            logger.warning(
                "PlannerAgent: structured output validation failed; using fallback plan: %s",
                exc,
            )
            return _fallback_plan(request)
        raise

    structured = result.get("structured_response")
    if isinstance(structured, ResearchPlan):
        out = _normalize_plan(structured, request)
        logger.info(
            "PlannerAgent: invoke end goal=%s queries=%d",
            preview_for_log(out.goal, 200),
            len(out.search_queries),
        )
        return out

    if hasattr(structured, "model_dump"):
        out = _normalize_plan(
            ResearchPlan.model_validate(structured.model_dump()), request
        )
        logger.info(
            "PlannerAgent: invoke end goal=%s queries=%d",
            preview_for_log(out.goal, 200),
            len(out.search_queries),
        )
        return out

    if isinstance(structured, dict):
        out = _normalize_plan(ResearchPlan.model_validate(structured), request)
        logger.info(
            "PlannerAgent: invoke end goal=%s queries=%d",
            preview_for_log(out.goal, 200),
            len(out.search_queries),
        )
        return out

    logger.warning("PlannerAgent: structured_response missing; using fallback plan")
    return _fallback_plan(request)


def plan_request_json(request: str) -> str:
    """Serializes the structured research plan to a pretty-printed JSON string.
    Серіалізує структурований план дослідження у JSON-рядок із форматуванням."""

    return json.dumps(plan_request(request).model_dump(), ensure_ascii=False, indent=2)
