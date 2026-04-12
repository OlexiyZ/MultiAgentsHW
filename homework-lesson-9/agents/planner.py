from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PLANNER_SYSTEM_PROMPT, Settings, preview_for_log
from schemas import ResearchPlan

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


async def plan_request_json_async(request: str, tools: list[BaseTool]) -> str:
    """Run planner with MCP-backed LangChain tools; return ResearchPlan JSON string."""

    settings = Settings()
    llm = ChatOpenAI(
        api_key=settings.api_key.get_secret_value(),
        model=settings.planner_model_name,
        timeout=settings.request_timeout,
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=ResearchPlan,
    )

    logger.info("PlannerAgent: invoke start request=%s", preview_for_log(request))
    try:
        result = await agent.ainvoke({"messages": [("user", request)]})
    except Exception as exc:
        if _is_structured_output_validation_error(exc):
            logger.warning(
                "PlannerAgent: structured output validation failed; using fallback plan: %s",
                exc,
            )
            return json.dumps(_fallback_plan(request).model_dump(), ensure_ascii=False, indent=2)
        raise

    structured = result.get("structured_response")
    if isinstance(structured, ResearchPlan):
        out = _normalize_plan(structured, request)
    elif hasattr(structured, "model_dump"):
        out = _normalize_plan(
            ResearchPlan.model_validate(structured.model_dump()), request
        )
    elif isinstance(structured, dict):
        out = _normalize_plan(ResearchPlan.model_validate(structured), request)
    else:
        logger.warning("PlannerAgent: structured_response missing; using fallback plan")
        out = _fallback_plan(request)

    logger.info(
        "PlannerAgent: invoke end goal=%s queries=%d",
        preview_for_log(out.goal, 200),
        len(out.search_queries),
    )
    return json.dumps(out.model_dump(), ensure_ascii=False, indent=2)
