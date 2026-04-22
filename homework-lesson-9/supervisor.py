from __future__ import annotations

import asyncio
import logging

import httpx
from acp_sdk.client import Client as ACPClient
from acp_sdk.models import Run
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from config import SUPERVISOR_SYSTEM_PROMPT, Settings, preview_for_log
from mcp_utils import mcp_tools_to_langchain

logger = logging.getLogger(__name__)
settings = Settings()

llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.supervisor_model_name,
    timeout=settings.request_timeout,
)

_report_tool_cache: BaseTool | None = None


def _get_save_report_mcp_tool() -> BaseTool:
    global _report_tool_cache
    if _report_tool_cache is not None:
        return _report_tool_cache

    async def _load() -> BaseTool:
        tools = await mcp_tools_to_langchain(settings.report_mcp_url, server_key="report")
        for t in tools:
            if t.name == "save_report":
                return t
        raise RuntimeError("save_report not exposed by ReportMCP")

    _report_tool_cache = asyncio.run(_load())
    return _report_tool_cache


def _run_output_text(run: Run) -> str:
    if not run.output:
        return ""
    chunks: list[str] = []
    for message in run.output:
        for part in message.parts:
            if part.content:
                chunks.append(str(part.content))
    return "\n".join(chunks).strip()


def _acp_sync(agent_name: str, text: str) -> str:
    async def _go() -> str:
        # ACP agents run LLM + MCP (SearchMCP), so use an explicit longer timeout.
        timeout = httpx.Timeout(settings.acp_http_timeout)
        async with ACPClient(
            base_url=settings.acp_base_url,
            timeout=timeout,
        ) as client:
            run = await client.run_sync(text, agent=agent_name)
            run.raise_for_status()
            return _run_output_text(run)

    return asyncio.run(_go())


@tool
def plan(request: str, decision_reason: str = "") -> str:
    """Build a structured research plan via ACP planner agent."""

    logger.info(
        "Supervisor tool plan: reason=%s request=%s",
        preview_for_log(decision_reason, 200),
        preview_for_log(request),
    )
    out = _acp_sync("planner", request)
    logger.info("Supervisor tool plan: done json_chars=%d", len(out))
    return out


@tool
def research(request: str, decision_reason: str = "") -> str:
    """Run research via ACP researcher agent (MCP tools on server side)."""

    logger.info(
        "Supervisor tool research: reason=%s request=%s",
        preview_for_log(decision_reason, 200),
        preview_for_log(request),
    )
    out = _acp_sync("researcher", request)
    logger.info("Supervisor tool research: done output_chars=%d", len(out))
    return out


@tool
def critique(findings: str, decision_reason: str = "") -> str:
    """Critique findings via ACP critic agent; returns CritiqueResult JSON."""

    logger.info(
        "Supervisor tool critique: reason=%s findings_chars=%d preview=%s",
        preview_for_log(decision_reason, 200),
        len(findings),
        preview_for_log(findings, 300),
    )
    out = _acp_sync("critic", findings)
    logger.info("Supervisor tool critique: done json_chars=%d", len(out))
    return out


@tool
def save_report(filename: str, content: str) -> str:
    """Save report via ReportMCP (HITL gated on supervisor)."""

    logger.info(
        "Supervisor save_report MCP: filename=%r content_chars=%d",
        filename,
        len(content),
    )
    mcp_tool = _get_save_report_mcp_tool()
    return asyncio.run(mcp_tool.ainvoke({"filename": filename, "content": content}))


memory = InMemorySaver()

supervisor = create_agent(
    model=llm,
    tools=[plan, research, critique, save_report],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"save_report": True})],
    checkpointer=memory,
)


SUPERVISOR_CONFIG = {
    "configurable": {"thread_id": "lesson-9-supervisor-cli"},
    "recursion_limit": settings.max_iterations,
}
