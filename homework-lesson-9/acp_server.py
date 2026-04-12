"""ACP server: planner, researcher, critic agents (each uses SearchMCP tools via LangChain)."""

from __future__ import annotations

import asyncio
import logging

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Server
from acp_sdk.server.context import Context
from acp_sdk.server.server import create_app
import uvicorn

from agents.critic import critique_findings_json_async
from agents.planner import plan_request_json_async
from agents.research import research_request_async
from config import Settings
from mcp_utils import mcp_tools_to_langchain

logger = logging.getLogger(__name__)
settings = Settings()
srv = Server()

_tools: list | None = None
_tools_lock = asyncio.Lock()


async def _get_search_tools():
    global _tools
    async with _tools_lock:
        if _tools is None:
            logger.info("ACP: loading LangChain tools from SearchMCP %s", settings.search_mcp_url)
            _tools = await mcp_tools_to_langchain(settings.search_mcp_url, server_key="search")
            logger.info("ACP: loaded %d tools from SearchMCP", len(_tools))
        return _tools


def _user_text(messages: list[Message]) -> str:
    chunks: list[str] = []
    for message in messages:
        for part in message.parts:
            if part.content:
                chunks.append(str(part.content))
    return "\n".join(chunks).strip()


@srv.agent(
    name="planner",
    description="Builds a structured ResearchPlan JSON using SearchMCP tools.",
)
async def planner(input: list[Message], context: Context) -> Message:
    text = _user_text(input)
    tools = await _get_search_tools()
    out = await plan_request_json_async(text, tools)
    return Message(role="agent", parts=[MessagePart(content=out)])


@srv.agent(
    name="researcher",
    description="Runs research and returns markdown findings using SearchMCP tools.",
)
async def researcher(input: list[Message], context: Context) -> Message:
    text = _user_text(input)
    tools = await _get_search_tools()
    out = await research_request_async(text, tools)
    return Message(role="agent", parts=[MessagePart(content=out)])


@srv.agent(
    name="critic",
    description="Returns CritiqueResult JSON using SearchMCP tools for verification.",
)
async def critic(input: list[Message], context: Context) -> Message:
    text = _user_text(input)
    tools = await _get_search_tools()
    out = await critique_findings_json_async(text, tools)
    return Message(role="agent", parts=[MessagePart(content=out)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app(*srv.agents, lifespan=srv.lifespan)
    uvicorn.run(app, host="0.0.0.0", port=settings.acp_port, log_level="info")
