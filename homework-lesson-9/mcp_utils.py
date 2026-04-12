"""Load MCP tools as LangChain BaseTool instances (lesson 9)."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


async def mcp_tools_to_langchain(
    mcp_http_url: str,
    server_key: str = "search",
) -> list[BaseTool]:
    """Connect to a remote FastMCP HTTP server and return LangChain tools."""

    client = MultiServerMCPClient(
        {server_key: {"url": mcp_http_url, "transport": "http"}}
    )
    tools: Sequence[BaseTool] = await client.get_tools()
    return list(tools)
