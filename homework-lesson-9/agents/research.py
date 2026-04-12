from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RESEARCH_SYSTEM_PROMPT, Settings, preview_for_log


logger = logging.getLogger(__name__)


def _extract_last_ai_message(messages: list) -> str:
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai" and getattr(message, "content", ""):
            return message.content
    return "No research findings generated."


async def research_request_async(request: str, tools: list[BaseTool]) -> str:
    settings = Settings()
    llm = ChatOpenAI(
        api_key=settings.api_key.get_secret_value(),
        model=settings.research_model_name,
        timeout=settings.request_timeout,
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )

    logger.info("ResearchAgent: invoke start request=%s", preview_for_log(request))
    result = await agent.ainvoke({"messages": [("user", request)]})
    text = _extract_last_ai_message(result.get("messages", []))
    logger.info("ResearchAgent: invoke end output_chars=%d", len(text))
    logger.debug("ResearchAgent: output preview=%s", preview_for_log(text, 800))
    return text
