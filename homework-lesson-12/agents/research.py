from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agent_metrics import record_agent_invoke
from config import RESEARCH_SYSTEM_PROMPT, Settings, preview_for_log
from tracing import build_langchain_config, observe
from tools import RESEARCH_TOOLS


logger = logging.getLogger(__name__)

settings = Settings()
llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.research_model_name,
    timeout=settings.request_timeout,
)

research_agent = create_agent(
    model=llm,
    tools=RESEARCH_TOOLS,
    system_prompt=RESEARCH_SYSTEM_PROMPT,
)


def _extract_last_ai_message(messages: list) -> str:
    """Returns the content of the last non-empty AI message or a fallback string.
    Повертає вміст останнього непорожнього повідомлення AI або запасний рядок."""

    for message in reversed(messages):
        if getattr(message, "type", "") == "ai" and getattr(message, "content", ""):
            return message.content
    return "No research findings generated."


@observe()
def research_request(request: str) -> str:
    """Runs the research agent and returns its final assistant text as findings.
    Запускає дослідницького агента й повертає фінальний текст асистента як знахідки."""

    logger.info("ResearchAgent: invoke start request=%s", preview_for_log(request))
    record_agent_invoke("research")
    result = research_agent.invoke(
        {"messages": [("user", request)]},
        config=build_langchain_config(run_name="research_agent"),
    )
    text = _extract_last_ai_message(result.get("messages", []))
    logger.info("ResearchAgent: invoke end output_chars=%d", len(text))
    logger.debug("ResearchAgent: output preview=%s", preview_for_log(text, 800))
    return text
