from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import RESEARCH_SYSTEM_PROMPT, Settings
from tools import RESEARCH_TOOLS


settings = Settings()
llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
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


def research_request(request: str) -> str:
    """Runs the research agent and returns its final assistant text as findings.
    Запускає дослідницького агента й повертає фінальний текст асистента як знахідки."""

    result = research_agent.invoke({"messages": [("user", request)]})
    return _extract_last_ai_message(result.get("messages", []))
