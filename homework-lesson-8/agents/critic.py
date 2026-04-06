from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import CRITIC_SYSTEM_PROMPT, Settings
from schemas import CritiqueResult
from tools import RESEARCH_TOOLS


settings = Settings()
llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.model_name,
    timeout=settings.request_timeout,
)

critic_agent = create_agent(
    model=llm,
    tools=RESEARCH_TOOLS,
    system_prompt=CRITIC_SYSTEM_PROMPT,
    response_format=CritiqueResult,
)


def critique_findings(findings: str) -> CritiqueResult:
    """Runs the critic agent on research text and returns a validated CritiqueResult.
    Запускає критичного агента на тексті дослідження й повертає перевірений CritiqueResult."""

    result = critic_agent.invoke({"messages": [("user", findings)]})
    structured = result.get("structured_response")
    if isinstance(structured, CritiqueResult):
        return structured
    if hasattr(structured, "model_dump"):
        return CritiqueResult.model_validate(structured.model_dump())
    if isinstance(structured, dict):
        return CritiqueResult.model_validate(structured)
    return CritiqueResult(
        verdict="REVISE",
        is_fresh=False,
        is_complete=False,
        is_well_structured=False,
        strengths=[],
        gaps=["Critic could not produce structured response"],
        revision_requests=["Run additional verification and provide complete findings"],
    )


def critique_findings_json(findings: str) -> str:
    """Serializes the critic's structured verdict to a pretty-printed JSON string.
    Серіалізує структурований вердикт критика у JSON-рядок із форматуванням."""

    return json.dumps(
        critique_findings(findings).model_dump(),
        ensure_ascii=False,
        indent=2,
    )
