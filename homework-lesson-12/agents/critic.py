from __future__ import annotations

import json
import logging

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# from agent_metrics import record_agent_invoke
from config import CRITIC_SYSTEM_PROMPT, Settings, preview_for_log
from schemas import CritiqueResult
from tracing import build_langchain_config, observe
from tools import RESEARCH_TOOLS


logger = logging.getLogger(__name__)

settings = Settings()
llm = ChatOpenAI(
    api_key=settings.api_key.get_secret_value(),
    model=settings.critic_model_name,
    timeout=settings.request_timeout,
)

critic_agent = create_agent(
    model=llm,
    tools=RESEARCH_TOOLS,
    system_prompt=CRITIC_SYSTEM_PROMPT,
    response_format=CritiqueResult,
)


@observe()
def critique_findings(findings: str) -> CritiqueResult:
    """Runs the critic agent on research text and returns a validated CritiqueResult.
    Запускає агента-критика на тексті дослідження й повертає перевірений CritiqueResult."""

    logger.info(
        "CriticAgent: invoke start findings_chars=%d preview=%s",
        len(findings),
        preview_for_log(findings, 300),
    )
    record_agent_invoke("critic")
    result = critic_agent.invoke(
        {"messages": [("user", findings)]},
        config=build_langchain_config(run_name="critic_agent"),
    )
    structured = result.get("structured_response")
    if isinstance(structured, CritiqueResult):
        logger.info(
            "CriticAgent: invoke end verdict=%s fresh=%s complete=%s structured=%s",
            structured.verdict,
            structured.is_fresh,
            structured.is_complete,
            structured.is_well_structured,
        )
        return structured
    if hasattr(structured, "model_dump"):
        out = CritiqueResult.model_validate(structured.model_dump())
        logger.info("CriticAgent: invoke end verdict=%s", out.verdict)
        return out
    if isinstance(structured, dict):
        out = CritiqueResult.model_validate(structured)
        logger.info("CriticAgent: invoke end verdict=%s", out.verdict)
        return out
    logger.warning("CriticAgent: structured_response missing; using fallback critique")
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
